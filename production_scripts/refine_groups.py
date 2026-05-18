 # -*- coding: utf-8 -*-
#
# Group refinement — merge split DBSCAN groups using clothing similarity + consecutive photo bonus
# stdout: final JSON only
# stderr: progress and error logs
#
# Strategy:
#   DBSCAN groups are kept intact (no members removed).
#   Small groups that likely represent the same person are merged using:
#     1. Face embedding similarity veto (centroid cosine similarity)
#     2. Clothing color similarity (HSV histogram, upper + lower body)
#     3. Consecutive filename bonus (FAJ_2847, FAJ_2848 → same camera burst)
#   Final merge score = face_sim × 0.7 + clothing_sim × 0.3

import os
import re
import sys
import json
import logging
import argparse
import numpy as np

from raceocr.embedding_store import iter_face_records

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("refine_groups")


def _emit_json_and_exit(data, code=0):
    # write JSON result to stdout so the caller (PHP) can parse it cleanly
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
    sys.stdout.write("done\n")
    sys.stdout.flush()
    sys.exit(code)


def load_face_embeddings(embeddings_dir):
    embeddings = {}
    for face_id, emb, _ in iter_face_records(embeddings_dir, include_embeddings=True):
        if emb is not None:
            embeddings[face_id] = emb.astype(np.float32)
    return embeddings


def group_face_centroid(face_ids, embeddings):
    # mean embedding vector for a group — represents the "average identity" of the cluster
    vectors = [embeddings[fid] for fid in face_ids if fid in embeddings]
    if not vectors:
        return None
    return np.mean(np.array(vectors), axis=0)


def load_clothing_features(clothing_dir):
    # load upper and lower clothing feature matrices produced by clothing_extractor.py
    # upper: always present; lower: only for faces where the lower body was visible in frame
    features_path = os.path.join(clothing_dir, "clothing_features.npy")
    lower_path = os.path.join(clothing_dir, "clothing_features_lower.npy")
    index_path = os.path.join(clothing_dir, "clothing_index.json")

    if not os.path.exists(features_path) or not os.path.exists(index_path):
        raise FileNotFoundError(f"Clothing features not found in {clothing_dir}. Run clothing_extractor.py first.")

    upper_feats = np.load(features_path).astype(np.float32)
    with open(index_path) as f:
        index = json.load(f)

    upper = {}
    lower = {}

    has_lower_file = os.path.exists(lower_path)
    lower_feats = np.load(lower_path).astype(np.float32) if has_lower_file else None

    for i, entry in enumerate(index):
        fid = entry["face_id"]
        upper[fid] = upper_feats[i]
        # only include lower features where the region was actually visible (not zero-padded)
        if has_lower_file and entry.get("has_lower", False):
            lower[fid] = lower_feats[i]

    return upper, lower


def load_groups(groups_path):
    with open(groups_path) as f:
        data = json.load(f)
    return data["groups"], data.get("noise", [])


def face_id_to_photo_base(face_id):
    # "FAJ_5195_0" → "FAJ_5195"  (strips the face index suffix)
    return "_".join(face_id.rsplit("_", 1)[:-1])


def cosine_similarity(a, b):
    # safe cosine similarity — eps in denominator prevents divide-by-zero on zero vectors
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def group_clothing_vector(face_ids, clothing):
    # mean clothing feature vector for a group
    vectors = [clothing[fid] for fid in face_ids if fid in clothing]
    if not vectors:
        return None
    return np.mean(np.array(vectors), axis=0)


def extract_photo_number(filename):
    # extract the trailing number from a filename: "FAJ_2847.jpg" → 2847
    # used to detect consecutive photos (same camera burst)
    base = os.path.splitext(filename)[0]
    m = re.search(r"(\d+)$", base)
    return int(m.group(1)) if m else None


def extract_photo_prefix(filename):
    # extract the non-numeric prefix: "FAJ_2847" → "FAJ_"
    base = os.path.splitext(filename)[0]
    return re.sub(r"\d+$", "", base)


def are_consecutive(face_ids_a, face_ids_b, gap=3):
    # check if any photo from group A and group B are within `gap` frames of each other
    # consecutive photos from the same camera burst strongly suggest the same person
    # e.g. FAJ_2852_0 and FAJ_2853_1 → consecutive (gap=1)
    for fa in face_ids_a:
        base_a = face_id_to_photo_base(fa)
        prefix_a = extract_photo_prefix(base_a)
        num_a = extract_photo_number(base_a)
        if num_a is None:
            continue
        for fb in face_ids_b:
            base_b = face_id_to_photo_base(fb)
            prefix_b = extract_photo_prefix(base_b)
            num_b = extract_photo_number(base_b)
            if num_b is None:
                continue
            if prefix_a == prefix_b and 1 <= abs(num_a - num_b) <= gap:
                return True
    return False


def should_merge(face_ids_a, face_ids_b, upper, lower, embeddings,
                 clothing_threshold=0.6, lower_veto_threshold=0.5,
                 max_group_size=5, face_veto_threshold=0.25):
    # merge decision logic — three gates, all must pass:
    #
    # Gate 1: both groups are small (≤ max_group_size) AND have consecutive photos
    #         large groups are already confident clusters, don't merge them
    #
    # Gate 2: face embedding veto — if face centroid similarity < face_veto_threshold
    #         the two groups are clearly different people → hard reject
    #
    # Gate 3: final score = face_sim × 0.7 + clothing_sim × 0.3 ≥ clothing_threshold
    #         clothing_sim = upper × 0.5 + lower × 0.5 (if lower visible) else upper only

    if len(face_ids_a) > max_group_size or len(face_ids_b) > max_group_size:
        return False, 0.0

    consecutive = are_consecutive(face_ids_a, face_ids_b)
    if not consecutive:
        return False, 0.0

    # gate 2: face embedding veto
    cent_a = group_face_centroid(face_ids_a, embeddings)
    cent_b = group_face_centroid(face_ids_b, embeddings)
    face_sim = 0.0
    if cent_a is not None and cent_b is not None:
        face_sim = cosine_similarity(cent_a, cent_b)
    if face_sim < face_veto_threshold:
        return False, 0.0  # faces are too different — definitely different people

    # gate 3: clothing similarity
    vec_a = group_clothing_vector(face_ids_a, upper)
    vec_b = group_clothing_vector(face_ids_b, upper)
    upper_sim = cosine_similarity(vec_a, vec_b) if vec_a is not None and vec_b is not None else 0.0

    lower_a = group_clothing_vector(face_ids_a, lower)
    lower_b = group_clothing_vector(face_ids_b, lower)
    if lower_a is not None and lower_b is not None:
        # combine upper + lower when both are visible — lower body is often more distinctive
        lower_sim = cosine_similarity(lower_a, lower_b)
        # veto: if lower body looks clearly different, block merge regardless of face/upper score
        if lower_sim < lower_veto_threshold:
            return False, 0.0
        clothing_sim = upper_sim * 0.5 + lower_sim * 0.5
    else:
        clothing_sim = upper_sim

    final_score = face_sim * 0.7 + clothing_sim * 0.3
    if final_score >= clothing_threshold:
        return True, final_score

    return False, final_score


def merge_all_groups(groups, upper, lower, embeddings, clothing_threshold=0.6, lower_veto_threshold=0.5, max_group_size=5):
    # Union-Find with an explicit root→members index.
    # Old approach: inner O(n_groups) scan to collect members per root → O(n_groups³) total.
    # New approach: maintain root_members dict, update on every union → O(n_groups²) total.
    group_keys = list(groups.keys())
    parent = {k: k for k in group_keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # root_members[root] = combined face_id list for that cluster
    root_members = {k: list(groups[k]) for k in group_keys}

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        # merge smaller into larger to keep root_members balanced
        if len(root_members[rx]) < len(root_members[ry]):
            rx, ry = ry, rx
        parent[ry] = rx
        # O(members) merge — no full group scan needed
        root_members[rx].extend(root_members.pop(ry))

    merge_count = 0
    for i in range(len(group_keys)):
        for j in range(i + 1, len(group_keys)):
            ka, kb = group_keys[i], group_keys[j]
            root_a, root_b = find(ka), find(kb)
            if root_a == root_b:
                continue

            members_a = root_members[root_a]
            members_b = root_members[root_b]

            do_merge, sim = should_merge(
                members_a, members_b, upper, lower, embeddings,
                clothing_threshold=clothing_threshold,
                lower_veto_threshold=lower_veto_threshold,
                max_group_size=max_group_size
            )
            if do_merge:
                union(root_a, root_b)
                merge_count += 1

    logger.info(f"Merged {merge_count} group pairs -> {len(root_members)} groups")
    return dict(root_members)


def main():
    parser = argparse.ArgumentParser(description="Refine face groups by merging split groups")
    parser.add_argument("--groups", required=True, help="Path to groups.json")
    parser.add_argument("--embeddings", required=True, help="Face embeddings folder")
    parser.add_argument("--clothing", required=True, help="Clothing features folder")
    parser.add_argument("--output", required=True, help="Output folder")
    # clothing_threshold: raise to require stronger evidence before merging (fewer false merges)
    parser.add_argument("--clothing-threshold", type=float, default=0.6,
                        help="Min combined score to merge two groups (default: 0.6)")
    # lower_veto_threshold: if lower body similarity is below this, block the merge even if upper matches
    parser.add_argument("--lower-veto-threshold", type=float, default=0.5,
                        help="Lower body similarity veto threshold (default: 0.5)")
    # max_group_size: only attempt to merge small groups — large groups are already correct
    parser.add_argument("--max-group-size", type=int, default=5,
                        help="Only merge groups smaller than this (default: 5)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    try:
        t0 = __import__("time").time()
        logger.info("Loading data...")
        groups, noise = load_groups(args.groups)
        upper, lower = load_clothing_features(args.clothing)
        embeddings = load_face_embeddings(args.embeddings)
        logger.info(f"Groups: {len(groups)} | Upper features: {len(upper)} | Lower features: {len(lower)} | Embeddings: {len(embeddings)}")

        refined = merge_all_groups(
            groups, upper, lower, embeddings,
            clothing_threshold=args.clothing_threshold,
            lower_veto_threshold=args.lower_veto_threshold,
            max_group_size=args.max_group_size
        )

        # sort final groups by size descending — largest groups first
        refined = dict(sorted(refined.items(), key=lambda x: len(x[1]), reverse=True))

        output = {"groups": refined, "noise": sorted(list(set(noise)))}
        out_path = os.path.join(args.output, "refined_groups.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        duration = round(__import__("time").time() - t0, 3)
        largest = len(list(refined.values())[0]) if refined else 0
        logger.info(f"Done. Groups: {len(refined)} | Largest: {largest} | Duration: {duration}s")

        _emit_json_and_exit({
            "status": "success",
            "groups_before": len(groups),
            "groups_after": len(refined),
            "largest_group": largest,
            "output": out_path,
            "duration_seconds": duration
        })

    except Exception as e:
        logger.error(str(e))
        _emit_json_and_exit({"status": "error", "error": str(e)}, 1)


if __name__ == "__main__":
    main()
