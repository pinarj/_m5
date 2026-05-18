# -*- coding: utf-8 -*-
#
# Face clustering — FAISS kNN + Union-Find
# Replaces DBSCAN: scales to 1M+ faces, O(n log n) instead of O(n²)
# stdout: final JSON only
# stderr: progress and error logs

import os
import sys
import json
import logging
import argparse
import numpy as np
import faiss
from datetime import datetime
from sklearn.preprocessing import normalize
from raceocr.embedding_store import iter_face_records

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("group_faces")


def _emit_json_and_exit(data, code=0):
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
    sys.stdout.write("done\n")
    sys.stdout.flush()
    sys.exit(code)


def load_embeddings(embeddings_dir):
    embeddings = []
    face_keys = []
    photo_names = set()

    for face_id, emb, meta in iter_face_records(embeddings_dir, include_embeddings=True):
        if emb is None:
            continue
        embeddings.append(emb.astype(np.float32))
        face_keys.append(face_id)
        photo_names.add(meta.get("original_filename", ""))

    if not embeddings:
        raise FileNotFoundError(f"No embeddings found in {embeddings_dir}")

    logger.info(f"Loaded {len(embeddings)} embeddings from {len(photo_names)} unique photos.")
    return np.array(embeddings), face_keys


class UnionFind:
    """Fast Union-Find for grouping connected faces."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def cluster_embeddings(embeddings, eps=0.7, min_samples=2):
    """
    FAISS kNN + Union-Find clustering.

    Steps:
      1. Build a FAISS IndexFlatL2 on L2-normalized embeddings
      2. For each face, find its k nearest neighbours
      3. If distance < eps, connect them (union)
      4. Extract connected components → clusters
      5. Groups smaller than min_samples → noise (-1)

    Why this beats DBSCAN/HDBSCAN at scale:
      - FAISS kNN is O(n * k) with SIMD — handles 1M+ vectors in seconds
      - Union-Find is O(n * alpha(n)) ≈ O(n)
      - Total: O(n log n) vs O(n²) for sklearn DBSCAN
    """
    n = len(embeddings)
    normed = normalize(embeddings, norm="l2").astype(np.float32)

    # FAISS: each face searches its k nearest neighbours
    # k=10 is enough to capture all same-person faces in a group
    k = min(10, n)
    dim = normed.shape[1]

    # For small datasets: exact brute-force (IndexFlatL2)
    # For large datasets: approximate IVF (50x faster, negligible accuracy loss for clustering)
    if n < 10_000:
        index = faiss.IndexFlatL2(dim)
        index.add(normed)
    else:
        nlist = min(int(n ** 0.5), 4096)   # number of Voronoi cells
        nprobe = min(20, nlist)             # cells to search per query
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        index.train(normed)
        index.add(normed)
        index.nprobe = nprobe
        logger.info(f"IVF index: nlist={nlist}, nprobe={nprobe}")

    distances, indices = index.search(normed, k + 1)  # +1 because result[0] = self

    logger.info(f"FAISS kNN search done. Building clusters...")

    uf = UnionFind(n)
    for i in range(n):
        for j_pos in range(1, k + 1):          # skip result[0] = self
            j = indices[i, j_pos]
            if j < 0:
                continue
            dist = distances[i, j_pos]
            # L2 distance on unit vectors: dist=0 → identical, dist=2 → opposite
            # eps=0.7 in original DBSCAN maps to L2 distance ≈ 0.7 (same scale)
            if dist < eps:
                uf.union(i, j)

    # collect clusters from Union-Find roots
    from collections import defaultdict
    clusters = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    # assign labels: groups >= min_samples get a positive label, rest → noise (-1)
    labels = [-1] * n
    cluster_id = 0
    for members in clusters.values():
        if len(members) >= min_samples:
            for i in members:
                labels[i] = cluster_id
            cluster_id += 1

    n_clusters = cluster_id
    n_noise = labels.count(-1)
    logger.info(f"Clusters: {n_clusters} | Noise: {n_noise}")
    return labels


def build_groups(labels, face_keys):
    groups = {}
    noise = []

    for label, face_key in zip(labels, face_keys):
        if label == -1:
            noise.append(face_key)
        else:
            key = f"group_{label + 1}"
            if key not in groups:
                groups[key] = []
            groups[key].append(face_key)

    groups = dict(sorted(groups.items(), key=lambda x: len(x[1]), reverse=True))
    return groups, noise


def main():
    t0 = datetime.now()
    parser = argparse.ArgumentParser(description="Group faces by identity using FAISS kNN + Union-Find")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings folder")
    parser.add_argument("--output", required=True, help="Path to output folder")
    parser.add_argument("--eps", type=float, default=0.7, help="Distance threshold (default: 0.7)")
    parser.add_argument("--min-samples", type=int, default=2, help="Min faces per group (default: 2)")
    parser.add_argument("--workers", type=int, default=None, help="FAISS thread count (default: all cores)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    try:
        if args.workers:
            faiss.omp_set_num_threads(args.workers)

        embeddings, face_keys = load_embeddings(args.embeddings)
        labels = cluster_embeddings(embeddings, eps=args.eps, min_samples=args.min_samples)
        groups, noise = build_groups(labels, face_keys)

        output = {
            "groups": groups,
            "noise": sorted(list(set(noise)))
        }
        out_path = os.path.join(args.output, "groups.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        duration = round((datetime.now() - t0).total_seconds(), 3)
        logger.info(f"Done. Groups: {len(groups)} | Noise: {len(set(noise))} | Duration: {duration}s")

        _emit_json_and_exit({
            "status": "success",
            "groups": len(groups),
            "noise": len(set(noise)),
            "largest_group": len(list(groups.values())[0]) if groups else 0,
            "output": out_path,
            "duration_seconds": duration
        })

    except Exception as e:
        logger.error(str(e))
        _emit_json_and_exit({"status": "error", "error": str(e)}, 1)


if __name__ == "__main__":
    main()
