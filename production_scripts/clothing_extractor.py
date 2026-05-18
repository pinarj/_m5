# -*- coding: utf-8 -*-
#
# Clothing feature extractor — upper and lower body from race photos
# stdout: final JSON only
# stderr: progress and error logs

import os
import sys
import json
import logging
import argparse
import numpy as np
import cv2
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.feature import local_binary_pattern

from raceocr.embedding_store import iter_face_records

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clothing_extractor")


def _emit_json_and_exit(data, code=0):
    # write JSON result to stdout so the caller (PHP) can parse it cleanly
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
    sys.stdout.write("done\n")
    sys.stdout.flush()
    sys.exit(code)


def extract_clothing_crop(img, bbox, upper_scale=1.5, lower_scale_start=1.5, lower_scale_end=4.0):
    # derive crop boundaries from the face bounding box
    # the clothing region is estimated relative to face height — no separate body detector needed
    x1, y1, x2, y2 = [int(v) for v in bbox]
    face_h = y2 - y1
    face_w = x2 - x1
    h, w = img.shape[:2]

    # horizontal extent: centered on the face, slightly narrower (0.8× face width each side)
    cx = (x1 + x2) // 2
    half_w = int(face_w * 0.8)
    bx1 = max(0, cx - half_w)
    bx2 = min(w, cx + half_w)

    # upper body: starts at bottom of face, extends 1.5× face height downward (chest/torso)
    ub_y1 = y2
    ub_y2 = y2 + int(face_h * upper_scale)
    if ub_y2 > h or ub_y2 <= ub_y1 or bx2 <= bx1:
        return None, None
    upper_crop = img[ub_y1:ub_y2, bx1:bx2]
    upper_crop = upper_crop if upper_crop.size > 0 else None

    # lower body: 1.5–4.0× face height below face bottom (waist/legs area)
    lb_y1 = y2 + int(face_h * lower_scale_start)
    lb_y2 = y2 + int(face_h * lower_scale_end)
    lower_crop = None
    # only extract lower body if at least 60% of the region is within the frame
    # — partially cropped lower regions produce unreliable color histograms
    if lb_y1 < h and bx2 > bx1:
        visible_h = min(lb_y2, h) - lb_y1
        total_h = lb_y2 - lb_y1
        if visible_h / total_h >= 0.6:
            lower_crop = img[lb_y1:min(lb_y2, h), bx1:bx2]
            if lower_crop.size == 0:
                lower_crop = None

    return upper_crop, lower_crop


def extract_clothing_features(crop, hsv_bins=(8, 8, 8), lbp_points=24, lbp_radius=3, lbp_bins=256):
    # --- HSV color histogram ---
    # HSV is more robust to lighting changes than RGB
    # 8 bins per channel → 512-element color descriptor
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv], [0, 1, 2], None, list(hsv_bins), [0, 180, 0, 256, 0, 256])
    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten().astype(np.float32)

    # --- LBP texture descriptor ---
    # Local Binary Pattern captures texture (fabric weave, patterns, logos)
    # this helps distinguish people wearing the same color but different clothing
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, lbp_points, lbp_radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=lbp_bins, range=(0, lbp_bins))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_norm = lbp_hist.sum()
    if lbp_norm > 0:
        lbp_hist /= lbp_norm

    # concatenate color + texture into a single feature vector
    return np.concatenate([hsv_hist, lbp_hist])


def load_embeddings_index(embeddings_dir):
    entries = []
    for face_id, _, meta in iter_face_records(embeddings_dir, include_embeddings=False):
        bbox = meta.get("bbox", None)
        entries.append((meta["original_filename"], face_id, bbox))
    return entries


def build_photo_lookup(photos_dir):
    # scan photos_dir once and build {filename: full_path} — called once in main
    # avoids recursive glob per photo (250k photos × glob = millions of disk ops)
    lookup = {}
    for root, _, files in os.walk(photos_dir):
        for f in files:
            lookup[f] = os.path.join(root, f)
    logger.info(f"Photo lookup built: {len(lookup)} files in {photos_dir}")
    return lookup


def process_photo(args_tuple):
    photo_path, filename, photo_entries, save_crops, crops_dir = args_tuple

    if not photo_path:
        return [], [], len(photo_entries)

    img = cv2.imread(photo_path)
    if img is None:
        return [], [], len(photo_entries)

    features = []
    index = []
    skipped = 0

    for face_id, bbox in photo_entries:
        if bbox is None:
            skipped += 1
            continue
        upper_crop, lower_crop = extract_clothing_crop(img, bbox)
        if upper_crop is None:
            skipped += 1
            continue
        upper_feat = extract_clothing_features(upper_crop)
        lower_feat = extract_clothing_features(lower_crop) if lower_crop is not None else None
        features.append((upper_feat, lower_feat))
        index.append({
            "face_id": face_id,
            "filename": filename,
            "has_lower": lower_crop is not None
        })
        # optional: save crop images to disk for visual inspection
        if save_crops and crops_dir:
            cv2.imwrite(os.path.join(crops_dir, f"{face_id}_upper.jpg"), upper_crop)
            if lower_crop is not None:
                cv2.imwrite(os.path.join(crops_dir, f"{face_id}_lower.jpg"), lower_crop)

    return features, index, skipped


def main():
    t0 = datetime.now()
    parser = argparse.ArgumentParser(description="Extract clothing features from race photos")
    parser.add_argument("--photos", required=True, help="Root folder containing original photos")
    parser.add_argument("--embeddings", required=True, help="Embeddings folder")
    parser.add_argument("--output", required=True, help="Output folder")
    # --save-crops: write upper/lower crop images to disk — useful for debugging crop boundaries
    parser.add_argument("--save-crops", action="store_true", help="Save crop images for visual inspection")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if args.save_crops:
        os.makedirs(os.path.join(args.output, "crops"), exist_ok=True)

    try:
        entries = load_embeddings_index(args.embeddings)
        logger.info(f"Processing {len(entries)} embeddings from {len(set(e[0] for e in entries))} photos...")

        no_bbox = sum(1 for e in entries if e[2] is None)
        if no_bbox > 0:
            logger.warning(f"{no_bbox} embeddings have no bbox — re-run embeddings.py to fix.")

        features = []
        index = []
        skipped = 0

        photo_to_entries = defaultdict(list)
        for filename, face_id, bbox in entries:
            photo_to_entries[filename].append((face_id, bbox))

        crops_dir = os.path.join(args.output, "crops") if args.save_crops else None
        photo_lookup = build_photo_lookup(args.photos)
        tasks = [
            (photo_lookup.get(filename), filename, photo_entries, args.save_crops, crops_dir)
            for filename, photo_entries in photo_to_entries.items()
        ]

        logger.info(f"Processing with {args.workers} workers...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_photo, task): task for task in tasks}
            done = 0
            for future in as_completed(futures):
                f_list, i_list, sk = future.result()
                features.extend(f_list)
                index.extend(i_list)
                skipped += sk
                done += 1
                if done % 50 == 0 or done == len(tasks):
                    logger.info(f"Progress: {done}/{len(tasks)} photos")

        if not features:
            _emit_json_and_exit({"status": "error", "error": "No features extracted. Check --photos path."}, 1)

        upper_feats = np.array([f[0] for f in features], dtype=np.float32)
        lower_feats_raw = [f[1] for f in features]
        has_lower = [v is not None for v in lower_feats_raw]

        # fill missing lower-body features with zeros so all rows have the same shape
        feat_dim = upper_feats.shape[1]
        lower_feats = np.array(
            [v if v is not None else np.zeros(feat_dim, dtype=np.float32) for v in lower_feats_raw],
            dtype=np.float32
        )

        # save upper and lower feature matrices separately
        # refine_groups.py reads these to combine with face similarity scores
        np.save(os.path.join(args.output, "clothing_features.npy"), upper_feats)
        np.save(os.path.join(args.output, "clothing_features_lower.npy"), lower_feats)

        # store has_lower flag in index so refine_groups.py knows which rows are real lower features
        for i, entry in enumerate(index):
            entry["has_lower"] = has_lower[i]

        with open(os.path.join(args.output, "clothing_index.json"), "w") as f:
            json.dump(index, f, indent=2)

        duration = round((datetime.now() - t0).total_seconds(), 3)
        lower_count = sum(has_lower)
        logger.info(f"Done. Features: {len(features)} | Lower visible: {lower_count} | Skipped: {skipped} | Duration: {duration}s")

        _emit_json_and_exit({
            "status": "success",
            "features_extracted": len(features),
            "lower_visible": lower_count,
            "skipped": skipped,
            "feature_dim": int(upper_feats.shape[1]),
            "output": args.output,
            "duration_seconds": duration
        })

    except Exception as e:
        logger.error(str(e))
        _emit_json_and_exit({"status": "error", "error": str(e)}, 1)


if __name__ == "__main__":
    main()
