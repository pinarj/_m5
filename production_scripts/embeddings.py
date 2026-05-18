#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Face embedding generator — stdout: final JSON only | stderr: logs

import multiprocessing
import platform
_start_method = "fork" if platform.system() != "Windows" else "spawn"
multiprocessing.set_start_method(_start_method, force=True)

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
 
import warnings
warnings.filterwarnings(
    "ignore",
    message="Specified provider 'CUDAExecutionProvider' is not in available provider names",
    category=UserWarning,
    module="onnxruntime"
)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import json
import time
import psutil
import logging
import argparse
import numpy as np
from datetime import datetime
from PIL import Image, ExifTags, ImageEnhance
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from raceocr.embedding_store import processed_image_bases, write_face_bundle

import cv2
cv2.setNumThreads(1)

logger = logging.getLogger("embeddings")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False

_console = logging.StreamHandler(sys.stderr)
_console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_console)


def _emit_json_and_exit(summary: dict, exit_code: int) -> None:
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("done", flush=True)
    sys.exit(exit_code)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images and generate face embeddings (CPU, fork, shared model).")
    parser.add_argument("--input", "-i", type=str, default="dataset",
                        help="Input folder containing images (default: dataset)")
    parser.add_argument("--output", "-o", type=str, default="embeddings",
                        help="Output folder for embeddings (default: embeddings)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force reprocessing even if outputs exist")
    parser.add_argument("--max-workers", "-w", type=int, default=None,
                        help="Number of parallel processes (default: auto)")
    parser.add_argument("--batch-size", "-b", type=int, default=None,
                        help="Submit window size (default: 2 x workers)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU (CUDA) for inference — requires onnxruntime-gpu")
    parser.add_argument("--io-threads", type=int, default=4,
                        help="I/O + preprocessing threads for GPU pipeline (default: 4)")
    return parser.parse_args()




def adjust_brightness_contrast(image, brightness=0, contrast=0):
    if brightness != 0:
        shadow = brightness if brightness > 0 else 0
        highlight = 255 if brightness > 0 else 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        image = cv2.convertScaleAbs(image, alpha=alpha_b, beta=gamma_b)

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.convertScaleAbs(image, alpha=alpha_c, beta=gamma_c)

    return image


def enhance_image(img_bgr):
    try:
        if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr

        avg_brightness = float(np.mean(gray))

        if avg_brightness < 60:
            enhanced = adjust_brightness_contrast(
                img_bgr,
                brightness=min(100, int(100 - avg_brightness)),
                contrast=30
            )
            enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
            return enhanced

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.2)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)
        enhanced = np.array(pil_img)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=20)
        return cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
    except Exception:
        return img_bgr


def apply_exif_orientation(image_path):
    try:
        with Image.open(image_path) as image:
            if hasattr(image, "_getexif") and image._getexif() is not None:
                try:
                    exif_dict = image._getexif()
                    if exif_dict:
                        exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_dict.items() if k in ExifTags.TAGS}
                        orientation = exif.get("Orientation", 1)
                        if orientation == 2:
                            image = image.transpose(Image.FLIP_LEFT_RIGHT)
                        elif orientation == 3:
                            image = image.rotate(180, expand=True)
                        elif orientation == 4:
                            image = image.rotate(180, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                        elif orientation == 5:
                            image = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                        elif orientation == 6:
                            image = image.rotate(-90, expand=True)
                        elif orientation == 7:
                            image = image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                        elif orientation == 8:
                            image = image.rotate(90, expand=True)
                except Exception:
                    pass
            img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        return enhance_image(img_cv)
    except Exception:
        try:
            return cv2.imread(image_path)
        except Exception:
            return None


from insightface.app import FaceAnalysis

_IS_FORK = (platform.system() != "Windows")
APP_SHARED = None
APP_SHARED_HIGH = None


def _init_worker(ctx_id=-1):
    global APP_SHARED, APP_SHARED_HIGH
    import warnings as _w
    _w.filterwarnings("ignore", category=FutureWarning)
    _w.filterwarnings(
        "ignore",
        message="Specified provider 'CUDAExecutionProvider' is not in available provider names",
        category=UserWarning,
        module="onnxruntime"
    )
    if APP_SHARED is None:
        APP_SHARED = FaceAnalysis(name="buffalo_l")
        APP_SHARED.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=0.7)
    if APP_SHARED_HIGH is None:
        APP_SHARED_HIGH = FaceAnalysis(name="buffalo_l")
        APP_SHARED_HIGH.prepare(ctx_id=ctx_id, det_size=(1280, 1280), det_thresh=0.7)


def _process_image_worker(input_folder, filename):
    try:
        img_path = os.path.join(input_folder, filename)
        img = apply_exif_orientation(img_path)
        if img is None:
            return filename, None, f"Error: {filename} could not be loaded"

        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray_full.mean() < 30:
            return filename, [], None

        faces = APP_SHARED.get(img)
        if not faces:
            enhanced_img = enhance_image(img)
            faces = APP_SHARED.get(enhanced_img)
        if not faces:
            faces = APP_SHARED_HIGH.get(img)
        if not faces:
            faces = APP_SHARED_HIGH.get(enhance_image(img))

        if not faces:
            return filename, [], None

        out = []
        for i, face in enumerate(faces):
            emb = getattr(face, "embedding", None)
            if emb is None:
                continue

            bbox = face.bbox.astype(int)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            if w < 40 or h < 40:
                continue

            try:
                x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), bbox[2], bbox[3]
                crop = img[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                if cv2.Laplacian(gray, cv2.CV_64F).var() < 20:
                    continue
            except Exception:
                continue

            nrm = float(np.linalg.norm(emb))
            if nrm == 0.0:
                continue

            out.append((i, (emb / nrm).astype(np.float32), bbox.tolist()))

        return filename, out, None
    except Exception as e:
        return filename, None, f"Error: {str(e)}"


def _preprocess_image(input_folder, filename):
    try:
        img_path = os.path.join(input_folder, filename)
        img = apply_exif_orientation(img_path)
        if img is None:
            return filename, None, f"Error: {filename} could not be loaded"
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray_full.mean() < 30:
            return filename, None, None
        return filename, img, None
    except Exception as e:
        return filename, None, f"Error: {str(e)}"


def _run_inference(img):
    faces = APP_SHARED.get(img)
    if not faces:
        enhanced_img = enhance_image(img)
        faces = APP_SHARED.get(enhanced_img)
    if not faces:
        faces = APP_SHARED_HIGH.get(img)
    if not faces:
        faces = APP_SHARED_HIGH.get(enhance_image(img))
    if not faces:
        return []

    out = []
    for i, face in enumerate(faces):
        emb = getattr(face, "embedding", None)
        if emb is None:
            continue
        bbox = face.bbox.astype(int)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w < 40 or h < 40:
            continue
        try:
            x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), bbox[2], bbox[3]
            crop = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < 20:
                continue
        except Exception:
            continue
        nrm = float(np.linalg.norm(emb))
        if nrm == 0.0:
            continue
        out.append((i, (emb / nrm).astype(np.float32), bbox.tolist()))
    return out


def _run_gpu_pipeline(submit_list, input_folder, output_folder, num_io_threads=4):
    """GPU pipeline: parallel I/O preprocessing + single-thread GPU inference."""
    preprocess_q = queue.Queue(maxsize=64)
    _DONE = object()

    failed_files = []
    total = len(submit_list)

    def io_worker():
        try:
            with ThreadPoolExecutor(max_workers=num_io_threads) as pool:
                futures = {pool.submit(_preprocess_image, input_folder, f): f for f in submit_list}
                for future in as_completed(futures):
                    preprocess_q.put(future.result())
        except Exception as e:
            logger.error(f"I/O thread error: {e}")
        finally:
            preprocess_q.put(_DONE)

    io_thread = threading.Thread(target=io_worker, daemon=True)
    io_thread.start()

    processed = 0
    done = 0
    while True:
        item = preprocess_q.get()
        if item is _DONE:
            break
        filename, img, error = item
        if error:
            failed_files.append((filename, error))
            logger.error(error)
        elif img is not None:
            try:
                out = _run_inference(img)
                if out:
                    write_face_bundle(output_folder, filename, out)
            except Exception as e:
                failed_files.append((filename, str(e)))
                logger.error(f"Inference/save error {filename}: {e}")
        processed += 1
        done += 1
        if done % 500 == 0 or done == total:
            logger.info(f"Progress: {done}/{total} images processed")

    io_thread.join()
    return processed, failed_files




def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def main():
    args = parse_arguments()
    INPUT_FOLDER = args.input
    OUTPUT_FOLDER = args.output

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    cpu_count = os.cpu_count() or 1
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    max_by_ram = max(1, int(total_ram_gb // 2))
    auto_workers = min(cpu_count, max_by_ram)

    import onnxruntime as ort
    gpu_available = "CUDAExecutionProvider" in ort.get_available_providers()
    use_gpu = args.gpu or gpu_available
    ctx_id = 0 if use_gpu else -1
    device_label = f"GPU (ctx_id={ctx_id})" if use_gpu else "CPU"

    if not use_gpu:
        MAX_WORKERS = max(1, args.max_workers if args.max_workers else auto_workers)
        BATCH_SIZE = max(1, args.batch_size if args.batch_size else MAX_WORKERS * 2)

    if use_gpu or _IS_FORK:
        global APP_SHARED, APP_SHARED_HIGH
        APP_SHARED = FaceAnalysis(name="buffalo_l")
        APP_SHARED.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=0.7)
        APP_SHARED_HIGH = FaceAnalysis(name="buffalo_l")
        APP_SHARED_HIGH.prepare(ctx_id=ctx_id, det_size=(1280, 1280), det_thresh=0.7)

    started_at = datetime.now()
    t0 = time.time()
    logger.info(f"START embeddings started_at={started_at.isoformat(timespec='seconds')} device={device_label}")

    failed_files = []
    processed = 0
    skipped = 0

    try:
        image_files = [
            f for f in os.listdir(INPUT_FOLDER)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"))
        ]
    except FileNotFoundError:
        logger.error(f"Input folder not found: {INPUT_FOLDER}")
        finished_at = datetime.now()
        duration_s = round(time.time() - t0, 3)
        logger.info(
            "END embeddings finished_at=%s duration_s=%.3f status=%s processed=%d failed=%d",
            finished_at.isoformat(timespec="seconds"),
            duration_s,
            "error",
            0,
            0
        )
        _emit_json_and_exit({
            "status": "error",
            "total_images": 0,
            "submitted": 0,
            "skipped_existing": 0,
            "processed": 0,
            "failed": 0,
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_seconds": duration_s,
            "error_message": f"Input folder not found: {INPUT_FOLDER}",
        }, 1)

    if not image_files:
        logger.error(f"No image files found in '{INPUT_FOLDER}'")
        finished_at = datetime.now()
        duration_s = round(time.time() - t0, 3)
        logger.info(
            "END embeddings finished_at=%s duration_s=%.3f status=%s processed=%d failed=%d",
            finished_at.isoformat(timespec="seconds"),
            duration_s,
            "error",
            0,
            0
        )
        _emit_json_and_exit({
            "status": "error",
            "total_images": 0,
            "submitted": 0,
            "skipped_existing": 0,
            "processed": 0,
            "failed": 0,
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_seconds": duration_s,
            "error_message": f"No image files found in '{INPUT_FOLDER}'",
        }, 1)

    submit_list = []
    if not args.force:
        processed_bases = processed_image_bases(OUTPUT_FOLDER)
        for fname in image_files:
            base = os.path.splitext(fname)[0]
            if base in processed_bases:
                skipped += 1
                continue
            submit_list.append(fname)
    else:
        submit_list = image_files

    if use_gpu:
        processed, failed_files = _run_gpu_pipeline(
            submit_list, INPUT_FOLDER, OUTPUT_FOLDER,
            num_io_threads=args.io_threads
        )
    else:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_init_worker, initargs=(ctx_id,)) as executor:
            total_to_process = len(submit_list)
            processed_count = 0
            for chunk in chunks(submit_list, BATCH_SIZE):
                futures = {executor.submit(_process_image_worker, INPUT_FOLDER, fname): fname for fname in chunk}
                for future in as_completed(futures):
                    filename, embeddings, error = future.result()
                    base_filename = os.path.splitext(filename)[0]

                    if error:
                        logger.error(error)
                        failed_files.append((filename, error))
                    else:
                        if embeddings:
                            write_face_bundle(OUTPUT_FOLDER, filename, embeddings)

                    processed += 1
                    processed_count += 1
                    if processed_count % 500 == 0 or processed_count == total_to_process:
                        logger.info(f"Progress: {processed_count}/{total_to_process} images processed")

    finished_at = datetime.now()
    duration_s = round(time.time() - t0, 3)

    status = "error" if processed == 0 else ("partial" if len(failed_files) > 0 else "success")

    summary = {
        "status": status,
        "total_images": len(image_files),
        "submitted": len(submit_list),
        "skipped_existing": skipped,
        "processed": processed,
        "failed": len(failed_files),
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_seconds": duration_s,
    }

    logger.info(
        "END embeddings finished_at=%s duration_s=%.3f status=%s processed=%d failed=%d",
        finished_at.isoformat(timespec="seconds"),
        duration_s,
        summary["status"],
        summary["processed"],
        summary["failed"]
    )

    _emit_json_and_exit(summary, 0 if status != "error" else 1)


if __name__ == "__main__":
    main()
