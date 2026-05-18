import os
import contextlib
import numpy as np
import json
import cv2
import pickle
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import argparse
import logging
import sys

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

@contextlib.contextmanager
def redirect_stdout_to_stderr():
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout


EMBEDDINGS_DIR = None

SIMILARITY_THRESHOLD = 0.6
HIGH_CONFIDENCE_THRESHOLD = 0.80
MEDIUM_CONFIDENCE_THRESHOLD = 0.70
MIN_FACE_CONFIDENCE = 0.1
MIN_FACE_SIZE = 3
MAX_FACES_TO_PROCESS = 200
DETECTION_SCALES = [0.6, 1.0, 1.4]

MODEL_CONFIG = {
    'name': 'buffalo_l',
    'root': '~/.insightface/models',
    'allowed_modules': ['detection', 'recognition'],
    'providers': ['CPUExecutionProvider'],
    'det_size': (640, 640),
    'det_thresh': MIN_FACE_CONFIDENCE
}

# ANSI color codes for colored output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def load_models():
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name=MODEL_CONFIG['name'],
                         root=MODEL_CONFIG['root'],
                         allowed_modules=MODEL_CONFIG['allowed_modules'],
                         providers=MODEL_CONFIG['providers'])
        app.prepare(ctx_id=0,
                   det_size=MODEL_CONFIG.get('det_size', (640, 640)),
                   det_thresh=MODEL_CONFIG.get('det_thresh', 0.2))
        return app
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def load_embeddings():
    logger.info("Loading embeddings...")
    embeddings = {}

    for filename in os.listdir(EMBEDDINGS_DIR):
        if filename.endswith(".npy"):
            try:
                base_name = os.path.splitext(filename)[0]  # GLM01812_0
                emb_path = os.path.join(EMBEDDINGS_DIR, filename)
                meta_path = os.path.join(EMBEDDINGS_DIR, f"{base_name}_meta.json")

                if not os.path.exists(meta_path):
                    continue

                # Load meta
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    original_file = meta.get("original_filename", None)

                if not original_file:
                    continue

                emb = np.load(emb_path)

                if emb.shape[0] != 512:
                    continue
                if np.isclose(np.linalg.norm(emb), 0.0):
                    continue

                emb = emb / np.linalg.norm(emb)

                if original_file in embeddings:
                    embeddings[original_file]['embeddings'].append(emb)
                else:
                    embeddings[original_file] = {
                        'embeddings': [emb],
                        'original_file': original_file
                    }

            except Exception as e:
                logger.info(f"{Colors.WARNING}⚠️  Error loading {filename}: {str(e)}{Colors.ENDC}")

    # Final processing
    final_embeddings = {}
    for original_file, data in embeddings.items():
        avg_emb = np.mean(data['embeddings'], axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)

        for i, emb in enumerate(data['embeddings']):
            final_embeddings[f"{original_file}_ver{i}"] = {
                'embedding': emb,
                'original_file': original_file,
                'version': i,
                'avg_embedding': avg_emb
            }

    logger.info(f"{len(final_embeddings)} embeddings loaded.")
    return final_embeddings


def apply_exif_orientation(image_path):
    try:
        img_pil = Image.open(image_path)
        exif = img_pil._getexif()
        ORIENTATION_TAG = 274
        if exif is not None:
            for tag, value in exif.items():
                if tag == ORIENTATION_TAG:
                    if value == 2:
                        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    elif value == 3:
                        img_pil = img_pil.transpose(Image.ROTATE_180)
                    elif value == 4:
                        img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
                    elif value == 5:
                        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
                    elif value == 6:
                        img_pil = img_pil.transpose(Image.ROTATE_270)
                    elif value == 7:
                        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
                    elif value == 8:
                        img_pil = img_pil.transpose(Image.ROTATE_90)
                    break
        img_cv2 = np.array(img_pil)
        if len(img_cv2.shape) == 3 and img_cv2.shape[2] == 3:
            img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
        return img_cv2
    except:
        return cv2.imread(image_path)

def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        limg = cv2.merge((clahe.apply(l), a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(enhanced, -1, kernel)
    except Exception:
        return image

def align_face(img: np.ndarray, face, iteration: int = 0) -> np.ndarray:
    try:
        if img is None or not hasattr(face, 'landmark_2d_106'):
            return img

        lm = face.landmark_2d_106
        if lm is None or len(lm) < 5:
            return img

        left_eye = np.mean(lm[33:42], axis=0)
        right_eye = np.mean(lm[87:96], axis=0)
        nose = lm[97]
        mouth_left = lm[52]
        mouth_right = lm[61]

        face_center = np.mean([left_eye, right_eye, nose, mouth_left, mouth_right], axis=0)
        face_width = np.linalg.norm(left_eye - right_eye) * 3.0
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

        desired_width = 256
        scale = desired_width / face_width

        M = cv2.getRotationMatrix2D(tuple(face_center), angle, scale)
        tX = img.shape[1] / 2 - face_center[0] * scale
        tY = img.shape[0] / 2 - face_center[1] * scale
        M[0, 2] += tX
        M[1, 2] += tY

        aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        x = max(0, int(face_center[0] * scale + tX - desired_width / 2))
        y = max(0, int(face_center[1] * scale + tY - desired_width / 2))
        w = min(desired_width, img.shape[1] - x)
        h = min(desired_width, img.shape[0] - y)

        if w <= 0 or h <= 0:
            return img

        cropped = aligned[y:y+h, x:x+w]
        if cropped is None or cropped.size == 0 or cropped.shape[0] < 20 or cropped.shape[1] < 20:
            return img

        return cropped

    except Exception:
        return img

def detect_faces_multi_scale(img: np.ndarray, app, scales: list = None) -> list:
    h, w = img.shape[:2]

    try:
        fast_faces = app.get(img)
        if fast_faces:
            valid = [f for f in fast_faces
                     if (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) >= MIN_FACE_SIZE * MIN_FACE_SIZE]
            if valid:
                return valid
    except Exception as e:
        logger.warning(f"Face detection error: {str(e)}")

    if scales is None:
        scales = DETECTION_SCALES

    all_faces = []

    for scale in scales:
        # Resize the image
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w < MIN_FACE_SIZE or new_h < MIN_FACE_SIZE:
            continue

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Perform face detection
        try:
            faces = app.get(resized)

            # Adjust the bounding boxes to the original size
            for face in faces:
                # Scale the bounding boxes
                scale_factor = 1.0 / scale
                face.bbox = (face.bbox * scale_factor).astype('int32')

                if hasattr(face, 'kps') and face.kps is not None:
                    face.kps = (face.kps * scale_factor).astype('int32')

                # Check the face size
                face_w = face.bbox[2] - face.bbox[0]
                face_h = face.bbox[3] - face.bbox[1]
                face_size = face_w * face_h

                if face_size >= (MIN_FACE_SIZE * MIN_FACE_SIZE):
                    all_faces.append(face)

        except Exception as e:
            logger.warning(f"Face detection error at scale {scale}: {str(e)}")

    return all_faces

def select_best_face(faces: list) -> tuple:
    if not faces:
        return None, -1

    best_face = None
    best_score = -1

    for face in faces:
        face_w = face.bbox[2] - face.bbox[0]
        face_h = face.bbox[3] - face.bbox[1]
        face_size = face_w * face_h
        aspect_ratio = min(face_w, face_h) / max(face_w, face_h) if max(face_w, face_h) > 0 else 0

        angle_score = 1.0
        if hasattr(face, 'kps') and face.kps is not None and len(face.kps) >= 2:
            dY = face.kps[1][1] - face.kps[0][1]
            dX = face.kps[1][0] - face.kps[0][0]
            angle = abs(np.degrees(np.arctan2(dY, dX)))
            angle_score = max(0, 1 - (min(angle, 180 - angle) / 30))

        score = (face_size * face.det_score * aspect_ratio * angle_score) ** 0.25

        if score > best_score:
            best_score = score
            best_face = face

    return best_face, best_score

def process_selfie(image_path: str, app) -> tuple:
    img = apply_exif_orientation(image_path)
    if img is None:
        return None, None

    # Downscale to 1024px max — detection and alignment both use this image so coordinates match
    _max_dim = 1024.0
    _h, _w = img.shape[:2]
    if max(_h, _w) > _max_dim:
        _scale = _max_dim / float(max(_h, _w))
        img = cv2.resize(img, (int(_w * _scale), int(_h * _scale)), interpolation=cv2.INTER_AREA)
        logger.info(f"[SPEED FIX] Image resized from {_w}x{_h} to {int(_w*_scale)}x{int(_h*_scale)}")

    # Stage 1: Face detection at different scales
    all_faces = detect_faces_multi_scale(img, app)

    if not all_faces:
        return None, None

    best_face, _ = select_best_face(all_faces)
    if best_face is None:
        return None, None

    aligned_face = align_face(img, best_face)
    if aligned_face is None:
        aligned_face = img

    try:
        fallback = best_face.embedding if hasattr(best_face, 'embedding') and best_face.embedding is not None else None
        embedding = augment_and_average_embedding(aligned_face, app, fallback_embedding=fallback)
        return embedding, img
    except Exception as e:
        logger.error(f"Error processing selfie: {str(e)}")
        return None, None

def augment_and_average_embedding(img, app, fallback_embedding=None):
    """Average embedding over 5 augmentations for a more robust query vector."""
    try:
        def _get_best_emb(image):
            try:
                faces = app.get(image)
                if not faces:
                    return None
                best = max(faces, key=lambda x: x.det_score)
                emb = getattr(best, 'embedding', None)
                if emb is None:
                    return None
                n = np.linalg.norm(emb)
                return (emb / n).astype(np.float32) if n > 0 else None
            except Exception:
                return None

        embeddings = []

        # 1. Original
        e = _get_best_emb(img)
        if e is not None:
            embeddings.append(e)

        # 2. Horizontal flip
        try:
            e = _get_best_emb(cv2.flip(img, 1))
            if e is not None:
                embeddings.append(e)
        except Exception:
            pass

        # 3. Brightness boost
        try:
            e = _get_best_emb(cv2.convertScaleAbs(img, alpha=1.0, beta=40))
            if e is not None:
                embeddings.append(e)
        except Exception:
            pass

        # 4-5. Small rotations
        h, w = img.shape[:2]
        for angle in (-5, 5):
            try:
                M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
                rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
                e = _get_best_emb(rot)
                if e is not None:
                    embeddings.append(e)
            except Exception:
                pass

        if embeddings:
            avg = np.mean(embeddings, axis=0)
            n = np.linalg.norm(avg)
            if n > 0:
                logger.info(f"[AUG] Averaged {len(embeddings)}/5 augmentations")
                return (avg / n).astype(np.float32)

        if fallback_embedding is not None:
            norm = np.linalg.norm(fallback_embedding)
            return (fallback_embedding / norm).astype(np.float32) if norm > 0 else None

        return None

    except Exception as e:
        logger.error(f"Embedding extraction error: {str(e)}")
        if fallback_embedding is not None:
            norm = np.linalg.norm(fallback_embedding)
            return (fallback_embedding / norm).astype(np.float32) if norm > 0 else None
        return None

def build_faiss_index(embeddings):
    if not embeddings:
        return None, []

    emb_dim = None
    for emb_data in embeddings.values():
        if 'embedding' in emb_data and emb_data['embedding'] is not None:
            emb_dim = emb_data['embedding'].shape[0]
            break

    if emb_dim is None:
        return None, []

    import faiss
    index = faiss.IndexFlatIP(emb_dim)
    all_embeddings = []
    valid_keys = []
    for filename, data in embeddings.items():
        if 'embedding' in data and data['embedding'] is not None:
            all_embeddings.append(data['embedding'].astype('float32'))
            valid_keys.append(filename)

    if not all_embeddings:
        return None, []

    index.add(np.vstack(all_embeddings))

    return index, valid_keys

def search_similar_faces(query_embedding, embeddings, top_k=200):
    if query_embedding is None or not embeddings:
        return []

    # Create the FAISS index
    index, valid_keys = build_faiss_index(embeddings)
    if index is None or not valid_keys:
        return []

    query_vector = query_embedding.astype('float32').reshape(1, -1)
    search_k = min(top_k * 3, len(valid_keys))
    similarities, indices = index.search(query_vector, search_k)

    results = []
    for sim, idx in zip(similarities[0], indices[0]):
        if idx < 0 or idx >= len(valid_keys):
            continue

        similarity = float((1.0 + sim) / 2.0)
        if similarity < SIMILARITY_THRESHOLD:
            continue

        filename = valid_keys[idx]
        data = embeddings[filename]
        target_emb = data['embedding']
        avg_emb = data.get('avg_embedding', target_emb)

        results.append({
            'filename': filename,
            'original_file': data['original_file'],
            'version': data.get('version', 0),
            'similarity': similarity,
            'distance': float(1.0 - similarity),
            'cos_sim': float(np.dot(query_embedding, target_emb)),
            'avg_cos_sim': float(np.dot(query_embedding, avg_emb)),
            'euclidean_score': float(np.linalg.norm(query_embedding - target_emb)),
            'avg_euclidean_score': float(np.linalg.norm(query_embedding - avg_emb)),
        })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]

def load_prebuilt_faiss(faiss_dir):
    if not faiss_dir:
        return None, None

    index_path = os.path.join(faiss_dir, "index.faiss")
    meta_path = os.path.join(faiss_dir, "index.pkl")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None, None

    try:
        import faiss
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        filenames = meta.get("filenames", [])
        logger.info(f"FAISS index loaded ({index.ntotal} vectors)")
        return index, filenames
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        return None, None


def _faiss_search(query_embedding, index, valid_keys, top_k=200):
    """Standard FAISS search — returns top matches above SIMILARITY_THRESHOLD."""
    query_vector = query_embedding.astype("float32").reshape(1, -1)
    search_k = min(top_k * 3, len(valid_keys))
    similarities, indices_arr = index.search(query_vector, search_k)
    results = []
    for sim, idx in zip(similarities[0], indices_arr[0]):
        if idx < 0 or idx >= len(valid_keys):
            continue
        similarity = float((1.0 + sim) / 2.0)
        if similarity < SIMILARITY_THRESHOLD:
            continue
        results.append({
            "filename": valid_keys[idx],
            "original_file": valid_keys[idx],
            "similarity": similarity,
            "cos_sim": float(sim),
            "euclidean_score": float(1.0 - similarity),
        })
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def load_groups_for_search(groups_path, faiss_filenames):
    """
    Build lookup maps from refined_groups.json for group-based search.

    Returns:
        photo_to_group: {"FAJ_1688.jpg": "group_3", ...}
        group_to_photos: {"group_3": ["FAJ_1688.jpg", "FAJ_2847.jpg", ...], ...}
    """
    if not groups_path or not os.path.exists(groups_path):
        return {}, {}

    try:
        with open(groups_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load groups file: {e}")
        return {}, {}

    # Build base → original_filename from FAISS index (handles any extension)
    base_to_original = {}
    for fname in faiss_filenames:
        base = os.path.splitext(fname)[0]
        base_to_original[base] = fname

    photo_to_group = {}
    group_to_photos = {}

    for group_id, face_ids in data.get("groups", {}).items():
        photos = []
        seen = set()
        for face_id in face_ids:
            # "FAJ_1688_0" → "FAJ_1688"
            base = "_".join(face_id.rsplit("_", 1)[:-1])
            original = base_to_original.get(base)
            if original and original not in seen:
                photos.append(original)
                seen.add(original)
                photo_to_group[original] = group_id
        if photos:
            group_to_photos[group_id] = photos

    logger.info(f"Groups loaded: {len(group_to_photos)} groups, {len(photo_to_group)} photos mapped")
    return photo_to_group, group_to_photos


def group_search(query_embedding, index, valid_keys, photo_to_group, group_to_photos,
                 candidate_threshold=0.5, top_k=200):
    """
    Group-based search:
      1. FAISS search with lower threshold to find candidate photos
      2. Vote for the best group based on similarity-weighted matches
      3. Return ALL photos from the winning group

    Solves the triathlon problem: running photos find the group,
    which also contains swimming/cycling photos.
    """
    query_vector = query_embedding.astype("float32").reshape(1, -1)
    search_k = min(top_k * 5, len(valid_keys))
    similarities, indices_arr = index.search(query_vector, search_k)

    group_scores = {}
    group_best_sim = {}

    for sim, idx in zip(similarities[0], indices_arr[0]):
        if idx < 0 or idx >= len(valid_keys):
            continue
        similarity = float((1.0 + sim) / 2.0)
        if similarity < candidate_threshold:
            break  # results are sorted descending

        filename = valid_keys[idx]
        group_id = photo_to_group.get(filename)
        if not group_id:
            continue

        group_scores[group_id] = group_scores.get(group_id, 0.0) + similarity
        if similarity > group_best_sim.get(group_id, 0.0):
            group_best_sim[group_id] = similarity

    if not group_scores:
        logger.info("[GROUP] No group match found, returning empty results")
        return []

    best_group = max(group_scores, key=lambda g: group_scores[g])
    best_sim = group_best_sim[best_group]
    group_photos = group_to_photos.get(best_group, [])

    logger.info(f"[GROUP] Best group: {best_group} | score={group_scores[best_group]:.3f} "
                f"| best_sim={best_sim:.3f} | photos={len(group_photos)}")

    results = []
    for photo in group_photos:
        results.append({
            "filename": photo,
            "original_file": photo,
            "similarity": best_sim,
            "cos_sim": float(2.0 * best_sim - 1.0),
            "euclidean_score": float(1.0 - best_sim),
        })
    return results

def get_similarity_color(similarity):
    """Returns color based on similarity score"""
    if similarity >= 0.7:
        return Colors.OKGREEN
    elif similarity >= 0.6:
        return Colors.OKCYAN
    elif similarity >= 0.5:
        return Colors.WARNING
    else:
        return Colors.FAIL

def get_similarity_comment(similarity):
    """Returns comment based on similarity score"""
    if similarity >= HIGH_CONFIDENCE_THRESHOLD:
        return f"   {Colors.BOLD}✅ High Confidence (Same person - 95%+){Colors.ENDC}"
    elif similarity >= MEDIUM_CONFIDENCE_THRESHOLD:
        return f"   {Colors.OKGREEN}✅ Medium Confidence (Likely same person - 80%+){Colors.ENDC}"
    elif similarity >= SIMILARITY_THRESHOLD:
        return f"   {Colors.WARNING}⚠️  Low Confidence (May be same person - 70%+){Colors.ENDC}"
    else:
        return f"   {Colors.FAIL}❌ Low Similarity (Different person){Colors.ENDC}"

def main():
    with redirect_stdout_to_stderr():
        parser = argparse.ArgumentParser(description='Face similarity search using FAISS')
        parser.add_argument('embeddings_dir', type=str, help='Path to the embeddings directory')
        parser.add_argument('selfie_path', type=str, help='Path to the selfie image')
        parser.add_argument('--faiss-dir', type=str, help='Path to prebuilt FAISS index directory')
        parser.add_argument('--groups', type=str, default=None,
                            help='Path to refined_groups.json for group-based search')
        parser.add_argument('--candidate-threshold', type=float, default=0.5,
                            help='FAISS candidate threshold for group voting (default: 0.5)')
        parser.add_argument('--json', action='store_true', default=True, help='Output results as JSON')
        args = parser.parse_args()

        if args.json:
            Colors.HEADER = Colors.OKBLUE = Colors.OKCYAN = Colors.OKGREEN = Colors.WARNING = Colors.FAIL = Colors.ENDC = Colors.BOLD = ''

        if not os.path.exists(args.embeddings_dir):
            logger.error(f"Embeddings directory not found: {args.embeddings_dir}")
            return

        if not os.path.isfile(args.selfie_path):
            logger.error(f"Selfie image not found: {args.selfie_path}")
            return

        global EMBEDDINGS_DIR
        EMBEDDINGS_DIR = args.embeddings_dir

        try:
            app = load_models()
            if not app:
                logger.error("Failed to load model")
                return

            query_embedding, _ = process_selfie(args.selfie_path, app)
            if query_embedding is None:
                logger.error("Failed to process selfie")
                return

            if args.faiss_dir:
                index, valid_keys = load_prebuilt_faiss(args.faiss_dir)
                if index is not None and valid_keys:
                    if args.groups:
                        photo_to_group, group_to_photos = load_groups_for_search(args.groups, valid_keys)
                        if photo_to_group:
                            results = group_search(
                                query_embedding, index, valid_keys,
                                photo_to_group, group_to_photos,
                                candidate_threshold=args.candidate_threshold,
                            )
                        else:
                            logger.warning("Groups file empty or unreadable, falling back to normal search")
                            results = _faiss_search(query_embedding, index, valid_keys)
                    else:
                        results = _faiss_search(query_embedding, index, valid_keys)
                else:
                    embeddings = load_embeddings()
                    if not embeddings:
                        logger.error("No embeddings found")
                        return
                    results = search_similar_faces(query_embedding, embeddings, top_k=200)
            else:
                embeddings = load_embeddings()
                if not embeddings:
                    logger.error("No embeddings found")
                    return
                results = search_similar_faces(query_embedding, embeddings, top_k=200)

            output = {
                "status": "success",
                "matches": [],
                "summary": {
                    "total_faces": len(results),
                    "unique_files": len(set(r['original_file'] for r in results if 'original_file' in r))
                }
            }

            # group_search bypasses SIMILARITY_THRESHOLD — all photos in the group are valid
            group_mode = bool(args.faiss_dir and args.groups)
            for result in results:
                if not isinstance(result, dict) or 'similarity' not in result:
                    continue
                if not group_mode and result['similarity'] < SIMILARITY_THRESHOLD:
                    continue
                output["matches"].append({
                    "filename": os.path.basename(result['original_file']),
                    "similarity": float(result['similarity']),
                    "cosine_similarity": float(result.get('cos_sim', 0)),
                    "euclidean_distance": float(result.get('euclidean_score', 0))
                })

            output["matches"].sort(key=lambda x: x["similarity"], reverse=True)
            return output

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    result = main()
    if not isinstance(result, dict):
        result = {
            "status": "error",
            "message": "Invalid or empty result from script"
        }
    print(json.dumps(result, ensure_ascii=False))
