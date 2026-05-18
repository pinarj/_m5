import os
import sys
import time
import logging
import traceback
from flask import Flask, request, jsonify

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from search_faiss_production import (
    load_models,
    process_selfie,
    load_prebuilt_faiss,
    load_groups_for_search,
    group_search,
    _faiss_search,
    SIMILARITY_THRESHOLD,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("search_server")
logging.getLogger("werkzeug").setLevel(logging.ERROR)

app = Flask(__name__)

# faiss_dir → (index, valid_keys, mtime_at_load)
faiss_index_cache = {}

# groups_path → (photo_to_group, group_to_photos, mtime_at_load)
groups_cache = {}

logger.info("Loading face recognition models...")
face_app = load_models()
if not face_app:
    logger.error("Failed to load models!")
    sys.exit(1)
logger.info("Models loaded. Server ready.")


def get_faiss_index(faiss_dir):
    """Load FAISS index into memory. Auto-reloads if index.faiss is updated on disk."""
    index_file = os.path.join(faiss_dir, "index.faiss")

    try:
        current_mtime = os.path.getmtime(index_file)
    except OSError:
        logger.warning(f"index.faiss not found: {index_file}")
        return None, None

    cached = faiss_index_cache.get(faiss_dir)
    if cached is not None:
        index, valid_keys, cached_mtime = cached
        if current_mtime == cached_mtime:
            return index, valid_keys
        logger.info(f"FAISS index updated on disk, reloading: {faiss_dir}")

    index, valid_keys = load_prebuilt_faiss(faiss_dir)
    if index is None:
        return None, None

    faiss_index_cache[faiss_dir] = (index, valid_keys, current_mtime)
    logger.info(f"FAISS index cached: {index.ntotal} vectors")
    return index, valid_keys


def get_groups(groups_path, faiss_filenames):
    """Load and cache group maps. Auto-reloads if groups file changes on disk."""
    if not groups_path:
        return {}, {}
    try:
        current_mtime = os.path.getmtime(groups_path)
    except OSError:
        return {}, {}

    cached = groups_cache.get(groups_path)
    if cached is not None:
        p2g, g2p, cached_mtime = cached
        if current_mtime == cached_mtime:
            return p2g, g2p
        logger.info(f"Groups file updated on disk, reloading: {groups_path}")

    p2g, g2p = load_groups_for_search(groups_path, faiss_filenames)
    groups_cache[groups_path] = (p2g, g2p, current_mtime)
    return p2g, g2p


@app.route('/search', methods=['POST'])
def search_endpoint():
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No JSON data provided"}), 400

        selfie_path = data.get('selfie_path')
        faiss_dir = data.get('faiss_dir')
        groups_path = data.get('groups_path')          # optional — enables group-based search
        top_k = data.get('top_k', 200)
        threshold = data.get('threshold', SIMILARITY_THRESHOLD)
        candidate_threshold = data.get('candidate_threshold', 0.5)

        if not selfie_path or not os.path.exists(selfie_path):
            return jsonify({"status": "error", "message": f"Selfie path invalid: {selfie_path}"}), 400

        if not faiss_dir:
            return jsonify({"status": "error", "message": "faiss_dir is required"}), 400

        start_time = time.time()

        query_embedding, _ = process_selfie(selfie_path, face_app)
        if query_embedding is None:
            return jsonify({"status": "error", "message": "Failed to extract embedding from selfie"}), 400

        index, valid_keys = get_faiss_index(faiss_dir)
        if index is None or not valid_keys:
            return jsonify({"status": "error", "message": "FAISS index not found"}), 404

        if groups_path:
            photo_to_group, group_to_photos = get_groups(groups_path, valid_keys)
            if photo_to_group:
                results = group_search(
                    query_embedding, index, valid_keys,
                    photo_to_group, group_to_photos,
                    candidate_threshold=candidate_threshold,
                    top_k=top_k,
                )
            else:
                logger.warning("Groups file empty or unreadable, falling back to normal search")
                results = _faiss_search(query_embedding, index, valid_keys, top_k=top_k)
        else:
            results = _faiss_search(query_embedding, index, valid_keys, top_k=top_k)

        # For group search: threshold filter not applied (all group photos are valid)
        # For normal search: _faiss_search already applied SIMILARITY_THRESHOLD
        matches = []
        for r in results:
            matches.append({
                'filename': os.path.basename(r['filename']),
                'similarity': float(r['similarity']),
                'cosine_similarity': float(r.get('cos_sim', 0)),
                'euclidean_distance': float(r.get('euclidean_score', 0)),
            })

        total_time = time.time() - start_time
        logger.info(f"Search completed in {total_time:.2f}s. Found {len(matches)} matches.")

        return jsonify({
            "status": "success",
            "matches": matches,
            "summary": {
                "total_faces": len(matches),
                "unique_files": len(set(r['filename'] for r in matches))
            },
            "processing_time": f"{total_time:.2f}s"
        })

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/reload', methods=['POST'])
def reload_endpoint():
    """Clear FAISS and groups cache — next /search request will reload from disk."""
    data = request.json or {}
    faiss_dir = data.get('faiss_dir')
    if faiss_dir and faiss_dir in faiss_index_cache:
        del faiss_index_cache[faiss_dir]
        groups_cache.clear()
        logger.info(f"Cache cleared for: {faiss_dir}")
        return jsonify({"status": "ok", "message": f"Cache cleared for: {faiss_dir}"})
    elif not faiss_dir:
        faiss_index_cache.clear()
        groups_cache.clear()
        logger.info("All cache cleared")
        return jsonify({"status": "ok", "message": "All cache cleared"})
    return jsonify({"status": "ok", "message": "Key not in cache"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
