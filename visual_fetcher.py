import requests
import json
import re
from google import genai

# ─────────────────────────────────────────────
# PROMPT SCENE SPLITTER
# ─────────────────────────────────────────────
SCENE_PROMPT = """
Kamu adalah video scene analyzer untuk konten YouTube berbahasa Indonesia.

Tugas: Pecah teks voiceover berikut menjadi scenes untuk video.

Rules:
- Pecah per 1-2 kalimat yang memiliki tema visual SAMA
- Setiap scene wajib punya visual_query dalam Bahasa Inggris (3-6 kata, spesifik)
- visual_query harus cocok dicari di stock footage website
- mood pilih dari: optimistic, serious, hopeful, dramatic, calm, energetic, neutral

PENTING: Return HANYA JSON array valid. Tanpa penjelasan, tanpa markdown, tanpa backtick.

Format output:
[
  {
    "scene_id": 1,
    "text": "teks narasi scene ini",
    "visual_query": "specific english stock footage query",
    "mood": "mood"
  }
]

VOICEOVER:
{text}
"""

# ─────────────────────────────────────────────
# SCENE SPLITTER via Gemini
# ─────────────────────────────────────────────
def split_scenes(text, gemini_api_key):
    """Kirim voiceover ke Gemini, dapat JSON scenes + visual query per scene."""
    client = genai.Client(api_key=gemini_api_key)
    prompt = SCENE_PROMPT.format(text=text.strip())

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    raw = response.text.strip()

    # Bersihkan markdown backtick
    raw = re.sub(r'```json|```', '', raw).strip()

    # Ekstrak hanya bagian [ ... ] array JSON
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        raise ValueError(f"Gemini tidak return JSON array valid. Response: {raw[:200]}")

    clean_json = match.group(0).strip()
    scenes = json.loads(clean_json)
    return scenes


# ─────────────────────────────────────────────
# RELEVANCE SCORER
# ─────────────────────────────────────────────
STOP_WORDS = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'for', 'with', 'and', 'or', 'to', 'is', 'are'}

def score_relevance(visual_query, title="", tags="", description=""):
    """
    Hitung skor relevansi clip terhadap visual_query.
    Return float 0.0 - 1.0
    """
    query_words = set(visual_query.lower().split()) - STOP_WORDS
    if not query_words:
        return 0.0

    haystack = f"{title} {tags} {description}".lower()
    matched = sum(1 for w in query_words if w in haystack)
    return round(matched / len(query_words), 2)


# ─────────────────────────────────────────────
# PEXELS SEARCH
# ─────────────────────────────────────────────
def search_pexels(query, api_key, per_page=5):
    """Cari video di Pexels. Return list of clip dict."""
    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": per_page, "size": "medium"}

    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for video in data.get("videos", []):
        files = sorted(video.get("video_files", []),
                       key=lambda x: x.get("width", 0), reverse=True)
        download_url = files[0]["link"] if files else None
        thumbnail = video.get("image", "")
        tags_str = " ".join([t.get("title", "") for t in video.get("tags", [])])

        results.append({
            "source": "Pexels",
            "id": video.get("id"),
            "thumbnail": thumbnail,
            "download_url": download_url,
            "title": video.get("url", ""),
            "tags": tags_str,
            "description": "",
            "duration": video.get("duration", 0),
            "width": video.get("width", 0),
            "height": video.get("height", 0),
            "relevance_score": 0.0
        })
    return results


# ─────────────────────────────────────────────
# PIXABAY SEARCH
# ─────────────────────────────────────────────
def search_pixabay(query, api_key, per_page=5):
    """Cari video di Pixabay. Return list of clip dict."""
    url = "https://pixabay.com/api/videos/"
    params = {
        "key": api_key,
        "q": query,
        "per_page": per_page,
        "video_type": "film"
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for hit in data.get("hits", []):
        videos = hit.get("videos", {})
        medium = videos.get("medium") or videos.get("small") or videos.get("tiny") or {}
        download_url = medium.get("url", "")
        thumbnail = medium.get("thumbnail", "")
        tags_str = hit.get("tags", "")

        results.append({
            "source": "Pixabay",
            "id": hit.get("id"),
            "thumbnail": thumbnail,
            "download_url": download_url,
            "title": tags_str,
            "tags": tags_str,
            "description": "",
            "duration": hit.get("duration", 0),
            "width": medium.get("width", 0),
            "height": medium.get("height", 0),
            "relevance_score": 0.0
        })
    return results


# ─────────────────────────────────────────────
# FIND BEST VISUAL (dengan fallback + scoring)
# ─────────────────────────────────────────────
def find_best_visual(visual_query, pexels_key, pixabay_key, threshold=0.25):
    """
    Cari visual terbaik untuk 1 scene.
    Flow:
      1. Search Pexels → score semua hasil
      2. Kalau ada yang >= threshold → pakai
      3. Kalau tidak → fallback Pixabay → score
      4. Return clip dengan score tertinggi dari semua kandidat
    """
    all_candidates = []
    log = []

    # --- Step 1: Pexels ---
    try:
        pexels_results = search_pexels(visual_query, pexels_key)
        for clip in pexels_results:
            clip["relevance_score"] = score_relevance(
                visual_query, clip["title"], clip["tags"], clip["description"]
            )
            all_candidates.append(clip)
        log.append(f"Pexels: {len(pexels_results)} hasil ditemukan")
    except Exception as e:
        log.append(f"Pexels gagal: {str(e)[:60]}")

    # --- Step 2: Cek apakah Pexels sudah cukup ---
    pexels_best = max(
        [c for c in all_candidates if c["source"] == "Pexels"],
        key=lambda x: x["relevance_score"],
        default=None
    )

    if pexels_best and pexels_best["relevance_score"] >= threshold:
        log.append(f"✅ Pexels lolos threshold (score: {pexels_best['relevance_score']})")
        return pexels_best, log

    # --- Step 3: Fallback Pixabay ---
    log.append(f"Pexels kurang relevan → fallback ke Pixabay")
    try:
        pixabay_results = search_pixabay(visual_query, pixabay_key)
        for clip in pixabay_results:
            clip["relevance_score"] = score_relevance(
                visual_query, clip["title"], clip["tags"], clip["description"]
            )
            all_candidates.append(clip)
        log.append(f"Pixabay: {len(pixabay_results)} hasil ditemukan")
    except Exception as e:
        log.append(f"Pixabay gagal: {str(e)[:60]}")

    # --- Step 4: Ambil terbaik dari semua ---
    if all_candidates:
        best = max(all_candidates, key=lambda x: x["relevance_score"])
        log.append(f"✅ Best dari semua source: {best['source']} (score: {best['relevance_score']})")
        return best, log

    return None, log        })
    return results


# ─────────────────────────────────────────────
# PIXABAY SEARCH
# ─────────────────────────────────────────────
def search_pixabay(query, api_key, per_page=5):
    """Cari video di Pixabay. Return list of clip dict."""
    url = "https://pixabay.com/api/videos/"
    params = {
        "key": api_key,
        "q": query,
        "per_page": per_page,
        "video_type": "film"
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for hit in data.get("hits", []):
        videos = hit.get("videos", {})
        medium = videos.get("medium") or videos.get("small") or videos.get("tiny") or {}
        download_url = medium.get("url", "")
        thumbnail = medium.get("thumbnail", "")
        tags_str = hit.get("tags", "")

        results.append({
            "source": "Pixabay",
            "id": hit.get("id"),
            "thumbnail": thumbnail,
            "download_url": download_url,
            "title": tags_str,
            "tags": tags_str,
            "description": "",
            "duration": hit.get("duration", 0),
            "width": medium.get("width", 0),
            "height": medium.get("height", 0),
            "relevance_score": 0.0
        })
    return results


# ─────────────────────────────────────────────
# FIND BEST VISUAL (dengan fallback + scoring)
# ─────────────────────────────────────────────
def find_best_visual(visual_query, pexels_key, pixabay_key, threshold=0.25):
    """
    Cari visual terbaik untuk 1 scene.
    Flow:
      1. Search Pexels → score semua hasil
      2. Kalau ada yang >= threshold → pakai
      3. Kalau tidak → fallback Pixabay → score
      4. Return clip dengan score tertinggi dari semua kandidat
    """
    all_candidates = []
    log = []

    # --- Step 1: Pexels ---
    try:
        pexels_results = search_pexels(visual_query, pexels_key)
        for clip in pexels_results:
            clip["relevance_score"] = score_relevance(
                visual_query, clip["title"], clip["tags"], clip["description"]
            )
            all_candidates.append(clip)
        log.append(f"Pexels: {len(pexels_results)} hasil ditemukan")
    except Exception as e:
        log.append(f"Pexels gagal: {str(e)[:60]}")

    # --- Step 2: Cek apakah Pexels sudah cukup ---
    pexels_best = max(
        [c for c in all_candidates if c["source"] == "Pexels"],
        key=lambda x: x["relevance_score"],
        default=None
    )

    if pexels_best and pexels_best["relevance_score"] >= threshold:
        log.append(f"✅ Pexels lolos threshold (score: {pexels_best['relevance_score']})")
        return pexels_best, log

    # --- Step 3: Fallback Pixabay ---
    log.append(f"Pexels kurang relevan → fallback ke Pixabay")
    try:
        pixabay_results = search_pixabay(visual_query, pixabay_key)
        for clip in pixabay_results:
            clip["relevance_score"] = score_relevance(
                visual_query, clip["title"], clip["tags"], clip["description"]
            )
            all_candidates.append(clip)
        log.append(f"Pixabay: {len(pixabay_results)} hasil ditemukan")
    except Exception as e:
        log.append(f"Pixabay gagal: {str(e)[:60]}")

    # --- Step 4: Ambil terbaik dari semua ---
    if all_candidates:
        best = max(all_candidates, key=lambda x: x["relevance_score"])
        log.append(f"✅ Best dari semua source: {best['source']} (score: {best['relevance_score']})")
        return best, log

    return None, log
