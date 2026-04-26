import streamlit as st
import time
import io
import array
from google import genai
from google.genai import types
from visual_fetcher import split_scenes, find_best_visual

# ─────────────────────────────────────────────
# HELPERS - API KEYS
# ─────────────────────────────────────────────
def get_gemini_keys():
    keys = []
    for k in ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"]:
        try:
            val = st.secrets[k]
            if val:
                keys.append(val)
        except:
            pass
    return keys

def get_secret(key_name):
    try:
        return st.secrets[key_name]
    except:
        return None

# ─────────────────────────────────────────────
# HELPERS - AUDIO
# ─────────────────────────────────────────────
def change_speed_pcm(raw_pcm, speed):
    samples = array.array('h', raw_pcm)
    if speed == 1.0:
        return bytes(samples)
    new_length = int(len(samples) / speed)
    new_samples = array.array('h')
    for i in range(new_length):
        src_idx = int(i * speed)
        if src_idx < len(samples):
            new_samples.append(samples[src_idx])
    return bytes(new_samples)

def pcm_to_mp3(pcm_data, sample_rate=24000):
    import lameenc
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(2)
    mp3 = encoder.encode(pcm_data)
    mp3 += encoder.flush()
    return bytes(mp3)

def process_tts(text, voice_name, speed_value):
    KEYS = get_gemini_keys()
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    if not KEYS:
        st.error("🚨 API Key tidak ditemukan!")
        return None
    for i, key in enumerate(KEYS):
        try:
            status_placeholder.info(f"🔄 Menggunakan API Key #{i+1}...")
            progress_bar.progress(20)
            client = genai.Client(api_key=key)
            status_placeholder.info("🎙️ Meminta suara dari Google...")
            progress_bar.progress(50)
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                        )
                    )
                )
            )
            raw_pcm = response.candidates[0].content.parts[0].inline_data.data
            status_placeholder.info("🛠️ Convert ke MP3...")
            progress_bar.progress(80)
            adjusted_pcm = change_speed_pcm(raw_pcm, speed_value)
            mp3_bytes = pcm_to_mp3(adjusted_pcm)
            buf = io.BytesIO(mp3_bytes)
            progress_bar.progress(100)
            status_placeholder.success("✨ Selesai!")
            time.sleep(1)
            status_placeholder.empty()
            return buf
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                st.warning(f"⚠️ Key #{i+1} Limit. Mencoba Key berikutnya...")
                continue
            else:
                st.error(f"❌ Error: {e}")
                break
    return None

# ─────────────────────────────────────────────
# SCORE COLOR HELPER
# ─────────────────────────────────────────────
def score_color(score):
    if score >= 0.6:
        return "🟢"
    elif score >= 0.3:
        return "🟡"
    else:
        return "🔴"

# ─────────────────────────────────────────────
# APP CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Voiceover + Visual Pipeline", page_icon="🎬", layout="wide")
st.title("🎬 Voiceover & Visual Pipeline")

tab1, tab2, tab3 = st.tabs(["🎙️ Generate Voiceover", "🔗 Merge MP3", "🖼️ Visual Per Scene"])

# ═══════════════════════════════════════════════
# TAB 1 - VOICEOVER
# ═══════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 3], gap="large")
    with col1:
        st.subheader("⚙️ Pengaturan")
        voice_choice = st.selectbox("Pilih Karakter Suara", ["Puck", "Charon", "Kore", "Fenrir"])
        speed_label = st.radio("Kecepatan Bicara:", ["Pelan (Santai)", "Normal", "Cepat"], index=1)
        speed_map = {"Pelan (Santai)": 0.85, "Normal": 1.0, "Cepat": 1.15}
        selected_speed = speed_map[speed_label]
        file_name = st.text_input("Nama File Output", "hasil_suara.mp3")
    with col2:
        st.subheader("📝 Konten Teks")
        text_area = st.text_area("Masukkan teks narasi:", height=300, placeholder="Tempel naskah di sini...")
        if st.button("🚀 Generate Voiceover Now", use_container_width=True):
            if text_area.strip():
                buf = process_tts(text_area, voice_choice, speed_value=selected_speed)
                if buf:
                    st.divider()
                    st.audio(buf, format="audio/mpeg")
                    st.download_button("📥 Download MP3", buf, file_name=file_name, mime="audio/mpeg", use_container_width=True)
            else:
                st.warning("Silakan masukkan teks terlebih dahulu.")

# ═══════════════════════════════════════════════
# TAB 2 - MERGE MP3
# ═══════════════════════════════════════════════
with tab2:
    st.subheader("🔗 Gabungkan File MP3")
    st.caption("Upload beberapa file MP3, cek urutannya, lalu klik Merge.")
    uploaded = st.file_uploader(
        "Upload file MP3 (pilih semua sekaligus)",
        type=["mp3"],
        accept_multiple_files=True,
        key="merge_uploader"
    )
    if uploaded is not None and len(uploaded) > 0:
        st.success(f"✅ {len(uploaded)} file berhasil diupload!")
        st.divider()
        st.subheader("📋 Urutan Penggabungan:")
        for i, f in enumerate(uploaded):
            size_kb = len(f.getvalue()) / 1024
            st.write(f"**{i+1}.** `{f.name}` — {size_kb:.1f} KB")
        st.divider()
        merge_name = st.text_input("💾 Nama file hasil gabungan:", value="gabungan_voiceover.mp3", key="merge_name")
        if st.button("🔗 Merge Semua MP3", use_container_width=True, type="primary"):
            with st.spinner(f"Menggabungkan {len(uploaded)} file..."):
                combined = b""
                for f in uploaded:
                    f.seek(0)
                    combined += f.read()
                result_buf = io.BytesIO(combined)
            st.success(f"✨ Berhasil! {len(uploaded)} file digabungkan menjadi 1.")
            st.audio(result_buf, format="audio/mpeg")
            st.download_button(
                label=f"📥 Download {merge_name}",
                data=io.BytesIO(combined),
                file_name=merge_name,
                mime="audio/mpeg",
                use_container_width=True,
                type="primary"
            )
    else:
        st.info("👆 Upload minimal 2 file MP3 di atas untuk mulai menggabungkan.")

# ═══════════════════════════════════════════════
# TAB 3 - VISUAL PER SCENE
# ═══════════════════════════════════════════════
with tab3:
    st.subheader("🖼️ Cari Visual Otomatis Per Scene")
    st.caption("Paste voiceover teks → AI pecah jadi scenes → cari visual stock footage otomatis per scene.")

    # --- API Keys ---
    with st.expander("🔑 API Keys (isi jika belum di secrets)", expanded=False):
        col_k1, col_k2, col_k3 = st.columns(3)
        with col_k1:
            input_gemini = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
        with col_k2:
            input_pexels = st.text_input("Pexels API Key", type="password", placeholder="R03L...")
        with col_k3:
            input_pixabay = st.text_input("Pixabay API Key", type="password", placeholder="5560...")

    # Resolve keys: secrets > input box
    gemini_key = get_secret("GEMINI_API_KEY") or input_gemini.strip() or None
    pexels_key = get_secret("PEXELS_API_KEY") or input_pexels.strip() or None
    pixabay_key = get_secret("PIXABAY_API_KEY") or input_pixabay.strip() or None

    # Key status indicator
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Gemini", "✅ Ready" if gemini_key else "❌ Kosong")
    col_s2.metric("Pexels", "✅ Ready" if pexels_key else "❌ Kosong")
    col_s3.metric("Pixabay", "✅ Ready" if pixabay_key else "❌ Kosong")

    st.divider()

    # --- Input Voiceover ---
    voiceover_text = st.text_area(
        "📝 Paste teks voiceover di sini:",
        height=200,
        placeholder="Contoh:\nIndonesia sedang memasuki era baru teknologi. AI dan otomasi mulai mengubah cara kita bekerja..."
    )

    # --- Relevance threshold ---
    threshold = st.slider(
        "🎯 Threshold relevansi minimum",
        min_value=0.0, max_value=1.0, value=0.25, step=0.05,
        help="Skor di bawah ini akan trigger fallback ke Pixabay. Semakin tinggi = makin ketat."
    )

    col_btn1, col_btn2 = st.columns(2)
    analyze_btn = col_btn1.button("🧠 Analisis Scenes", use_container_width=True, type="primary")
    search_btn = col_btn2.button("🔍 Cari Visual Semua Scene", use_container_width=True)

    # ── STEP 1: ANALISIS SCENES ──────────────────
    if analyze_btn:
        if not voiceover_text.strip():
            st.warning("Isi teks voiceover dulu.")
        elif not gemini_key:
            st.error("Gemini API Key belum diisi.")
        else:
            with st.spinner("🧠 AI sedang memecah script menjadi scenes..."):
                try:
                    scenes = split_scenes(voiceover_text, gemini_key)
                    st.session_state["scenes"] = scenes
                    st.session_state["visual_results"] = {}
                    st.success(f"✅ Script dipecah menjadi **{len(scenes)} scenes**")
                except Exception as e:
                    st.error(f"❌ Gagal analisis: {e}")

    # Tampilkan scenes hasil analisis
    if "scenes" in st.session_state and st.session_state["scenes"]:
        scenes = st.session_state["scenes"]
        st.divider()
        st.subheader(f"📋 {len(scenes)} Scenes Terdeteksi")

        for scene in scenes:
            sid = scene.get("scene_id", "?")
            with st.container(border=True):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown(f"**Scene {sid}**")
                    st.write(scene.get("text", ""))
                with c2:
                    st.caption("🔍 Visual Query")
                    st.code(scene.get("visual_query", ""), language=None)
                    st.caption(f"Mood: `{scene.get('mood', '-')}`")

    # ── STEP 2: CARI VISUAL ──────────────────────
    if search_btn:
        if "scenes" not in st.session_state or not st.session_state["scenes"]:
            st.warning("Analisis scenes dulu sebelum cari visual.")
        elif not pexels_key:
            st.error("Pexels API Key belum diisi.")
        elif not pixabay_key:
            st.error("Pixabay API Key belum diisi.")
        else:
            scenes = st.session_state["scenes"]
            results = {}
            progress = st.progress(0)
            status = st.empty()

            for idx, scene in enumerate(scenes):
                sid = scene.get("scene_id", idx + 1)
                query = scene.get("visual_query", "")
                status.info(f"🔍 Mencari visual untuk Scene {sid}: `{query}`")

                try:
                    best_clip, log = find_best_visual(query, pexels_key, pixabay_key, threshold)
                    results[sid] = {"scene": scene, "clip": best_clip, "log": log}
                except Exception as e:
                    results[sid] = {"scene": scene, "clip": None, "log": [f"Error: {e}"]}

                progress.progress((idx + 1) / len(scenes))
                time.sleep(0.3)  # Rate limit safety

            status.success(f"✅ Selesai! Visual ditemukan untuk {sum(1 for r in results.values() if r['clip'])} dari {len(scenes)} scenes.")
            st.session_state["visual_results"] = results

    # ── TAMPILKAN HASIL VISUAL ───────────────────
    if "visual_results" in st.session_state and st.session_state["visual_results"]:
        results = st.session_state["visual_results"]
        st.divider()
        st.subheader("🎞️ Hasil Visual Per Scene")

        for sid, data in results.items():
            scene = data["scene"]
            clip = data["clip"]
            log = data["log"]

            with st.container(border=True):
                st.markdown(f"### Scene {sid}")
                st.caption(f"*{scene.get('text', '')}*")

                if clip:
                    col_img, col_info = st.columns([1, 2])
                    with col_img:
                        if clip.get("thumbnail"):
                            st.image(clip["thumbnail"], use_container_width=True)
                        else:
                            st.info("No preview")
                    with col_info:
                        score = clip.get("relevance_score", 0)
                        st.markdown(f"**Source:** `{clip.get('source', '-')}`")
                        st.markdown(f"**Relevansi:** {score_color(score)} `{score:.0%}`")
                        st.markdown(f"**Query digunakan:** `{scene.get('visual_query', '')}`")
                        if clip.get("tags"):
                            st.markdown(f"**Tags clip:** {clip['tags'][:80]}")
                        if clip.get("duration"):
                            st.markdown(f"**Durasi:** {clip['duration']} detik")
                        if clip.get("download_url"):
                            st.link_button("🔗 Buka / Preview Clip", clip["download_url"], use_container_width=True)

                    with st.expander("📋 Log pencarian"):
                        for line in log:
                            st.caption(line)
                else:
                    st.error("❌ Visual tidak ditemukan untuk scene ini.")
                    with st.expander("📋 Log pencarian"):
                        for line in log:
                            st.caption(line)

        # Summary stats
        st.divider()
        total = len(results)
        found = sum(1 for r in results.values() if r["clip"])
        pexels_count = sum(1 for r in results.values() if r["clip"] and r["clip"].get("source") == "Pexels")
        pixabay_count = sum(1 for r in results.values() if r["clip"] and r["clip"].get("source") == "Pixabay")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Scenes", total)
        c2.metric("Visual Ditemukan", found)
        c3.metric("Dari Pexels", pexels_count)
        c4.metric("Dari Pixabay", pixabay_count)

st.markdown("---")
st.caption("Powered by Gemini · Pexels · Pixabay | Deployed via Streamlit Community Cloud")
