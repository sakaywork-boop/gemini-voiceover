import streamlit as st
import os
import time
import io
from pathlib import Path
from pydub import AudioSegment
from google import genai
from google.genai import types

# 1. SETUP LINGKUNGAN
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Ambil API Keys dari Streamlit Secrets
def get_keys():
    keys = []
    for k in ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"]:
        try:
            val = st.secrets[k]
            if val:
                keys.append(val)
        except:
            pass
    return keys

def process_tts(text, voice_name, filename, speed_value):
    KEYS = get_keys()
    final_mp3 = OUTPUT_DIR / filename

    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    if not KEYS:
        st.error("🚨 API Key tidak ditemukan! Pastikan sudah diisi di Streamlit Secrets.")
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

            raw_audio = response.candidates[0].content.parts[0].inline_data.data

            status_placeholder.info("🛠️ Mengatur tempo audio...")
            progress_bar.progress(80)

            # Konversi PCM ke AudioSegment pakai pydub (tanpa ffmpeg eksternal)
            audio = AudioSegment(
                data=raw_audio,
                sample_width=2,   # 16-bit = 2 bytes
                frame_rate=24000,
                channels=1
            )

            # Ubah kecepatan dengan mengubah frame_rate lalu resample
            new_frame_rate = int(audio.frame_rate * speed_value)
            audio_speed = audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate})
            audio_final = audio_speed.set_frame_rate(24000)

            # Export ke MP3
            buffer = io.BytesIO()
            audio_final.export(buffer, format="mp3", bitrate="128k")
            mp3_bytes = buffer.getvalue()

            with open(final_mp3, "wb") as f:
                f.write(mp3_bytes)

            progress_bar.progress(100)
            status_placeholder.success("✨ Selesai!")
            time.sleep(1)
            status_placeholder.empty()
            return final_mp3, mp3_bytes

        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                st.warning(f"⚠️ Key #{i+1} Limit. Mencoba Key berikutnya...")
                continue
            else:
                st.error(f"❌ Error: {e}")
                break

    return None, None

# --- UI DASHBOARD ---
st.set_page_config(page_title="Gemini TTS Simple", page_icon="🎙️", layout="wide")
st.title("🎙️ Gemini Voiceover Dashboard")

col1, col2 = st.columns([1, 3], gap="large")

with col1:
    st.subheader("⚙️ Pengaturan")
    voice_choice = st.selectbox("Pilih Karakter Suara", ["Puck", "Charon", "Kore", "Fenrir"])

    speed_label = st.radio(
        "Kecepatan Bicara:",
        ["Pelan (Santai)", "Normal", "Cepat"],
        index=1
    )

    speed_map = {
        "Pelan (Santai)": 0.85,
        "Normal": 1.0,
        "Cepat": 1.15
    }
    selected_speed = speed_map[speed_label]

    file_name = st.text_input("Nama File Output", "hasil_suara.mp3")

with col2:
    st.subheader("📝 Konten Teks")
    text_area = st.text_area("Masukkan teks narasi:", height=300, placeholder="Tempel naskah di sini...")

    if st.button("🚀 Generate Voiceover Now", use_container_width=True):
        if text_area.strip():
            result, mp3_bytes = process_tts(text_area, voice_choice, file_name, speed_value=selected_speed)
            if result and mp3_bytes:
                st.divider()
                st.audio(mp3_bytes)
                st.download_button("📥 Download MP3", mp3_bytes, file_name=file_name, use_container_width=True)
        else:
            st.warning("Silakan masukkan teks terlebih dahulu.")

st.markdown("---")
st.caption("Deployed via Streamlit Community Cloud")
