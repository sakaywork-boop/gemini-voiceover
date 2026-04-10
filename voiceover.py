import streamlit as st
import time
import io
import wave
import array
from google import genai
from google.genai import types

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
    return bytes(mp3)  # pastikan return bytes murni

def process_tts(text, voice_name, speed_value):
    KEYS = get_keys()
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

# --- UI ---
st.set_page_config(page_title="Gemini TTS Simple", page_icon="🎙️", layout="wide")
st.title("🎙️ Gemini Voiceover Dashboard")

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

st.markdown("---")
st.caption("Deployed via Streamlit Community Cloud")
