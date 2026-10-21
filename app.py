
import streamlit as st
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import scipy.io.wavfile as wavfile
import soundfile as sf
import io

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Whisper Speech-to-Text",
    page_icon="🎙️",
    layout="centered",
)

# --------------------------------------------------
# DARK / LIGHT MODE
# --------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

dark_mode = st.session_state.dark_mode

# Theme colors
if dark_mode:
    bg_color = "#0E1117"
    text_color = "#FAFAFA"
    card_color = "#1B202B"
    border_color = "#343B49"
    muted_color = "#B8C0CC"
    input_bg = "#252B38"
    accent_color = "#7C83FF"
else:
    bg_color = "#FFFFFF"
    text_color = "#1F2937"
    card_color = "#F5F7FB"
    border_color = "#D7DCE5"
    muted_color = "#596579"
    input_bg = "#FFFFFF"
    accent_color = "#4F46E5"

# --------------------------------------------------
# APPLY THEME CSS
# --------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
        transition: background-color 0.2s ease;
    }}

    [data-testid="stHeader"] {{
        background-color: {bg_color};
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: {bg_color};
    }}

    .stApp h1, .stApp h2, .stApp h3,
    .stApp p, .stApp label,
    [data-testid="stMarkdownContainer"] {{
        color: {text_color};
    }}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {card_color};
        border-color: {border_color};
        border-radius: 14px;
    }}

    [data-testid="stFileUploader"] {{
        background-color: {card_color};
        border: 1px dashed {border_color};
        border-radius: 12px;
        padding: 12px;
    }}

    [data-testid="stFileUploader"] section {{
        background-color: {card_color};
    }}

    [data-testid="stTextArea"] textarea {{
        background-color: {input_bg};
        color: {text_color};
        border: 1px solid {border_color};
        border-radius: 10px;
    }}

    [data-testid="stAudio"] {{
        background-color: {card_color};
        border-radius: 10px;
    }}

    [data-testid="stAlert"] {{
        border-radius: 10px;
    }}

    .stButton > button,
    .stDownloadButton > button {{
        border-radius: 10px;
        font-weight: 600;
        min-height: 42px;
    }}

    .stButton > button[kind="primary"] {{
        background-color: {accent_color};
        color: white;
        border: none;
    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        border-color: {accent_color};
        color: {accent_color};
    }}

    [data-testid="stCaptionContainer"] {{
        color: {muted_color};
    }}

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# HEADER AND THEME TOGGLE
# --------------------------------------------------
col1, col2 = st.columns([4, 1])

with col1:
    st.title("🎙️ Whisper Speech-to-Text")

with col2:
    st.write("")
    st.write("")

    toggle_value = st.toggle(
        "Dark Mode",
        value=st.session_state.dark_mode,
        key="theme_toggle",
    )

    if toggle_value != st.session_state.dark_mode:
        st.session_state.dark_mode = toggle_value
        st.rerun()

st.caption("Convert your audio into text using OpenAI Whisper.")
st.write("Upload an audio file to generate a written transcript.")

# --------------------------------------------------
# LOAD WHISPER MODEL
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "openai/whisper-small"

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name
    )

    model.eval()

    return processor, model

# --------------------------------------------------
# LOAD AND PREPROCESS AUDIO
# --------------------------------------------------
def load_audio(uploaded_file):
    file_bytes = uploaded_file.getvalue()

    try:
        data, sample_rate = sf.read(
            io.BytesIO(file_bytes),
            dtype="float32",
        )
    except Exception:
        sample_rate, data = wavfile.read(
            io.BytesIO(file_bytes)
        )

    # Convert stereo audio to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Normalize integer PCM audio before conversion
    if np.issubdtype(data.dtype, np.integer):
        max_value = max(
            abs(np.iinfo(data.dtype).min),
            np.iinfo(data.dtype).max,
        )
        data = data.astype(np.float32) / max_value
    else:
        data = data.astype(np.float32)

    # Prevent invalid values
    data = np.nan_to_num(data)

    waveform = torch.from_numpy(data.copy())

    # Resample to Whisper's required sample rate
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000,
        )
        waveform = resampler(waveform)

    return waveform

# --------------------------------------------------
# TRANSCRIBE AUDIO
# --------------------------------------------------
def transcribe(audio_tensor, processor, model):
    inputs = processor(
        audio_tensor.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs["input_features"]
        )

    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
    )[0]

    return transcription.strip()

# --------------------------------------------------
# AUDIO UPLOAD SECTION
# --------------------------------------------------
with st.container(border=True):
    st.subheader("📁 Upload Audio")

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "flac", "mp3", "ogg", "m4a"],
        help="Supported formats: WAV, FLAC, MP3, OGG and M4A.",
    )

    if uploaded_file is not None:
        st.audio(uploaded_file)

        st.caption(
            f"File: {uploaded_file.name} "
            f"({uploaded_file.size / 1024:.1f} KB)"
        )

        transcribe_btn = st.button(
            "✨ Transcribe Audio",
            type="primary",
            use_container_width=True,
        )
    else:
        transcribe_btn = False
        st.info("👆 Upload an audio file to get started.")

# --------------------------------------------------
# TRANSCRIPTION RESULTS
# --------------------------------------------------
if uploaded_file is not None and transcribe_btn:
    try:
        with st.spinner(
            "Loading Whisper model... "
            "The first load may take a few minutes."
        ):
            processor, model = load_model()

        with st.spinner(
            "🎧 Transcribing audio... Please wait."
        ):
            audio_tensor = load_audio(uploaded_file)

            if audio_tensor.numel() == 0:
                st.error("The uploaded audio file is empty.")
                st.stop()

            transcription = transcribe(
                audio_tensor,
                processor,
                model,
            )

        if transcription:
            st.success("Transcription completed!")

            with st.container(border=True):
                st.subheader("📝 Transcript")

                st.text_area(
                    "Transcribed text",
                    value=transcription,
                    height=200,
                    label_visibility="collapsed",
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        label="📥 Download TXT",
                        data=transcription,
                        file_name="transcription.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

                with col2:
                    st.button(
                        "✅ Transcription Ready",
                        disabled=True,
                        use_container_width=True,
                    )
        else:
            st.warning(
                "No speech was detected. "
                "Please try another audio file."
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")

        st.info(
            "Check your audio file and installed dependencies. "
            "The first Whisper model download requires an internet "
            "connection."
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.write("")

st.markdown(
    "<div style='text-align:center; opacity:0.7; "
    "font-size:13px;'>"
    "Powered by OpenAI Whisper · Built with Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
