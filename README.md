# Automatic Speech Recognition Using Whisper

A simple, beginner-friendly speech-to-text web app built with **Python**, **Streamlit**, **Hugging Face Transformers**, and **OpenAI Whisper Small**. Upload an audio file and the app converts the spoken words into written text.

## Technologies Used

- **Python** – main programming language
- **Hugging Face Transformers** – loads the pretrained Whisper model and processor
- **OpenAI Whisper Small** – speech recognition model
- **PyTorch** – runs the deep learning model
- **Torchaudio** – loads audio and resamples it to 16 kHz
- **Streamlit** – simple web interface

## Features

- Upload WAV, FLAC, MP3, OGG, or M4A audio files
- Automatic mono conversion and resampling to 16 kHz
- Displays the transcription on the page
- Download the transcription as a `.txt` file
- Clear error messages for invalid or unsupported audio

## Project Structure

```text
SpeechToText/
├── app.py
├── requirements.txt
└── README.md
```

## Installation (Windows)

1. **Create a virtual environment**

   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**

   ```bash
   venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   > The first time you run the app, the Whisper Small model (~1 GB) will be downloaded automatically. Ensure you have an internet connection and enough RAM (at least 4–6 GB free).

## Run the App

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

## How to Use

1. Upload an audio file (WAV, FLAC, MP3, etc.).
2. Click **Transcribe**.
3. Wait for the model to process the audio.
4. Read the transcription and click **Download Transcription** to save it.

## Expected Output

**Input:** An audio recording of a person speaking.

**Output:**

```text
Transcription:
He hoped there would be stew for dinner,
turnips and carrots and bruised potatoes...
```

## Troubleshooting

- **Missing dependency errors** – Run `pip install -r requirements.txt` again inside the activated virtual environment.
- **Audio-loading errors** – Convert your file to WAV (16 kHz, mono) with a tool like Audacity or `ffmpeg`, then re-upload.
- **Memory issues** – Close other apps, use a shorter audio clip, or run on a machine with more RAM. GPU is optional but speeds up inference.
- **`torchaudio` backend errors** – Make sure `ffmpeg` is installed and available on your PATH.
- **Slow first run** – The model is downloaded and cached on first use; later runs are much faster.

## Notes

- This app works with **uploaded audio files**, not real-time microphone input.
- Only the Whisper Small model is used, as required by the project spec.

venv\Scripts\activate
streamlit run app.py
