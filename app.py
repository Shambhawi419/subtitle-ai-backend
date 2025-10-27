%%writefile app.py
import openai
from flask import Flask, request, jsonify
from pydub import AudioSegment, silence
import whisper, torch, librosa, os, json, textwrap, math, urllib.request, gdown, pysrt
from resemblyzer import VoiceEncoder
from spectralcluster import SpectralClusterer
from transformers import pipeline
import numpy as np
import openai

# ---------- INIT ----------
app = Flask(__name__)

# ---------- OPENAI KEY ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
if not OPENAI_API_KEY:
    OPENAI_API_KEY = "YOUR_API_KEY_HERE"
openai.api_key = OPENAI_API_KEY

# ---------- Emotion Colors ----------
EMOTION_COLORS = {
    "joy": "#FFD700",
    "anger": "#FF4500",
    "sadness": "#1E90FF",
    "fear": "#9400D3",
    "surprise": "#00CED1",
    "love": "#FF69B4",
    "neutral": "#FFFFFF"
}

# ---------- Helpers ----------
def sec_to_srt(ts):
    millis = int((ts - math.floor(ts)) * 1000)
    s = int(math.floor(ts))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:02}:{m:02}:{sec:02},{millis:03}"

def sec_to_vtt(ts):
    millis = int((ts - math.floor(ts)) * 1000)
    s = int(math.floor(ts))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:02}:{m:02}:{sec:02}.{millis:03}"

def download_video(url, output_path="video.mp4"):
    if "drive.google.com" in url:
        try:
            file_id = url.split("/d/")[1].split("/")[0]
        except Exception:
            file_id = url.split("id=")[-1].split("&")[0]
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
    else:
        urllib.request.urlretrieve(url, output_path)
    return output_path

def extract_audio(video_path, audio_path="audio.wav"):
    AudioSegment.from_file(video_path).export(audio_path, format="wav")
    return audio_path

def autosync_shift(audio_path, segments):
    audio = AudioSegment.from_wav(audio_path)
    nonsilent = silence.detect_nonsilent(audio, min_silence_len=200, silence_thresh=-40)
    if not nonsilent:
        return segments
    first_ns_start = nonsilent[0][0] / 1000.0
    first_seg_start = segments[0]["start"] if segments else 0.0
    offset = first_ns_start - first_seg_start
    if abs(offset) > 0.2:
        for s in segments:
            s["start"] = max(0.0, s["start"] + offset)
            s["end"] = max(0.0, s["end"] + offset)
    return segments

# ---------- Text Punctuation ----------


def punctuate_text(text):
    try:
        prompt = "Fix capitalization and punctuation only:\n" + text

        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("Error in punctuate_text:", e)
        return text





# ---------- Emotion Detection ----------
emotion_pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion", top_k=1)

def detect_emotion(text):
    try:
        label = emotion_pipe(text[:512])[0]["label"].lower()
        for k in EMOTION_COLORS:
            if k in label:
                return k
        return "neutral"
    except Exception:
        return "neutral"

# ---------- Core Transcription + Diarization ----------
def transcribe_and_diarize(audio_path, language=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1️⃣ Load Whisper model
    model = whisper.load_model("small", device=device)
    print("Transcribing audio...")
    result = model.transcribe(audio_path, language=language)
    segments = result["segments"]

    # 2️⃣ Speaker Embeddings (Resemblyzer)
    print("Extracting speaker embeddings...")
    wav, sr = librosa.load(audio_path, sr=16000)
    encoder = VoiceEncoder(device=device)
    print("Loaded voice encoder on", device)

    chunk_len = sr * 3
    chunks = [wav[i:i + chunk_len] for i in range(0, len(wav), chunk_len)]
    embeddings = np.array([encoder.embed_utterance(c) for c in chunks])

    # 3️⃣ Cluster Speakers
    clusterer = SpectralClusterer(min_clusters=1, max_clusters=4, p_percentile=0.9, gaussian_blur_sigma=1)
    labels = clusterer.predict(embeddings)
    print(f"Detected speakers: {len(set(labels))}")

    # 4️⃣ Assign Labels
    segs = []
    for i, seg in enumerate(segments):
        speaker = f"Speaker {labels[min(i, len(labels)-1)] + 1}" if len(labels) else "Speaker 1"
        segs.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "speaker": speaker
        })
    return segs

# ---------- Export ----------
def export_json(segments, path="transcript_data.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"subtitles": segments}, f, indent=2, ensure_ascii=False)
    return path

def export_srt(segments, path="final_subtitles.srt"):
    subs = pysrt.SubRipFile()
    for i, s in enumerate(segments):
        item = pysrt.SubRipItem(
            index=i + 1,
            start=pysrt.SubRipTime.from_string(sec_to_srt(s["start"])),
            end=pysrt.SubRipTime.from_string(sec_to_srt(s["end"])),
            text=f"[{s['speaker']}] {s['text']}"
        )
        subs.append(item)
    subs.save(path, encoding="utf-8")
    return path

def export_vtt(segments, path="final_subtitles.vtt"):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for s in segments:
            f.write(f"{sec_to_vtt(s['start'])} --> {sec_to_vtt(s['end'])}\n")
            f.write(f"{s['speaker']}: {s['text']}\n\n")
    return path

# ---------- API ROUTE ----------
@app.route('/generate_subtitles', methods=['POST'])
def generate_subtitles():
    try:
        data = request.json
        video_url = data.get("video_url")
        language = data.get("language", None)

        print("⬇️ Downloading video...")
        video = download_video(video_url)
        audio = extract_audio(video)
        print("🎧 Processing audio...")
        segs = transcribe_and_diarize(audio, language)
        segs = autosync_shift(audio, segs)

        print("🎨 Adding emotion + punctuation...")
        for s in segs:
            s["text"] = punctuate_text(s["text"])
            s["emotion"] = detect_emotion(s["text"])
            s["color"] = EMOTION_COLORS.get(s["emotion"], "#FFFFFF")

        export_json(segs)
        export_srt(segs)
        export_vtt(segs)

        return jsonify({
            "status": "success",
            "message": "Subtitles generated successfully",
            "data": segs
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------- RUN ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
