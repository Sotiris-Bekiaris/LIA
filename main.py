# import ffmpeg
# import numpy as np
# from whispercpp import Whisper

# w = Whisper.from_pretrained("tiny.en")

# try:
#     y, _ = (
#         ffmpeg.input("test.wav", threads=0)
#         .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar="16k")
#         .run(
#             cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
#         )
#     )
# except ffmpeg.Error as e:
#     raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

# arr = np.frombuffer(y, np.int16).flatten().astype(np.float32) / 32768.0

# what = w.transcribe_from_file("test.wav")

# print(what)


# --------------------------------------------
# --------------------------------------------


# # whispercpp.py
# import numpy as np
# from whispercpp import Whisper
# import pyaudio
# import wave
# import threading
# import time

# w = Whisper.from_pretrained("tiny.en")

# # Set up PyAudio for microphone input
# CHUNK = 1024
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000

# p = pyaudio.PyAudio()

# # Global variables
# frames = []
# is_recording = False

# def record_audio():
#     global frames, is_recording
    
#     stream = p.open(format=FORMAT,
#                     channels=CHANNELS,
#                     rate=RATE,
#                     input=True,
#                     frames_per_buffer=CHUNK)

#     print("Recording... Press Enter to stop.")
    
#     while is_recording:
#         data = stream.read(CHUNK)
#         frames.append(data)

#     stream.stop_stream()
#     stream.close()

# # Start recording
# is_recording = True
# record_thread = threading.Thread(target=record_audio)
# record_thread.start()

# # Wait for user input to stop recording
# input("Press Enter to stop recording...\n")
# is_recording = False
# record_thread.join()

# p.terminate()

# print("Recording finished.")

# # Convert the recorded audio to a numpy array
# audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
# audio_float32 = audio_data.astype(np.float32) / 32768.0

# # Transcribe the audio
# transcribedText = w.transcribe(audio_float32) # this is the transcribedText that i want to pass to ollama.py as a prompt


# print(transcribedText)



# --------------------------------------------
# --------------------------------------------

# combined_system.py

import numpy as np
from whispercpp import Whisper
import pyaudio
import threading
import requests
import json
from TTS.api import TTS
import sounddevice as sd

# WhisperCPP setup
w = Whisper.from_pretrained("./ggml-base.en.bin")

# PyAudio setup
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()

# Global variables
frames = []
is_recording = False

# Ollama setup
url = "http://localhost:11434/api/generate"

# TTS setup
device = "cpu"
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(device)

def record_audio():
    global frames, is_recording
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording... Press Enter to stop.")
    
    while is_recording:
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()

def transcribe_audio():
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    audio_float32 = audio_data.astype(np.float32) / 32768.0
    transcribed_text = w.transcribe(audio_float32)
    return transcribed_text

def generate_response(transcribed_text):
    payload = {
        "model": "llama3.1",
        "prompt": transcribed_text,
        "stream": False
    }
    
    response = requests.post(url, data=json.dumps(payload), headers={"Content-Type": "application/json"})
    
    try:
        result = response.json()
        if 'response' in result:
            return result['response']
        else:
            print("Key 'response' not found in the JSON response.")
            return None
    except json.JSONDecodeError as e:
        print("JSON Decode Error:", e)
        print("Response content was not valid JSON.")
        return None

def play_audio(audio, sample_rate):
    sd.play(audio, sample_rate)
    sd.wait()

def text_to_speech(text):
    wav = tts.tts(text)
    audio_np = np.array(wav, dtype=np.float32)
    play_audio(audio_np, tts.synthesizer.output_sample_rate)

def main():
    global is_recording, frames
    
    # Start recording
    is_recording = True
    record_thread = threading.Thread(target=record_audio)
    record_thread.start()

    # Wait for user input to stop recording
    input("Press Enter to stop recording...\n")
    is_recording = False
    record_thread.join()

    p.terminate()
    print("Recording finished.")

    # Transcribe the audio
    transcribed_text = transcribe_audio()
    print("Transcribed Text:", transcribed_text)

    # Generate response using Ollama
    generated_text = generate_response(transcribed_text)
    if generated_text:
        print("Generated Text:", generated_text)

        # Convert generated text to speech
        print("Converting text to speech...")
        text_to_speech(generated_text)
        print("Speech synthesis complete and streamed to speakers.")
    else:
        print("Failed to generate response.")

if __name__ == "__main__":
    main()