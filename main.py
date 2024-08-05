# main.py

import numpy as np
from whispercpp import Whisper
import pyaudio
import threading
import requests
import json
import sounddevice as sd

import os
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from trainer.io import get_user_data_dir
from TTS.utils.manage import ModelManager
import torchaudio
import time
from ollama import Client
# WhisperCPP setup
# Load the pre-trained Whisper model for speech recognition
w = Whisper.from_pretrained("./ggml-base.en.bin")

# PyAudio setup
# Configure audio recording parameters
CHUNK = 1024  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # Audio format (16-bit int)
CHANNELS = 1  # Mono audio
RATE = 16000  # Sampling rate (Hz)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Global variables
frames = []  # List to store audio frames
is_recording = False  # Flag to control recording state

# Ollama setup
# URL for the Ollama API endpoint
url = "http://localhost:11434/api/generate"


# Setup
torch.set_num_threads(int(os.environ.get("NUM_THREADS", os.cpu_count())))
device = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("USE_CPU", "0") == "0" else "cpu")

# Model loading
custom_model_path = os.environ.get("CUSTOM_MODEL_PATH", "./tts_models")

if os.path.exists(custom_model_path) and os.path.isfile(custom_model_path + "/config.json"):
    model_path = custom_model_path
    print("Loading custom model from", model_path)
else:
    print("Loading default model")
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    print("Downloading XTTS Model:", model_name)
    ModelManager().download_model(model_name)
    model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
    print("XTTS Model downloaded")

print("Loading XTTS")
config = XttsConfig()
config.load_json(os.path.join(model_path, "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=model_path, eval=True, use_deepspeed=device.type == "cuda")
model.to(device)
print("XTTS Loaded.")


def record_audio():
    """
    Function to record audio from the microphone.
    Runs in a separate thread to allow for asynchronous recording.
    """
    global frames, is_recording
    
    # Open audio stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording... Press Enter to stop.")
    
    # Continue recording while is_recording flag is True
    while is_recording:
        data = stream.read(CHUNK)
        frames.append(data)

    # Close the stream after recording is stopped
    stream.stop_stream()
    stream.close()

def transcribe_audio():
    """
    Function to transcribe the recorded audio using WhisperCPP.
    
    Returns:
    str: Transcribed text from the audio
    """
    # Convert byte data to numpy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    # Normalize audio data
    audio_float32 = audio_data.astype(np.float32) / 32768.0
    # Transcribe audio using Whisper model
    transcribed_text = w.transcribe(audio_float32)
    return transcribed_text

client = Client(host='http://localhost:11434')
def generate_response(transcribed_text):
    """
    Function to generate a response using the Ollama API.
    
    Args:
    transcribed_text (str): The text to generate a response for
    
    Returns:
    str: Generated response text, or None if an error occurs
    """
    response = client.chat(model='llama3.1', messages=[
    {
        'role': 'user',
        'content': f"{transcribed_text}",
    },
    ])
    print(response['message']['content'])
    return response['message']['content']

def play_audio(audio, sample_rate):
    """
    Function to play audio using sounddevice.
    
    Args:
    audio (numpy.ndarray): Audio data to play
    sample_rate (int): Sampling rate of the audio
    """
    sd.play(audio, sample_rate)
    sd.wait()  # Wait until playback is finished

def postprocess(wav):
    if isinstance(wav, list):
        wav = torch.cat(wav, dim=0)
    wav = wav.clone().detach().cpu().numpy()
    wav = wav[None, : int(wav.shape[0])]
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    return wav


def get_speakers():
    if hasattr(model, "speaker_manager") and hasattr(model.speaker_manager, "speakers"):
        return {
            speaker: {
                "speaker_embedding": model.speaker_manager.speakers[speaker]["speaker_embedding"].cpu().squeeze().half().tolist(),
                "gpt_cond_latent": model.speaker_manager.speakers[speaker]["gpt_cond_latent"].cpu().squeeze().half().tolist(),
            }
            for speaker in model.speaker_manager.speakers.keys()
        }
    else:
        return {}

def generate_speech(text, stream=False):
    print(f"Generating speech for: '{text}'")

    # Get speaker data
    speakers = get_speakers()

    speaker_data = list(speakers.values())[0]  # Use the first available speaker as a default


    speaker_embedding = torch.tensor(speaker_data["speaker_embedding"]).unsqueeze(0).unsqueeze(-1).to(device)
    gpt_cond_latent = torch.tensor(speaker_data["gpt_cond_latent"]).reshape((-1, 1024)).unsqueeze(0).to(device)

    if stream:
        sample_rate = 24000  # Assuming this is the correct sample rate

        sd.default.samplerate = sample_rate
        sd.default.channels = 1  # Mono audio

        stream = sd.OutputStream(samplerate=sample_rate, channels=1)
        stream.start()


        t0 = time.time()
        chunks = model.inference_stream(
            text,
            "en",
            gpt_cond_latent,
            speaker_embedding,
            stream_chunk_size=5,
            enable_text_splitting=False,
            temperature=0.7, # Add custom parameters here

        )

        wav_chuncks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                print(f"Time to first chunck: {time.time() - t0}")
            print(f"Received chunk {i} of audio length {chunk.shape[-1]}")

            # Convert chunk to numpy array and play it
            chunk_np = chunk.squeeze().cpu().numpy()
            stream.write(chunk_np)
            wav_chuncks.append(chunk)
        wav = torch.cat(wav_chuncks, dim=0)
        torchaudio.save("xtts_streaming.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)
        print(f"Total time for all chuncks: {time.time() - t0}")

        stream.stop()
        stream.close()


    else:
        t0 = time.time()

        out = model.inference(
            text,
            "en",
            gpt_cond_latent,
            speaker_embedding,
        )
        wav = postprocess(torch.tensor(out["wav"]))
        
        # Play the generated audio
        play_audio(wav.squeeze(), 24000)
        torchaudio.save("xtts-ft.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)
        print(f"Total time no streaming: {time.time() - t0}")
        

    print("Speech generation complete.")

def text_to_speech(text):
    """
    Function to convert text to speech and play it.
    
    Args:
    text (str): Text to be converted to speech
    """
    # Generate speech from text
    generate_speech(text)

def reset_recording():
    global frames
    frames = []
def main():
    global is_recording, frames
    
    while True:
        # Reset recording
        reset_recording()
        
        # Start recording
        is_recording = True
        record_thread = threading.Thread(target=record_audio)
        record_thread.start()

        print("Recording... Press Enter to stop.")
        input()  # Wait for user input to stop recording
        is_recording = False
        record_thread.join()

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
        
        # Ask if the user wants to continue
        choice = input("Do you want to record again? (y/n): ").lower()
        if choice != 'y':
            break

    # Terminate PyAudio
    p.terminate()
    print("Program ended.")

if __name__ == "__main__":
    main()