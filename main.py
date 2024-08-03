# main.py

import numpy as np
from whispercpp import Whisper
import pyaudio
import threading
import requests
import json
from TTS.api import TTS
import sounddevice as sd

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

# TTS (Text-to-Speech) setup
device = "cpu"  # Use CPU for TTS processing
# Initialize TTS model
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(device)

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

def generate_response(transcribed_text):
    """
    Function to generate a response using the Ollama API.
    
    Args:
    transcribed_text (str): The text to generate a response for
    
    Returns:
    str: Generated response text, or None if an error occurs
    """
    payload = {
        "model": "llama3.1",
        "prompt": transcribed_text,
        "stream": False
    }
    
    # Send POST request to Ollama API
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
    """
    Function to play audio using sounddevice.
    
    Args:
    audio (numpy.ndarray): Audio data to play
    sample_rate (int): Sampling rate of the audio
    """
    sd.play(audio, sample_rate)
    sd.wait()  # Wait until playback is finished

def text_to_speech(text):
    """
    Function to convert text to speech and play it.
    
    Args:
    text (str): Text to be converted to speech
    """
    # Generate speech from text
    wav = tts.tts(text)
    # Convert to numpy array
    audio_np = np.array(wav, dtype=np.float32)
    # Play the generated audio
    play_audio(audio_np, tts.synthesizer.output_sample_rate)

def main():
    """
    Main function to orchestrate the voice assistant process.
    """
    global is_recording, frames
    
    # Start recording
    is_recording = True
    record_thread = threading.Thread(target=record_audio)
    record_thread.start()

    # Wait for user input to stop recording
    input("Press Enter to stop recording...\n")
    is_recording = False
    record_thread.join()

    # Terminate PyAudio
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