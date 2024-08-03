# LIA

LIA (Local Intelligent Agent) is a project that implements an AI voice assistant capable of transcribing speech, generate a response, and convert that response text to speech. It utilizes WhisperCPP for speech recognition, Ollama for text generation, and TTS for speech synthesis.

## Features

- Real-time audio recording
- Speech-to-text transcription using WhisperCPP
- Text generation using Ollama
- Text-to-speech conversion using TTS
- Audio playback of generated responses

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher
- Ollama server running locally on port 11434
- WhisperCPP model file (`ggml-base.en.bin`) in the project directory

## Installation

1. Clone this repository:

```
git clone https://github.com/yourusername/ai-voice-assistant.git
```

```
cd ai-voice-assistant
```

2. Create a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

3. Install the required packages:

```
pip install -r requirements.txt
```

4. Download the WhisperCPP model file (`ggml-base.en.bin`) and place it in the project directory.

## Usage

1. Ensure that the Ollama server is running on `http://localhost:11434`.

2. Run the main script:

```
python main.py
```

3. The program will start recording audio. Speak into your microphone.

4. Press Enter to stop the recording.

5. The system will transcribe your speech, generate a response, and play it back through your speakers.

## Configuration

- To change the Ollama model, modify the `"model"` key in the `generate_response` function.
- To use a different TTS model, update the model path in the `TTS` initialization.

## Contributing

Contributions to this project are welcome. Please fork the repository and create a pull request with your changes.

## License

[MIT License](https://opensource.org/licenses/MIT)

## Acknowledgements

- [WhisperCPP](https://github.com/ggerganov/whisper.cpp) for speech recognition
- [Ollama](https://github.com/jmorganca/ollama) for text generation
- [TTS](https://github.com/coqui-ai/TTS) for text-to-speech conversion

## Contact

If you have any questions or feedback, please open an issue in this repository.
