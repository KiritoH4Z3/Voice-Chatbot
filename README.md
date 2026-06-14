# Voice Chatbot

A Python voice assistant that listens to spoken input, generates replies with a HuggingFace-hosted Mistral-7B model via LangChain, and speaks the answers back.

## Results

- Captures microphone audio and transcribes it to text using the Google Speech Recognition API.
- Sends transcribed questions through a LangChain `LLMChain` to the `mistralai/Mistral-7B-Instruct-v0.3` model served by a HuggingFace Endpoint.
- Applies a prompt template that instructs the model to answer the asked question directly, in a happy tone, without emojis or repetition.
- Converts model responses to speech locally with `pyttsx3`.
- Runs as a continuous listen-respond loop from the command line, with a standalone speech-to-text module (`speechrecognition.py`) for testing transcription on its own.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Mistral AI](https://img.shields.io/badge/Mistral--7B--Instruct-FF7000?style=flat&logo=mistralai&logoColor=white)
![SpeechRecognition](https://img.shields.io/badge/SpeechRecognition-4285F4?style=flat&logo=google&logoColor=white)
![pyttsx3](https://img.shields.io/badge/pyttsx3-TTS-306998?style=flat)

## Architecture

The `Chatbot` class wraps three stages: `speech_to_text` records from the microphone and transcribes it with the Google recognizer, `response` feeds the transcript into a LangChain `LLMChain` wired to a HuggingFace Endpoint running Mistral-7B-Instruct, and a `pyttsx3` engine handles text-to-speech. The HuggingFace API token is read from `key.py`, and the main loop in `main.py` keeps listening and responding until the program is stopped.

## How to Run

```bash
git clone https://github.com/KiritoH4Z3/Voice-Chatbot.git
pip install -r requirements.txt
python main.py
```

Before running, add your HuggingFace API token to `key.py` (replace the placeholder in the `YourKey` variable).

## About

Built by Abdullah Mohammed Hazeq as a learning project exploring voice-driven conversational AI with LangChain and HuggingFace-hosted LLMs.
