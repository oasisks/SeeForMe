import tempfile
from subprocess import CalledProcessError, run

import speech_recognition as sr
from typing import Callable
import whisper
from whisper import Whisper
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import resample
import soundfile as sf
import io
import numpy as np
import torch
import librosa
import time
import os

print(torch.cuda.is_available())


def start_listening(callback: Callable, device_index: int = 1):
    """
    This function continuously listens to the background audio

    Returns
    :param device_index: the microphone device
    :param callback: the callback to perform when it hears the audio
    :return: returns a callback function that when called stops the background listening
    """
    r = sr.Recognizer()
    m = sr.Microphone(device_index=device_index)
    with m as source:
        r.adjust_for_ambient_noise(source)

    stop_listening = r.listen_in_background(m, callback)

    def stop():
        stop_listening(wait_for_stop=False)

    return stop


def process_audio(audio_array, sample_rate, model):
    """
    Processes the audio array and ask the model to figure out the words
    spoken in the audio
    :param audio_array: a np array that stores the audio
    :param sample_rate: the sample rate the audio was sampled
    :param model: the model object that returns the text
    :return: the result from the model
    """
    # Step 3: Ensure audio is 16 kHz (Whisper's expected sample rate)
    target_sr = 16000
    if sample_rate != target_sr:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=target_sr)

    # Load a Whisper model (choose a size such as "small", "medium", etc.)
    model = whisper.load_model("small")

    # Preprocess audio for Whisper
    # Whisper expects a 1D float32 NumPy array
    audio_array = np.array(audio_array, dtype=np.float32)
    audio_array = whisper.pad_or_trim(audio_array)

    # Compute the log-Mel spectrogram
    mel = whisper.log_mel_spectrogram(audio_array).to(model.device)

    # Step 4: Decode/transcribe the audio using Whisper
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    print("Transcribed text:", result.text)
    return result


def audio_to_text(recognizer, audio, model: Whisper, executor: ThreadPoolExecutor):
    """
    The callback function used to return the text from the audio

    :param recognizer: the recognizer such as Google or OpenAI (However, not using this)
    :param audio: The audio object created when listening in the background
    :param model: the whisper model we are using
    :param executor: a thread pool to store all the threads
    :return: returns the text string from the audio
    """
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        print("Google Speech Recognition thinks you said " + recognizer.recognize_google(audio))
        wav_file = io.BytesIO(audio.get_wav_data())
        audio_array, sample_rate = sf.read(wav_file)
        future = executor.submit(process_audio, audio_array, sample_rate, model)
        future.add_done_callback(lambda fut: print("Transcribed result:", fut.result()))
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))





def callback(recognizer, audio):
    # received audio data, now we'll recognize it using Google Speech Recognition
    print(recognizer)
    print(audio)
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        print("Google Speech Recognition thinks you said " + recognizer.recognize_google(audio))
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


if __name__ == '__main__':
    # text = audio_to_text()
    # pass
    # # # print(text)
    # model = whisper.load_model("small")
    # result = model.transcribe("test.wav")
    # executor = ThreadPoolExecutor(max_workers=2)
    #
    # def callback(recognizer, audio):
    #     return audio_to_text(recognizer, audio, model, executor)
    #
    # stop = start_listening(callback, 1)
    # while True:
    #     pass
    # Load Whisper model locally
    model = whisper.load_model("small")  # or "small", "medium", etc.


    # def callback(recognizer, audio):
    #     # Save the audio snippet to a temporary file
    #     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
    #         temp_audio.write(audio.get_wav_data())
    #         temp_audio.flush()
    #         # Transcribe the audio file using Whisper
    #         result = model.transcribe(temp_audio.name)
    #         print("Transcription:", result["text"])
    #
    #
    # # Initialize recognizer and microphone
    # r = sr.Recognizer()
    #
    # m = sr.Microphone(device_index=1)
    # with m as source:
    #     r.adjust_for_ambient_noise(source)
    #
    # stop_listening = r.listen_in_background(m, callback)
    # while True:
    #     pass
