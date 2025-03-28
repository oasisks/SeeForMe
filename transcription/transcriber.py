import speech_recognition as sr
from typing import Callable


def active_listening(model: str = "base", device_index: int = 0, pause_threshold: float = 0.8) -> str:
    """
    When calling this function, it will interrupt the main thread and wait until a voice is heard
    followed by a silence

    :param model: the type of model ("tiny", "base", "small", "medium", "large", "turbo")
    :param device_index: the index of the microphone device
    :param pause_threshold: seconds of non-speaking audio before a phrase is considered complete
    :return: the text from the audio
    """
    r = sr.Recognizer()
    r.pause_threshold = pause_threshold
    m = sr.Microphone(device_index=device_index)
    with m as source:
        r.adjust_for_ambient_noise(source)

    with m as source:
        audio = r.listen(source)

    try:
        text = r.recognize_faster_whisper(audio, language="en", model=model)
        return text.lstrip(" ").rstrip(" ")
    except sr.UnknownValueError:
        return "Faster-Whisper couldn't understand audio"
    except sr.RequestError as e:
        return "Could not request result from Whisper"


def callback(recognizer, audio):
    """
    A sample callback function
    :param recognizer: the recognizer
    :param audio: the audio data
    :return:
    """


def background_listening(callback: Callable, model: str = "base", device_index: int = 0, pause_threshold: float = 0.8) -> Callable:
    """
    Listens to the audio in a separate thread. The main thread will still execute
    without interrupt.

    :param callback: The callback function that gets triggered when an audio is processed
    :param model: the type of model ("tiny", "base", "small", "medium", "large", "turbo")
    :param device_index: the index of the microphone device
    :param pause_threshold: seconds of non-speaking audio before a phrase is considered complete
    :return: returns a function that when activated stops the background listening
    """
    r = sr.Recognizer()
    r.pause_threshold = pause_threshold
    m = sr.Microphone(device_index=device_index)

    with m as source:
        r.adjust_for_ambient_noise(source)

    stop_listening = r.listen_in_background(m, callback)

    def stop():
        stop_listening(wait_for_stop=False)

    return stop


if __name__ == '__main__':
    print("Initiating listening")
    text = active_listening(model="turbo", device_index=1)
    print(text)