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
    print("I am here")
    r.pause_threshold = pause_threshold
    m = sr.Microphone(device_index=device_index)
    print("I am here")
    with m as source:
        r.adjust_for_ambient_noise(source)

    print("I am here")
    with m as source:
        audio = r.listen(source)
    print("I am here")
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


def whisper_process(queue, model: str = "base", mic_index: int = 0, pause_threshold: float = 0.8):
    """
    This creates a whisper process

    This listens in the background for any text that was spoken
    :param queue: the process queue
    :param model: the type of model ("tiny", "base", "small", "medium", "large", "turbo")
    :param mic_index: the index of the mic
    :param pause_threshold: the silent threshold in seconds before stop listening
    :return:
    """
    r = sr.Recognizer()
    r.pause_threshold = pause_threshold

    m = sr.Microphone(device_index=mic_index)

    with m as source:
        r.adjust_for_ambient_noise(source)
        while True:
            try:
                audio = r.listen(source)
                text = r.recognize_faster_whisper(audio, language="en", model=model)
                queue.put(("whisper", text.lstrip(" ").rstrip(" ")))

            except sr.UnknownValueError:
                queue.put(("whisper", "COULDN'T UNDERSTAND"))
            except sr.RequestError as e:
                queue.put(("whisper", "UNKNOWN ERROR"))

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
    text = active_listening(model="base", device_index=1)
    print(text)