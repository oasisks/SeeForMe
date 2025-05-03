import os

import numpy as np
import speech_recognition as sr
from google import genai
from typing import Callable
from typing import List, Dict
from dotenv import load_dotenv
from PIL import Image
from YOLO_test import YOLO
import cv2
import pprint

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))


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


def gemini_user_response(message_history: List[Dict[str, str]], image: np.ndarray, history_window: int = 10, model_name: str = "gemini-2.0-flash"):
    """
    Generates a response base on the message history. You can give it an arbitrary size message_history, but
    it will only look at the last 'history_window' history elements.

    For optimization, it is essential to truncate message_history for memory performance

    An example of a message_history:
    message_history = [
        {
            "agent": "user",
            "message": "What is in front of me?",
            "objects": {
                "left": {"chair": 1},
                "right": {"table": 1},
                "forward": {"mouse": 1}
            }
        },
        {
            "agent": "system",
            "message": "There is a mouse right in front of you"
        }
    ]
    :param message_history: a list of message histories
    :param history_window: the window in which the model looks at the last "history_window"
                            elements within the message history
    :param model_name: the name of the model
    :param image: an rgb image with 3 channels
    :return: it will return a new message_history
    """
    relevant_history = message_history[-history_window:]
    prompt = """
    Example 1:
    Image: An image of a brown chair on the left, a white table in the center, and a black mouse in the center.
    Message History: [
        {
            "agent": "user",
            "message": "What is in front of me?",
            "objects": {
                "left": {"chair": 1},
                "right": {"table": 1},
                "forward": {"mouse": 1}
            }
        }
    ]
    Output: "There is a mouse right in front of you."
    Example 2:
    Image: An image of a brown chair on the left, a white table in the center, and a black mouse in the center.
    Message History: [
        {
            "agent": "user",
            "message": "What is in front of me?",
            "objects": {
                "left": {"chair": 1},
                "right": {"table": 1},
                "forward": {"mouse": 1}
            }
        },
        {
            "agent": "system",
            "message": "There is a mouse right in front of you."
        },
        {
            "agent": "user",
            "message": "What is the color of the mouse?",
            "objects": {
                "left": {"chair": 1},
                "right": {"table": 1},
                "forward": {"mouse": 1}
            }
        }
    ]
    Output: "The color of the mouse is black."
    Example 3:
    Image: "An image of an empty table"
    Message History: [
        {
            "agent": "user",
            "message": "What is in front of me?",
            "objects": {
                "left": {},
                "right": {},
                "forward": {}
            }
        }
    ]
    Output: "There is no object in front of you."
    
    """

    query = f"""
    Now, for the given image and the message history, generate an output and return only the output:
    {relevant_history}
    """

    img = Image.fromarray(image)
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt + query, img]
    )

    relevant_history.append({
        "agent": "system",
        "message": response.text
    })
    return relevant_history


def history_entry_generator(message, yolo_objects) -> dict:
    """
    Returns a history entry for the user
    :param message: the query message
    :param yolo_objects: the yolo objects
    :return: a dictionary of the history entry
    """

    return {
        "agent": "user",
        "message": message,
        "objects": {
            "left": yolo_objects["left"]["objects"],
            "right": yolo_objects["right"]["objects"],
            "forward": yolo_objects["forward"]["objects"]
        }
    }


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


class Transcriber:
    """
    >>> transcriber = Transcriber()
    >>> yolo_object = {
    ...     'left': {
    ...         'objects': {'person': 1},
    ...         'bounding_boxes': [[0.17, 6.64, 85.15, 173.78]]
    ...     },
    ...     'forward': {
    ...         'objects': {'bottle': 1, 'dining table': 1},
    ...         'bounding_boxes': [[107.25, 68.64, 115.07, 96.16],
    ...                            [63.68, 69.62, 230.90, 187.31]]
    ...     },
    ...     'right': {
    ...         'objects': {'chair': 1},
    ...         'bounding_boxes': [[188.56, 85.65, 250.09, 179.17]]
    ...     }
    ... }
    >>> transcriber.push_user_query("What is in front of me?", yolo_object)
    >>> print(transcriber.history)
    [{'agent': 'user', 'message': 'What is in front of me?', 'objects': {'left': {'person': 1}, 'right': {'chair': 1}, 'forward': {'bottle': 1, 'dining table': 1}}}]
    >>> # won't call it here but after pushing a query, just call the gemini response and pass in the image rgb
    >>> # it will update the transcriber history, and also return a response back
    >>> # response = transcriber.get_gemini_user_response()
    """
    def __init__(self, model_name: str = "gemini-2.0-flash", history_window: int = 10):
        self._history = []
        self._model_name = model_name
        self._history_window = history_window

    def push_user_query(self, message: str, yolo_objects):
        """
        Pushes a message entry into the transcribers current history
        :param message: the query message
        :param yolo_objects: the yolo objects
        :return: none
        """
        entry = {
            "agent": "user",
            "message": message,
            "objects": {
                "left": yolo_objects["left"]["objects"],
                "right": yolo_objects["right"]["objects"],
                "forward": yolo_objects["forward"]["objects"]
            }
        }
        self._history = self._history[-(self._history_window - 1):]
        self._history.append(entry)

    def get_gemini_user_response(self, image: np.ndarray) -> str:
        """
        Generates a response base off the image and the current message history

        An example of a message_history:
        message_history = [
            {
                "agent": "user",
                "message": "What is in front of me?",
                "objects": {
                    "left": {"chair": 1},
                    "right": {"table": 1},
                    "forward": {"mouse": 1}
                }
            },
            {
                "agent": "system",
                "message": "There is a mouse right in front of you"
            }
        ]

        It WILL mutate the history by adding an entry with the system response


        :param image: a rgb image with 3 channels
        :return: updates the current history and also returns the system response
        """

        prompt = """
        Example 1:
        Image: An image of a brown chair on the left, a white table in the center, and a black mouse in the center.
        Message History: [
            {
                "agent": "user",
                "message": "What is in front of me?",
                "objects": {
                    "left": {"chair": 1},
                    "right": {"table": 1},
                    "forward": {"mouse": 1}
                }
            }
        ]
        Output: "There is a mouse right in front of you."
        Example 2:
        Image: An image of a brown chair on the left, a white table in the center, and a black mouse in the center.
        Message History: [
            {
                "agent": "user",
                "message": "What is in front of me?",
                "objects": {
                    "left": {"chair": 1},
                    "right": {"table": 1},
                    "forward": {"mouse": 1}
                }
            },
            {
                "agent": "system",
                "message": "There is a mouse right in front of you."
            },
            {
                "agent": "user",
                "message": "What is the color of the mouse?",
                "objects": {
                    "left": {"chair": 1},
                    "right": {"table": 1},
                    "forward": {"mouse": 1}
                }
            }
        ]
        Output: "The color of the mouse is black."
        Example 3:
        Image: "An image of an empty table"
        Message History: [
            {
                "agent": "user",
                "message": "What is in front of me?",
                "objects": {
                    "left": {},
                    "right": {},
                    "forward": {}
                }
            }
        ]
        Output: "There is no object in front of you."

        """

        query = f"""
        Now, for the given image and the message history, generate an output and return only the output:
        {self._history}
        """

        img = Image.fromarray(image)
        response = client.models.generate_content(
            model=self._model_name,
            contents=[prompt + query, img]
        )

        response_text = response.text.strip("\n")
        self._history.append({
            "agent": "system",
            "message": response_text
        })

        self._history = self._history[-self._history_window:]

        return response_text

    @property
    def history(self):
        return self._history


if __name__ == '__main__':
    print("Initiating listening")
    # text = active_listening(model="base", device_index=1)
    # print(text)
    image_path = "../table.jpeg"
    image = cv2.imread(image_path)
    detected_objects = YOLO.yolo_object_detection_v11(image_path)

    transcriber = Transcriber()
    transcriber.push_user_query("What is to the left of me?", detected_objects)
    pprint.pprint(transcriber.history)
    response = transcriber.get_gemini_user_response(image)

    print(response)
    pprint.pprint(transcriber.history)
    # print(detected_objects)
    # history_entry = history_entry_generator("What is to the left of me?", detected_objects)
    # message_history = [history_entry]
    # new_history = gemini_user_response(message_history, image)
    # pprint.pprint(new_history)