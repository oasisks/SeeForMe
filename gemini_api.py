from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import numpy as np
load_dotenv()

import cv2


# DOCS: https://github.com/googleapis/python-genai 
from PIL import Image
client = genai.Client(api_key=os.getenv("GEMINI_KEY"))


def gemini_image_description(img_array: np.ndarray, update_state: Dict[str, str]) -> str:
    """
    Generate a description of the image using Gemini API.
    """
    
    img = Image.fromarray(img_array)
    prompt = """
    Example 1:
    Image: An image of a pink bicycle, two clear bottles, one cup, and one bowl on a table.
    State: {"bicycle": "1", "bottle": "added 2", "cup": "removed 1"}
    Description: There is a pink bicycle parked by the wall. Two clear bottles were added to the table, and a cup has been removed.

    Example 2:
    Image: An image of a pink bicycle, two clear bottles, one cup, and one bowl on a table.
    State: {}
    Description: No changes detected.

    Example 3:
    Image: An image of an empty table.
    State: None
    Description: No objects detected.

    Example 4:
    Image: An image of an empty table.
    State: {}
    Description: No changes detected.

    """ + f"Now, for the given image, describe the objects in the current state based on the dictionary: {update_state}. Return only the description.\n" 
   
    model_name = "gemini-2.0-flash"
    response = client.models.generate_content(
        model= model_name,
        contents = [prompt, img]
    )
    return response.text

if __name__ == "__main__":

    # Load an image
    image = cv2.imread('table.jpeg')
    # Convert the image to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # same format as main.py

    print(gemini_image_description(img, {
        "person": "1",
        "bottle": "removed 2",
    }))

    print(gemini_image_description(img, {
       
    }))