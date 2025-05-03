from google import genai
from google.genai import types

import os
from dotenv import load_dotenv

load_dotenv()

# DOCS: https://github.com/googleapis/python-genai 
from PIL import Image
client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

def gemini_description_image(img_array, update_state):
    img = Image.open(img_array)
    state = {
        "person": "1",
        "bottle": "removed 2",
    }
    model_name = "gemini-2.0-flash"
    response = client.models.generate_content(
        model= model_name,
        contents = [
            f"Describe the image focusing only on the objects in the current state in the dictionary: {state}.", img
        ]
    )
    print("here", response.text)

gemini_description_image("table.jpeg", {})