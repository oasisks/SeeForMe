# SeeForMe
A project to aid visual impaired people

## Setup
To set up the project, please run
```
pip install -r requirements.txt
```
either locally on your computer or through a virtual machine.

The code will also require you to download CUDA as our code is run through the 
GPU. Please go to the CUDA website and download ```v11.2```.

Also if the pip install does not work properly, please manually go to 
PyTorch website and manually select the correct PyTorch version compatible with ```v11.2```
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
The above command should work.

The Python Version our code ran on is on Python Version ```3.11```.
The code is also ran on a Windows 10 machine (my computer). 


## Running the code
The main entry point of the program is within main.py. You can just run it and it should work properly.
Do note that you probably do not have haptics set up. Thus, there is a parameter within ```main.py``` called
```USE_HAPTICS```. Set that to ```False```. 

As we are using Google Gemini as our LLM, we generated a .env file where the key is
```
GEMINI_KEY=API_KEY_REPLACE_HERE
```

For the code to work, you also need to download a model from YOLO:

```
https://github.com/ultralytics/ultralytics?tab=readme-ov-file
```
We used the ```yolo11x``` model for our implementation. 

You can also individually test each thing by going to each folder and run its files.

Within ```face_tracker```, you can run ```tracking.py``` to test only the things within face tracking.

Within ```transcription```, you can run ```transcriberr.py``` to see how we use Google Gemini and the history generation.

Within ```YOLO_test```, you can run ```YOLO.py``` to test the yolo model

You most likely won't be able to test the haptics as we created specialized hardware for the haptics.
