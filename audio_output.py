import pyttsx3 

def text_to_speech(text: str, voice_id=None) -> None:
    """
    text: text to be spoken 
    voice_id: id of the voice to be
    Example voices: 1. com.apple.voice.compact.en-GB.Daniel 14
    2. com.apple.voice.compact.en-US.Samantha 108
    """
    engine = pyttsx3.init()
    
    # engine.setProperty('rate', 200)     # setting up new voice rate
    # engine.setProperty('volume', 1.0)    # setting up volume level  between 0 and 1

    voices = engine.getProperty('voices')       #getting details of current voice
    if voice_id:
        engine.setProperty('voice', voices[voice_id].id)

    engine.say(text)
    engine.runAndWait()
    engine.stop()

# text_to_speech("tôi muốn ăn", 73)
# text_to_speech("Hello, how are you?",  108)
# text_to_speech("Hello, how are you?", 14)
