import os
import time
import datetime
import platform
import numpy as np
import transformers
import speech_recognition as sr
from gtts import gTTS

class ChatBot():                                                    # Beginning of the AI
    def __init__(self, name):
        print("----- starting up", name, "-----")
        self.text = None
    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            print("listening...")
            audio = recognizer.listen(mic)
        try:
            self.text =  recognizer.recognize_google(audio)
            print("me --> ", self.text)
        except:
            print("me --> ERROR")
    
    #@staticmethod -- disable static method, using class method to replace
    def text_to_speech(self, text):
        print("AI --> ", text)
        speaker = gTTS(
            text=text, 
            lang="en", 
            slow=False
        )
        speaker.save("res.mp3")
        statbuf = os.stat("res.mp3")
        mbytes = statbuf.st_size / 1024
        duration = mbytes / 200
        #if you have a macbook->afplay or for windows use->start
        if platform.system() == "Windows":
            os.system("start res.mp3")
        elif platform.system() == "Darwin":
            os.system("afplay res.mp3")
        else:
            pass
        # os.system("close res.mp3")
        time.sleep(int(50*duration))
        os.remove("res.mp3")

    #@staticmethod -- disable static method, using class method to replace
    def action_time(self):
        return datetime.datetime.now().time().strftime('%H:%M')

    def wake_up(self, text):
        return True if self.name in text.lower() else False

if __name__ == "__main__":                                          # Running the AI
    ai = ChatBot(name="Dev")
    nlp = transformers.pipeline(
        "conversational",
        model="microsoft/DialoGPT-medium"
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    ex=True
    res = None
    while ex:
        ai.speech_to_text()
        if ai.wake_up(ai.text) is True:                             ## wake up
            res = "Hello I am Dev the AI, what can I do for you?"
        elif "time" in ai.text:                                     ## do any action      
            res = ai.action_time()
        elif any(i in ai.text for i in ["thank","thanks"]):         ## respond politely
            res = np.random.choice([
                "you're welcome!",
                "anytime!", 
                "no problem!",
                "cool!", 
                "I'm here if you need me!",
                "mention not!"
            ])
        elif any(i in ai.text for i in ["exit","close"]):           ## respond exit
            res = np.random.choice([
                "Tata",
                "Have a good day",
                "Bye",
                "Goodbye",
                "Hope to meet soon",
                "peace out!"
            ])
            ex = False
        else:                                                       ## conversation
            if ai.text == "ERROR":
                res = "Sorry, come again?"
            else:
                chat = nlp(
                    transformers.Conversation(ai.text), 
                    pad_token_id=50256
                )
                res = str(chat)
                res = res[res.find("bot >> ")+6:].strip()

        ai.text_to_speech(res)

    print("----- Closing down Dev -----")