import keyboard
import speech_recognition as sr
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv
import pyttsx3
import key

engine = pyttsx3.init()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = key.YourKey
load_dotenv()

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.7)

prompt = PromptTemplate(   
    input_variables= ["question"],
    template="Your are a bot, only answer what i have ask. Do not repeat yourself. Do not use any Emoji. Answer this question in a happy manner:" "{question}"
)

class Chatbot:
    # Constructor
    def __init__(self, name):
        print("----- Starting up -----")
        #self.userInput = ""
        self.audioText = ""
        self.bot_response = ""

    def response(self, user_input):
        if user_input != "":
            hub_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
            self.bot_response = hub_chain.run(self.audioText)
            print(self.bot_response)
            return self.bot_response
            # verbose parameter enables detailed output
            # Beneficial for understanding the abstraction that LangChain provides under the hood,
            # while executing our query.

        else:
            print("No input found please try again")

    def speech_to_text(self):
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = 300
        recognizer.pause_threshold = 0.5

        with sr.Microphone() as mic:
            print("Listening...")
            audio = recognizer.listen(mic)
        try:
            self.audioText = recognizer.recognize_google(audio, language = "en-US")
            print("You: ", self.audioText)
        except Exception:
            pass

# Boot the AI
if __name__ == "__main__":
    Ai_name = input("Name your bot: ")
    ai = Chatbot(Ai_name)
    ex = True
    while ex:
        ai.speech_to_text()
        #ai.userInput= input("You: ")
        #ai.response(ai.userInput)
        ai.response(ai.audioText)

    print("----- Closing down chatbot -----")