import pyttsx3

# Initialize the pyttsx3 engine
engine = pyttsx3.init()
dis='common cold'
des='you get it when you get it'
def speak(dis,des):
    engine.say(dis)
    engine.say(des)
    engine.runAndWait()

speak(dis,des)