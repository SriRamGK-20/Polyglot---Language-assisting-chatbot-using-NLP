

# In[1]:


#!pip install wikipedia
#!pip install translate
#!pip3 install pywin32 pypiwin32 pyttsx3


# In[16]:


import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# In[18]:


import speech_recognition as sr
r = sr.Recognizer()


# In[19]:


import nltk
import wikipedia
import nltk
import numpy as np
import random
import string


# In[20]:


from translate import Translator

translatorta = Translator(to_lang="ta")
# this code shows how the code for traslation works for language tamil

translatores = Translator(to_lang="es")
# this code shows how the code for traslation works for language spanish

translatorja = Translator(to_lang="ja")
# this code shows how the code for traslation works for language japanese

translatorfr = Translator(to_lang="fr")
# this code shows how the code for traslation works for language french


# In[21]:


import warnings
warnings.filterwarnings("ignore")


# In[22]:


f = open('chatbot.txt', 'r', errors='ignore')
raw = f.read()
raw = raw.lower()


# In[23]:


#nltk.download() # for downloading packages
#nltk.download('punkt')  # first-time use only
#nltk.download('wordnet')  # first-time use only


# In[24]:


sent_tokens = nltk.sent_tokenize(raw)
# converts to list of sentences
word_tokens = nltk.word_tokenize(raw)


# In[25]:


sent_tokens[:2]
word_tokens[:5]


# In[26]:


lemmer = nltk.stem.WordNetLemmatizer()


# In[27]:


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


# In[28]:


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there"]


# In[29]:


def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[31]:


def wiki_response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)

    # TF-IDF are word frequency scores that try to highlight words that are more interesting,
    # e.g. frequent in a document but not across documents.
    # The TfidfVectorizer will tokenize documents, learn the vocabulary and
    # inverse document frequency weightings, and allow you to encode new documents.
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')

    # Learn vocabulary and idf, return term-document matrix
    # Returns X : Tf-idf-weighted sparse matrix, [n_samples, n_features]
    tfidf = TfidfVec.fit_transform(sent_tokens)
    # print (tfidf.shape)

    # Cosine similarity is a measure of similarity between two non-zero vectors.
    # Using this formula we can find out the similarity between any two documents d1 and d2.
    # Cosine Similarity (d1, d2) =  Dot product(d1, d2) / ||d1|| * ||d2||
    vals = cosine_similarity(tfidf[-1], tfidf)

    # function is used to perform an indirect sort along the given axis using the algorithm
    # specified by the kind keyword. It returns an array of indices of the same shape as arr
    # that would sort the array.
    idx = vals.argsort()[0][-2]

    # Returns a new array that is a one-dimensional flattening of this array (recursively).
    # That is, for every element that is an array, extract its elements into the new array.
    # If the optional level argument determines the level of recursion to flatten.
    flat = vals.flatten()

    flat.sort()
    # flat is sorted
    req_tfidf = flat[-2]
    # second element from reverse order is taken as req_tfidf

    if (req_tfidf == 0):
        # check if req_tfidf is zero if yes then don the esceptional handling for wikipedia
        try:
            robo_response += wikipedia.summary(user_response, 4)
        except wikipedia.DisambiguationError as e:
            robo_response = robo_response + "I am sorry! Please enter detailed query."
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response


# In[32]:


flag = True
print("""POLYGLOT: I am POLYGLOT. I am your personal language assistant.
         As i know many things please enter your query precisely.
         If you want to exit, say Bye!
         Say **general** for Usual chatbot and **Polyglot** for searching specific query""")
engine.say("""I am POLYGLOT. I am your personal language assistant.""")
engine.runAndWait()


# In[37]:


#!pip install PyAudio
import pyjokes
import webbrowser

import chatterbot
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

import time
import datetime
  
def net_jokes(lang):
    categ=["neutral","all"]
    rand=random.randint(0,1)
    lol = pyjokes.get_joke(language=lang, category=categ[rand])
    print(lol)
    engine.say(lol)
    
    
def general_chat(user_response):
    # Give a name to the chatbot “general”
    # and assign a trainer component.
    chatbot=ChatBot('general')
     
    # Create a new trainer for the chatbot
    trainer = ChatterBotCorpusTrainer(chatbot)
      
    # Now let us train our bot with multiple corpus
    trainer.train("chatterbot.corpus.english.greetings","chatterbot.corpus.english.conversations" )
    
    flag=True
    while(flag==True):
        if ("bye" or "bhai" not in user_response):
            response = chatbot.get_response(user_response)
            print(response)
            engine.say(response)
            engine.runAndWait()
        else:
            response = chatbot.get_response(user_response)
            print(response)
            engine.say(response)
            engine.runAndWait()
        flag=False


def tellTime():
# This method will give the time
    time = str(datetime.datetime.now())
    # the time will be displayed like this "2020-06-05 17:50:14.582630"
    # nd then after slicing we can get time
    print(time)
    hour = time[11:13]
    min = time[14:16]
    print("The time is " + hour + " Hours and " + min + " Minutes")
    engine.say("The time is " + hour + " Hours and " + min + " Minutes")
    
    
# In[35]:


while (flag == True):
    with sr.Microphone() as source:
        print("User:", end="")
        audio = r.listen(source)
        try:
            user_response = r.recognize_google(audio)
        except:
            time.sleep(2)
            print()
            engine.say("I can't recognize what you said, please speak clearly.")
            print("""Polyglot: I'm gonna go cause you were not audible to me :(""")
            engine.runAndWait()
            continue
        
    user_response = user_response.lower()
    
    #if ("general" in user_response):
        #general_chat(user_response)
        
    #if ("polyglot" in user_response):
        
    if user_response == 'bhai':
           user_response = 'bye'
    print(user_response)
        
    if (user_response != ('bye')):
        if ("translate" in user_response):
            if ("tamil" in user_response):
                xy = len(user_response)
                translation = translatorta.translate(user_response[10:xy - 8])
                print("POLYGLOT:", translation)
                engine.say(translation)
                engine.runAndWait()
            if ("spanish" in user_response):
                xy = len(user_response)
                translation = translatores.translate(user_response[10:xy - 10])
                print("POLYGLOT:", translation)
                engine.say(translation)
                engine.runAndWait()
            if ("japanese" in user_response):
                xy = len(user_response)
                translation = translatorja.translate(user_response[10:xy - 11])
                print("POLYGLOT:", translation)
                engine.say(translation)
                engine.runAndWait()
            if ("french" in user_response):
                xy = len(user_response)
                translation = translatorfr.translate(user_response[10:xy - 10])
                print("POLYGLOT:", translation)
                engine.say(translation)
                engine.runAndWait()
                    
        if (user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            print("POLYGLOT: You are welcome..")
            engine.say("you are welcome")
            engine.runAndWait()
                
#new
        if ("joke" in user_response):
            #print("Which language")
            #engine.say("Which language?")
            f=1
        while(f==1):
            if ("english" in user_response):
                f=0
                net_jokes("en")
            elif ("german" in user_response):
                f=0
                net_jokes("de")
            elif ("spanish" in user_response):
                f=0 
                net_jokes("es")
        engine.runAndWait()
        
        if ("time" in user_response):
            tellTime()
        engine.runAndWait()        
                
        if ("search" in user_response):
            webbrowser.open("https://www.google.co.in/search?q="+user_response)
            #engine.runAndWait()
            break
        
        elif ("play" in user_response):
            webbrowser.open("https://www.youtube.com/results?search_query="+user_response)
            #engine.runAndWait()
            break

#new end         
            
        else:
            if greeting(user_response) is not None:
                print("POLYGLOT: " + greeting(user_response))
                engine.say(greeting(user_response))
                    
            elif ("what" in user_response):
                print("POLYGLOT: ", end="")
                print(wiki_response(user_response))
    
                sent_tokens.remove(user_response)
                engine.say(wiki_response(user_response))
                engine.runAndWait()
                    
    else:
        flag = False
        print("POLYGLOT: Bye! take care..")
        engine.say("Bye! take care..")
        engine.runAndWait()


# In[ ]:




