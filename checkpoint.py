# 1 - Importation des bibliothèques nécessaires
import speech_recognition as sr
import nltk
import string
import streamlit as st
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 2 - Charger le fichier texte et prétraiter les données
with open('surgical anatomy.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')

# Tokeniser en phrases
sentences = sent_tokenize(data)

# Prétraitement global
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(sentence):
    """Prétraite une phrase en supprimant les stopwords, la ponctuation et en appliquant la lemmatisation."""
    words = word_tokenize(sentence)
    words = [word.lower() for word in words if word.lower() not in stop_words and word not in string.punctuation]
    return [lemmatizer.lemmatize(word) for word in words]

# Prétraiter toutes les phrases
preprocessed_corpus = [preprocess(sentence) for sentence in sentences]

# Calculer la similarité et trouver la phrase la plus pertinente
def get_most_relevant_sentence(query):
    """Retourne la phrase la plus pertinente du texte en fonction de la requête."""
    query = preprocess(query)
    max_similarity = 0
    most_relevant_sentence = "Désolé, je n'ai pas trouvé de réponse pertinente."
    
    for sentence, preprocessed in zip(sentences, preprocessed_corpus):

        #Vérifiez si l'union des ensembles n'est pas vide
        union_length = len(set(query).union(preprocessed))
        if union_length == 0:
            continue # Ignorez les cas où l'union est vide

        similarity = len(set(query).intersection(preprocessed)) / float(len(set(query).union(preprocessed)))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = sentence

    return most_relevant_sentence

# 3 - Transcrire la parole en texte
def language_code(language):
    """Retourne le code de langue pour la reconnaissance vocale."""
    if language == "Français":
        return "fr"
    elif language == "Anglais":
        return "en"
    else:
        return "en"  # Par défaut

def transcribe_speech(api_choice, language, pause_and_resume=False):
    """Transcrit la parole en texte en utilisant l'API choisie."""
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Parlez maintenant...")
            
            if pause_and_resume:
                st.info("Enregistrement en pause...")
                return "Enregistrement en pause"
            
            # Enregistrement de l'audio
            audio_text = r.listen(source, timeout=10, phrase_time_limit=50)
            st.info("Transcription en cours...")

            # Utilisation de l'API choisie
            if api_choice == "Google":
                return r.recognize_google(audio_text, language=language_code(language))
            elif api_choice == "Sphinx":
                return r.recognize_sphinx(audio_text)
            elif api_choice == "Bing Speech (Azure)":
                st.warning("Intégration Bing/Azure Speech API à configurer.")
                return "API Bing/Azure Speech non encore implémentée"
            else:
                return "API non reconnue"
    except sr.UnknownValueError:
        return "Impossible de comprendre l'audio. Veuillez réessayer en parlant plus clairement."
    except sr.WaitTimeoutError:
        return "Temps d'attente dépassé. Veuillez parler après avoir cliqué sur le bouton."
    except sr.RequestError:
        return "Erreur de service. Vérifiez votre connexion Internet ou l'accès au service."
    except Exception as e:
        return f"Erreur inattendue : {e}"

# 4 - Fonction principale du chatbot
def chatbot(query, api_choice, language, pause_and_resume=False):
    """Gère les interactions avec le chatbot en texte ou voix."""
    if isinstance(query, str) and query.strip():  # Si l'entrée est du texte
        response = get_most_relevant_sentence(query)
    #elif isinstance(query, str) and not query:  # Si aucune entrée
        #response = "Aucune question entrée."
    else:  # Si l'entrée est vocale
        response = "Veuillez entrer une question valide."
    return response

# 5 - Création de l'application Streamlit
st.title("Chatbot Anatomie Chirurgicale")

# Sélection de l'API et de la langue
api_choice = st.selectbox("Choisissez l'API de reconnaissance vocale", ("Google", "Sphinx", "Bing Speech (Azure)"))
language = st.selectbox("Choisissez la langue", ("Français", "Anglais"))

# Mode d'entrée
mode = st.radio("Choisissez le mode d'entrée :", ("Texte", "Vocal"))

if mode == "Texte":
    user_input = st.text_input("Posez une question :")
    if user_input:
        response = chatbot(user_input, api_choice=api_choice, language=language)
        st.write("Réponse du chatbot :", response)

elif mode == "Vocal":
    if st.button("Cliquez pour parler"):
        user_query = transcribe_speech(api_choice=api_choice, language=language, pause_and_resume=False)
        if user_query:
            response = chatbot(user_query, api_choice=api_choice, language=language)
            st.write("Réponse du chatbot :", response)
