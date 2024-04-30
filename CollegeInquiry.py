import streamlit as st
from translate import Translator
import speech_recognition as sr
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import os  # Import os for file operations
import base64  # Import base64 for encoding files

DB_FAISS_PATH = 'vectorstore/'
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that I don't know, sorry for the inconvenience don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""
# Initialize a dictionary to store responses and associated download links
response_download_links = {}
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Main code that retrieves relevant text content from the vector database
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=10002,
        temperature=1.0
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    result = response.get('result')
    source = response.get('source_documents')
    return result, source

def translate_text(text, lang):
    # Define the maximum character limit for translation (adjust as needed)
    max_char_limit = 500

    # Initialize an empty list to store translated segments
    translated_segments = []

    # Split the input text into segments based on the maximum character limit
    segments = [text[i:i + max_char_limit] for i in range(0, len(text), max_char_limit)]

    # Translate each segment individually
    for segment in segments:
        translator = Translator(to_lang=lang)
        translated_segment = translator.translate(segment)
        translated_segments.append(translated_segment)

    # Combine the translated segments into a single translated text
    translated_text = ' '.join(translated_segments)

    return translated_text


language_codes = {
    'English': 'en',
    'Hindi': 'hi',
    'Bengali': 'bn',
    'Telugu': 'te',
    'Marathi': 'mr',
    'Tamil': 'ta',
    'Urdu': 'ur',
    'Gujarati': 'gu',
    'Kannada': 'kn',
    'Odia (Oriya)': 'or',
    'Malayalam': 'ml',
    'Punjabi': 'pa'
}

st.title("🤖💬 MiniLaw")
st.caption("⛏️ Mineout your answers")
languages = ['English', 'Hindi', 'Bengali', 'Telugu', 'Marathi',
             'Tamil', 'Urdu', 'Gujarati', 'Kannada',
             'Odia (Oriya)', 'Malayalam', 'Punjabi']
selected_language = st.sidebar.selectbox('Select a language:', languages)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    if msg["role"] == "source":
        link = msg["content"]
        st.markdown(link, unsafe_allow_html=True)
    else:
        st.chat_message(msg["role"]).write(msg["content"])

# Create a placeholder for the "Speak" button at the bottom
speak_placeholder = st.empty()

# Add a "Record Audio" button
record_audio = speak_placeholder.button("Speak")

if record_audio:
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = r.listen(source)
        st.write("Audio recording complete.")

    # Convert the audio to text using the speech recognition library
    try:
        user_input = r.recognize_google(audio)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Call your QA function with the user's voice input
        response, sources = final_result(user_input)

        selected_language_code = language_codes[selected_language]

        translated_response = translate_text(response, selected_language_code)

        msg = {"role": "assistant", "content": translated_response}

        st.session_state.messages.append(msg)
        st.chat_message("assistant").write(msg["content"])


    except sr.UnknownValueError:
        st.write("Sorry, I could not understand the audio.")
    except sr.RequestError:
        st.write("Sorry, I encountered an error while processing the audio.")

# Continue with text input handling
if prompt := st.chat_input("User Input"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response, sources = final_result(prompt)

    if response not in response_download_links:
        selected_language_code = language_codes[selected_language]

        translated_response = translate_text(response, selected_language_code)

        # Store the response in the dictionary
        response_download_links[response] = sources

        # Display the response
        st.chat_message("assistant").write(translated_response)
        # st.text(translated_response)
        st.session_state.messages.append({"role": "assistant", "content": translated_response})

        # Render stored responses and their download links
        
