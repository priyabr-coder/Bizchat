import streamlit as st
import webbrowser
import speech_recognition as sr
from knowledgepagebackend.test6 import scrape_website, create_vector_store, get_text, get_text_chunks, query_vector_store, load_existing_index

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio).lower()
        return text
    except sr.UnknownValueError:
        st.write("Could not understand the audio")
        return None
    except sr.RequestError:
        st.write("Error connecting to the speech recognition service")
        return None

def update_choice_from_speech(spoken_text):
    menu = ["Knowledge Base", "Leave Management System"]
    for option in menu:
        if spoken_text in option.lower():
            st.session_state.page_choice = option
            st.success(f"Selected via Voice: {option}")
            return
    st.error("Invalid voice input. Please try again.")

def speech_input_page():
    st.header("Select a System")

    if "page_choice" not in st.session_state:
        st.session_state.page_choice = None

    if st.button("🎙 Speak to Select"):
        spoken_text = recognize_speech()
        if spoken_text:
            update_choice_from_speech(spoken_text)

    if st.session_state.page_choice:
        vector_store = load_existing_index()
        route_engine(st.session_state.page_choice, vector_store)

def route_engine(choice, vector_store):
    if choice == "Knowledge Base":
        knowledge_base_page(vector_store)
    elif choice == "Leave Management System":
        leave_management_url = "https://voice-leave-app.onrender.com"
        redirect_to_leave_management(leave_management_url)
    else:
        st.error("Invalid option selected. Please choose a valid menu item.")

def knowledge_base_page(vector_store):
    st.header("Knowledge Base")
    menu = ["Scrape Website", "Upload PDF", "Ask a Question"]
    sub_choice = st.sidebar.selectbox("Select a Sub-Option", menu, key="sub_option")

    if sub_choice == "Scrape Website":
        scrape_website_page(vector_store)
    elif sub_choice == "Upload PDF":
        upload_pdf_page(vector_store)
    elif sub_choice == "Ask a Question":
        ask_question_page(vector_store)

def scrape_website_page(vector_store):
    st.subheader("Scrape a Website")
    url = st.text_input("Enter the URL to scrape")
    if st.button("Scrape and Process"):
        if vector_store:
            st.warning("Knowledge Repo already exists. You can query it below.")
        else:
            with st.spinner("Scraping website and processing content..."):
                content = scrape_website(url)
                if content:
                    create_vector_store(content)
                    st.success("Website scraped and processed successfully!")
                else:
                    st.error("No data found on the website.")

def upload_pdf_page(vector_store):
    st.subheader("Upload PDF Files")
    pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Submit and Process") and pdf_docs:
        if vector_store:
            st.warning("Knowledge Repo already exists. You can query it below.")
        else:
            with st.spinner("Processing PDF files..."):
                raw_text = get_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                create_vector_store(text_chunks)
                st.success("PDF files processed successfully!")

def ask_question_page(vector_store):
    st.subheader("Ask a Question")
    user_question = st.text_input("Enter your question")
    if st.button("Submit Query"):
        with st.spinner("Fetching answer..."):
            answer = query_vector_store(user_question, vector_store)
            if answer:
                st.write("Answer:", answer)
            else:
                st.error("No answer found for your query.")

def redirect_to_leave_management(leave_management_url):
    st.write("Redirecting to the Leave Management System...")
    webbrowser.open(leave_management_url)

def main():
    st.set_page_config(page_title="Invincix Portal", layout="centered")
    st.title("Invincix Portal")

    speech_input_page()

if __name__ == "__main__":
    main()
