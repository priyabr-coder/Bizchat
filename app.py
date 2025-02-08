from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)
CORS(app)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load or create vector store
vector_store = None
if os.path.exists("faiss_index"):
    vector_store = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)

# Data structure to simulate files list
uploaded_files = []  # To store the details of the uploaded files (URLs and PDFs)

def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
        content = [t.get_text(strip=True) for t in text]
        return content
    except requests.exceptions.RequestException as e:
        return str(e)

def get_text_from_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

import json

# Load the uploaded files from a JSON file
def load_uploaded_files():
    try:
        with open("uploaded_files.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []  # Return an empty list if the file doesn't exist

# Save the uploaded files to the JSON file
def save_uploaded_files():
    with open("uploaded_files.json", "w") as file:
        json.dump(uploaded_files, file)

# Initialize the uploaded_files list from the JSON file
uploaded_files = load_uploaded_files()

# Your routes here, make sure to call save_uploaded_files after any change


def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    return text_splitter.split_text(text)

def create_vector_store(data, file_info):
    global vector_store
    if vector_store is None:
        vector_store = FAISS.from_texts(data, embedding=embeddings)
    else:
        new_vectors = FAISS.from_texts(data, embedding=embeddings)
        vector_store.merge_from(new_vectors)

    vector_store.save_local("faiss_index")
    uploaded_files.append(file_info)  # Append the file info to the list

    # Save the updated list to the JSON file
    save_uploaded_files()  

def get_conversation_chain():
    prompt_template = """You are a friendly assistant. Your job is to help the user with their queries. 
    Refer to the knowledge base and provide precise answers based on the given context. 
    Do not provide incorrect information. If you don't know the answer, just say "I don't know." 
    Be polite and professional in your response.  
    Context:\n{context}\nQuestion:\n{Question}\nAnswer: """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "Question"])
    
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


@app.route('/')
def home():
    return "Welcome to knowlwdge repo!!", 200


@app.route('/scrape', methods=['POST'])
def scrape():
    data = request.json
    url = data.get("url", "")
    if not url:
        return jsonify({"error": "URL is required"}), 400

    content = scrape_website(url)
    if not content:
        return jsonify({"error": "No data found or failed to scrape website"}), 400

    create_vector_store(content, {"type": "url", "content": url})
    return jsonify({"message": "Website content stored successfully"})

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'files' not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400

    pdf_file = request.files['files']
    raw_text = get_text_from_pdfs([pdf_file])
    text_chunks = split_text(raw_text)

    create_vector_store(text_chunks, {"type": "pdf", "content": pdf_file.filename})
    return jsonify({"message": "PDF content stored successfully"})

@app.route('/query', methods=['POST'])
def query():
    global vector_store

    data = request.json
    user_question = data.get("message", "")
    if not user_question:
        return jsonify({"error": "Question is required"}), 400

    if not vector_store:
        return jsonify({"error": "No knowledge base found. Please scrape a website or upload PDFs."}), 400

    docs = vector_store.similarity_search(user_question)
    chain = get_conversation_chain()
    response = chain({"input_documents": docs, "Question": user_question}, return_only_outputs=True)

    return jsonify({"response": response["output_text"]})

# Fetch uploaded files (for the admin panel)
@app.route('/get_files', methods=['GET'])
def get_files():
    return jsonify({"files": uploaded_files})

# Remove file from vector store (for the admin panel)
@app.route('/remove_file', methods=['POST'])
def remove_file():
    file_info = request.json
    file_type = file_info.get("type")
    file_content = file_info.get("content")

    # Filter out the file info from the list
    global uploaded_files
    uploaded_files = [file for file in uploaded_files if file["content"] != file_content or file["type"] != file_type]

    # Save the updated list to the JSON file
    save_uploaded_files()

    # In reality, you'd remove the vectors here. This is a simple example.
    return jsonify({"message": f"File {file_content} removed successfully from the store"})

@app.route('/clear_vector_store', methods=['POST'])
def clear_vector_store():
    global vector_store
    if vector_store:
        vector_store = None  # Clear the vector store
       
        return jsonify({"message": "Vector store cleared successfully."})
    else:
        return jsonify({"message": "No vector store found to clear."}), 400


if __name__ == '__main__':
    app.run(port=5000, debug=True)
