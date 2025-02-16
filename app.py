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
from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from IPython.display import HTML, display
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from PIL import Image
import json
import os
from pdf2image import convert_from_path
import pytesseract
import io
import base64
import uuid
import re
import chromadb
from pathlib import Path
import logging

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)
CORS(app)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load or create vector store
vector_store = None
fname = None
fpath = None
text_summaries = []
texts = []
table_summaries = []
tables = []
img_base64_list = []
image_summaries = []


# if os.path.exists("faiss_index"):
#     vector_store = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
# The vectorstore to use to index the summaries
persistent_client = chromadb.PersistentClient()
vector_store = Chroma(
    client=persistent_client,
    collection_name="mm_rag_cj_blog", embedding_function=embeddings
)
# Data structure to simulate files list
uploaded_files = []  # To store the details of the uploaded files (URLs and PDFs)
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_pdfs(folder_path, chunking_strategy="by_title", extract_images_in_pdf=True, 
                 max_characters=3000, new_after_n_chars=2800, combine_text_under_n_chars=2000, 
                 image_output_dir_path="img_path"):
    pdf_elements = {}
    pdf_folder = Path(folder_path)
    
    for pdf_file in pdf_folder.glob("*.pdf"):
        pdf_elements = partition_pdf(
            str(pdf_file),
            chunking_strategy=chunking_strategy,
            extract_images_in_pdf=extract_images_in_pdf,
            max_characters=max_characters,
            new_after_n_chars=new_after_n_chars,
            combine_text_under_n_chars=combine_text_under_n_chars,
            image_output_dir_path=image_output_dir_path
        )
    
    return pdf_elements

# Example usage
pdf_folder_path = "./uploads"
image_output_dir_path = "./images"
# Extract elements from PDF
# def extract_pdf_elements(path, fname):
   
#     """
#     Extract images, tables, and chunk text from a PDF file.
#     path: File path, which is used to dump images (.jpg)
#     fname: File name
#     """
#     return partition_pdf(
#         filename=path,
#         extract_images_in_pdf=False,
#         infer_table_structure=True,
#         chunking_strategy="by_title",
#         max_characters=4000,
#         new_after_n_chars=3800,
#         combine_text_under_n_chars=2000,
#         image_output_dir_path=path,
#     )

# Categorize elements by type
def categorize_elements(raw_pdf_elements):
    """
    Categorize extracted elements from a PDF into tables and texts.
    raw_pdf_elements: List of unstructured.documents.elements
    """
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables


# Generate summaries of text elements
def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    Summarize text elements
    texts: List of str
    tables: List of str
    summarize_texts: Bool to summarize texts
    """

    # Prompt
    prompt_text = """You are an assistant named Thrylox tasked with summarizing tables and text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text or table elements. \
    Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Text summary chain
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Initialize empty summaries
    text_summaries = []
    table_summaries = []

    # Apply to text if texts are provided and summarization is requested
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts

    # Apply to tables if tables are provided
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
   
    return text_summaries, table_summaries

def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def image_summarize(img_base64, prompt):
    """Make image summary"""
    chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5,max_tokens=1024)

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content

def generate_img_summaries(path):
    """
    Generate summaries and base64 encoded strings for images
    path: Path to list of .jpg files extracted by Unstructured
    """
   
    # Store base64 encoded images
    img_base64_list = []

    # Store image summaries
    image_summaries = []

    # Prompt
    prompt = """You are an assistant named Thrylox tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""
    ##os.listdir(path)
    # Apply to images
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))

    return img_base64_list, image_summaries



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
    images_text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        # Extract images and perform OCR
        images = convert_from_path(pdf)
        for img in images:
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            images_text += pytesseract.image_to_string(img) + "\n"

    return text + "\n" + images_text


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
    prompt_template = """You are a friendly and intelligent assistant.Your name is Thrylox. Using the information contained in the context, provide a comprehensive answer that directly addresses the question. Ensure that your response includes the exact keywords and terminology used by the user where relevant.
    Learn from the conversation history to enhance response accuracy and relevance over time.Keep the response concise, professional, and relevant.
    Do not provide incorrect or speculative information. If the answer cannot be deduced from the context, indicate that appropriately.  
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


@app.route('/process_pdf', methods=['GET'])
def process_pdf():
    global texts, tables, text_summaries, table_summaries, img_base64_list, image_summaries, retriever_multi_vector_img
    logging.info("Starting PDF processing")
    parsed_pdfs = process_pdfs(pdf_folder_path)
    logging.info("PDFs parsed successfully")
    texts, tables = categorize_elements(parsed_pdfs)
    logging.info("Elements categorized into texts and tables")
    # # Optional: Enforce a specific token size for texts
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
       chunk_size=4000, chunk_overlap=0
    )
    joined_texts = " ".join(texts)
    texts_4k_token = text_splitter.split_text(joined_texts)
    logging.info("Texts split into 4k tokens")
   
    # # Get text, table summaries
    text_summaries, table_summaries = generate_text_summaries(
         texts_4k_token, tables, summarize_texts=True
    )
    logging.info("Text and table summaries generated")
    # Image summaries
    img_base64_list, image_summaries = generate_img_summaries("./figures")
    # Create retriever
    logging.info("Image summaries generated")
    retriever_multi_vector_img = create_multi_vector_retriever(
            vector_store,
            text_summaries,
            texts,
            table_summaries,
            tables,
            image_summaries,
            img_base64_list,
    )
    logging.info("Multi-vector retriever created successfully")
    return jsonify({"message": "PDF content processed successfully"})

    


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'files' not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400

    pdf_file = request.files['files']
    upload_folder = './uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, pdf_file.filename)
    pdf_file.save(file_path)   
   
   
    return jsonify({"message": "PDF content stored successfully"})
    

    
  


    


   
    

 



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





def create_multi_vector_retriever(
    vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images
):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """
   
    # Initialize the storage layer
    store = InMemoryStore()
    id_key = "doc_id"

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
   
    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever, doc_summaries, doc_contents):
      
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    # Add texts, tables, and images
    # Check that text_summaries is not empty before adding
    print("**************",text_summaries, texts,image_summaries)
    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    # Check that table_summaries is not empty before adding
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    # Check that image_summaries is not empty before adding
    if image_summaries:
        add_documents(retriever, image_summaries, images)

   

    return retriever







def plt_img_base64(img_base64):
    """Disply base64 encoded string as image"""
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    display(HTML(image_html))


def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}

def img_prompt_func(data_dict):
    """
    Join the context into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    # Adding the text for analysis
    text_message = {
        "type": "text",
        "text": (
            "You are a helpful assistant named Thrylox tasking with providing advice.\n"
            "You will be given a mixed of text, tables, and image(s) usually of application screens with text instructions.\n"
            "Use this information to provide advice related to the user question. \n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]

def multi_modal_rag_chain(retriever):
    """
    Multi-modal RAG chain
    """

    # Multi-modal LLM
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5,max_tokens=1024)

    # RAG pipeline
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    return chain



@app.route('/query', methods=['POST'])
def query():
    global retriever_multi_vector_img

    data = request.json
    user_question = data.get("message", "")
    if not user_question:
        return jsonify({"error": "Question is required"}), 400

    # if not vector_store:
    #     return jsonify({"error": "No knowledge base found. Please scrape a website or upload PDFs."}), 400

    # docs = vector_store.similarity_search(user_question)
    # chain = get_conversation_chain()
    chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)
    response = chain_multimodal_rag.invoke(user_question)
    # response = chain({"input_documents": docs, "Question": user_question}, return_only_outputs=True)

    return jsonify({"response": response})



if __name__ == '__main__':
    app.run(port=5000, debug=True)