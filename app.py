import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

vectorstore = None

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdf(file_path):
    global vectorstore
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return len(chunks)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Dosya seçilmedi"}), 400
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    chunks = process_pdf(file_path)
    return jsonify({"message": f"PDF işlendi. {chunks} parça oluşturuldu."})

@app.route("/ask", methods=["POST"])
def ask():
    global vectorstore
    if vectorstore is None:
        return jsonify({"error": "Önce bir PDF yükleyin"}), 400
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Soru boş olamaz"}), 400
    
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    response = llm.invoke(
        f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}"
    )
    
    return jsonify({"answer": response.content})

if __name__ == "__main__":
    app.run(debug=True)