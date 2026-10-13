 !pip install --upgrade langchain langchain-community langchain-google-genai langchain-huggingface python-dotenv sentence-transformers faiss-cpu protobuf
_________________________________________________________________________________________________

import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyC9iV96_5oEECqmxF3qURfApM-hHGc8L_k"
os.environ["NGROK_AUTH_TOKEN"] = "305a1DdX0Ut3GGDLuwx43irVn3O_7Fs44mUzWzY6P1dr5aSQn"
____________________________________________________________________________________________________

from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_API_KEY = "AIzaSyC9iV96_5oEECqmxF3qURfApM-hHGc8L_k"

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)

response = llm.invoke("Hello Gemini! Can you confirm you are working?")
print(response)
____________________________________________________________________________________________________

from google.colab import files
uploaded = files.upload()
___________________________________________________________________________________________________

!pip install PyPDF2 pandas

import PyPDF2
import pandas as pd

def extract_text_from_pdf(file_path, doc_type):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return [{"text": text.strip(), "type": doc_type}]

# Extract text from each PDF
data = []
data += extract_text_from_pdf("career tips.pdf", "career_tips")
data += extract_text_from_pdf("job description.pdf", "job_description")
data += extract_text_from_pdf("sample resume.pdf", "sample_resume")

# Create dataframe
df = pd.DataFrame(data)

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)
print("✅ Text extracted and saved to cleaned_data.csv")
df.head()
____________________________________________________________________________________________________

print(df.shape)      # Shows number of rows and columns
print(df.columns)    # Shows column names
df.head()            # Preview first few rows
____________________________________________________________________________________________________

!pip uninstall -y google-ai-generativelanguage langchain-google-genai requests protobuf
____________________________________________________________________________________________________

# Step 1: Core dependencies
!pip install requests==2.32.3 protobuf==4.25.3

# Step 2: LangChain + FAISS + HuggingFace
!pip install langchain langchain-community sentence-transformers faiss-cpu

# Step 3: LangChain Google Gemini integration (latest compatible)
!pip install langchain-google-genai==2.1.12 google-ai-generativelanguage==0.7.0
____________________________________________________________________________________________________

from langchain_community.embeddings import HuggingFaceEmbeddings

# Use lightweight embedding model
embed = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
____________________________________________________________________________________________________

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Assuming df is already loaded
docs = [
    Document(
        page_content=row['text'],
        metadata={"type": row['type']}
    )
    for _, row in df.iterrows()
]

# Build FAISS index
index = FAISS.from_documents(docs, embed)
index.save_local("faiss_jobmate_index_v2")

print("✅ FAISS index created and saved locally")
____________________________________________________________________________________________________

# Install / upgrade required packages
!pip install --upgrade langchain langchain-community sentence-transformers faiss-cpu protobuf

# Imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Step 1: Initialize embedding model
embed = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Step 2: Reload FAISS index safely
index = FAISS.load_local(
    "faiss_jobmate_index_v2",
    embeddings=embed,
    allow_dangerous_deserialization=True  # <-- Required for loading pickle files safely
)

# Step 3: Create retriever for RAG
retriever = index.as_retriever(search_type="similarity", search_kwargs={"k": 3})
print("✅ FAISS retriever ready and working")
___________________________________________________________________________________________________

# run this cell first
!pip install --upgrade pip
!pip install -U protobuf
!pip install -U langchain langchain-community langchain-huggingface langchain-google-genai google-generativeai
!pip install -U sentence-transformers faiss-cpu
!pip install -U python-dotenv pyngrok fastapi uvicorn nest_asyncio pydantic requests
___________________________________________________________________________________________________

# Install / upgrade required packages
!pip install --upgrade pip
!pip install -U protobuf
!pip install -U langchain langchain-community langchain-huggingface langchain-google-genai google-generativeai
!pip install -U sentence-transformers faiss-cpu
!pip install -U python-dotenv pyngrok fastapi uvicorn nest_asyncio pydantic requests

# ✅ Directly assign your key (no .env needed)
GOOGLE_API_KEY = "AIzaSyC9iV96_5oEECqmxF3qURfApM-hHGc8L_k"

# Create Gemini LLM wrapper
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",   # change to "models/gemini-1.5-pro" if you prefer
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)

# Quick sanity check
response = llm.invoke("Say hello briefly.")
print("✅ Gemini initialized and working:", response)
____________________________________________________________________________________________________
# Embeddings + FAISS load (robust imports)
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception:
    from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_community.vectorstores import FAISS
except Exception:
    from langchain.vectorstores import FAISS

from langchain.schema import Document
import os, sys

# create embedder
embed = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
print("✅ Embeddings ready")

# Ensure df is present (if you already made cleaned_data.csv)
import pandas as pd
if 'df' not in globals():
    if os.path.exists("cleaned_data.csv"):
        df = pd.read_csv("cleaned_data.csv")
        print("Loaded cleaned_data.csv into df")
    else:
        raise RuntimeError("No variable `df` and cleaned_data.csv not found. Extract PDFs into df first.")

# If a saved index exists, load it, else create and save it
index_folder = "faiss_jobmate_index_v2"
if os.path.exists(os.path.join(index_folder, "index.faiss")):
    print("Loading existing FAISS index from", index_folder)
    index = FAISS.load_local(index_folder, embeddings=embed, allow_dangerous_deserialization=True)
else:
    # build index from df — df must have 'text' and optionally 'type' columns
    docs = [Document(page_content=row['text'], metadata={"type": row.get('type', '')}) for _, row in df.iterrows()]
    index = FAISS.from_documents(docs, embed)
    index.save_local(index_folder)
    print("Created and saved FAISS index to", index_folder)

retriever = index.as_retriever(search_type="similarity", search_kwargs={"k": 3})
print("✅ FAISS retriever ready (k=3). Documents:", len(df))
___________________________________________________________________________________________________
!pip install langdetect
___________________________________________________________________________________________________
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Resume improvement prompt (multilingual)
resume_template = """
You are an expert career coach. Improve the resume to match the given job description.

IMPORTANT: Respond in the same language as {lang}.

Job Description:
{job}

Resume:
{resume}

Knowledge Context:
{context}

Output: Provide 3-5 concise ATS-friendly bullet points (one per line).
"""
resume_prompt = PromptTemplate(input_variables=["job","resume","context","lang"], template=resume_template)
resume_chain = LLMChain(llm=llm, prompt=resume_prompt)

# Interview generation prompt (multilingual)
interview_template = """
You are an interview coach. Based on the job description and the knowledge context,
generate 5 likely interview questions and provide short model answers (2-3 sentences each).

IMPORTANT: Respond in the same language as {lang}.

Job Description:
{job}

Knowledge Context:
{context}
"""
interview_prompt = PromptTemplate(input_variables=["job","context","lang"], template=interview_template)
interview_chain = LLMChain(llm=llm, prompt=interview_prompt)

print("✅ Multilingual LLM chains created (resume_chain, interview_chain).")
___________________________________________________________________________________________________
from langdetect import detect

def generate_resume_edits(jd_text: str, resume_text: str) -> str:
    # Detect language of JD
    try:
        lang = detect(jd_text)
    except:
        lang = "en"

    # Retrieve relevant context
    docs = retriever.get_relevant_documents(jd_text)
    context = "\n\n".join([d.page_content for d in docs])

    # Run chain with multilingual instruction
    return resume_chain.run({
        "job": jd_text,
        "resume": resume_text,
        "context": context,
        "lang": lang  # pass detected language
    })


def generate_interview_qs(jd_text: str) -> str:
    # Detect language of JD
    try:
        lang = detect(jd_text)
    except:
        lang = "en"

    # Retrieve relevant context
    docs = retriever.get_relevant_documents(jd_text)
    context = "\n\n".join([d.page_content for d in docs])

    # Run chain with multilingual instruction
    return interview_chain.run({
        "job": jd_text,
        "context": context,
        "lang": lang  # pass detected language
    })

print("✅ Multilingual generate_resume_edits and generate_interview_qs defined.")

____________________________________________________________________________________________________
jd_text = "Hiring: Data Scientist with Python, SQL, and ML experience."
resume_text = "Software developer with SQL experience and some machine learning coursework."

print("=== Resume Edits ===")
print(generate_resume_edits(jd_text, resume_text))

print("\n=== Interview Questions ===")
print(generate_interview_qs(jd_text))
___________________________________________________________________________________________________
import pandas as pd

# Example data — replace with your extracted PDFs/text
data = [
    {"id": 1, "type": "jd", "language": "en", "text": "Job description text..."},
    {"id": 2, "type": "resume", "language": "en", "text": "Resume text..."},
    {"id": 3, "type": "interview", "language": "en", "text": "Interview tips text..."}
]

# Convert to DataFrame
df = pd.DataFrame(data)
df.to_csv("cleaned_data.csv", index=False)
print("✅ DataFrame created and saved to cleaned_data.csv")
____________________________________________________________________________________________________
from langchain.embeddings import HuggingFaceEmbeddings

# Load a multilingual sentence-transformer
embed = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
print("✅ Multilingual embeddings ready")
__________________________________________________________________________________________________
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Convert DataFrame rows to FAISS documents
docs = [
    Document(page_content=row['text'], metadata={"type": row['type'], "language": row['language']})
    for _, row in df.iterrows()
]

# Build FAISS index
index = FAISS.from_documents(docs, embed)
index.save_local("faiss_jobmate_index")
print("✅ FAISS index saved successfully")
____________________________________________________________________________________________________
# Load FAISS index
loaded_index = FAISS.load_local("faiss_jobmate_index", embed, allow_dangerous_deserialization=True)

# Create retriever
retriever = loaded_index.as_retriever(search_kwargs={"k":3})

# Test retrieval
docs_found = retriever.get_relevant_documents("Tips for a data scientist resume")
for doc in docs_found:
    print(doc.page_content)
___________________________________________________________________________________________________

loaded_index = FAISS.load_local("faiss_jobmate_index", embed, allow_dangerous_deserialization=True)
retriever = loaded_index.as_retriever(search_kwargs={"k":3})
print(retriever.get_relevant_documents("Tips for a data scientist resume"))
___________________________________________________________________________________________________
!pip install -q streamlit pyngrok fastapi uvicorn requests PyPDF2 scikit-learn nest_asyncio
___________________________________________________________________________________________________
%%writefile mcp_server.py
import os
import time
import threading
import re
import nest_asyncio
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# -------------------------
# Gemini + LangChain + FAISS
# -------------------------
try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langdetect import detect
    import pandas as pd
    HAS_GENAI = True
except Exception as e:
    print("⚠️ LangChain/FAISS/Gemini not fully configured:", e)
    HAS_GENAI = False

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI()

class ResumeRequest(BaseModel):
    jd_text: str
    resume_text: str

class JDRequest(BaseModel):
    jd_text: str

class ChatRequest(BaseModel):
    query: str

# -------------------------
# Initialize FAISS + Chains
# -------------------------
retriever, resume_chain, interview_chain = None, None, None

if HAS_GENAI:
    try:
        # Load embeddings
        embed = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

        # Load FAISS index if exists
        index_folder = "faiss_jobmate_index_v2"
        if os.path.exists(os.path.join(index_folder, "index.faiss")):
            index = FAISS.load_local(index_folder, embeddings=embed, allow_dangerous_deserialization=True)
            retriever = index.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            print("✅ FAISS retriever loaded successfully")
        else:
            print("⚠️ FAISS index not found. Retrieval will be skipped.")

        # Gemini LLM
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.2, google_api_key=api_key)

            # Resume prompt
            resume_template = """
            You are an expert career coach. Improve the resume to match the given job description.

            IMPORTANT: Respond in the same language as {lang}.

            Job Description:
            {job}

            Resume:
            {resume}

            Knowledge Context:
            {context}

            Output: Provide 3-5 concise ATS-friendly bullet points (one per line).
            """
            resume_prompt = PromptTemplate(input_variables=["job","resume","context","lang"], template=resume_template)
            resume_chain = LLMChain(llm=llm, prompt=resume_prompt)

            # Interview prompt
            interview_template = """
            You are an interview coach. Based on the job description and the knowledge context,
            generate 5 likely interview questions and provide short model answers (2-3 sentences each).

            IMPORTANT: Respond in the same language as {lang}.

            Job Description:
            {job}

            Knowledge Context:
            {context}
            """
            interview_prompt = PromptTemplate(input_variables=["job","context","lang"], template=interview_template)
            interview_chain = LLMChain(llm=llm, prompt=interview_prompt)

            print("✅ LangChain chains ready")
        else:
            print("⚠️ GOOGLE_API_KEY not set. Gemini will not work.")
            HAS_GENAI = False

    except Exception as e:
        print("⚠️ Error initializing FAISS/LangChain:", e)
        HAS_GENAI = False

# -------------------------
# ATS Score
# -------------------------
def calculate_ats_score(jd_text: str, resume_text: str) -> float:
    """Calculate ATS match score between job description and resume."""
    jd_words = set(re.findall(r"\b\w+\b", jd_text.lower()))
    resume_words = set(re.findall(r"\b\w+\b", resume_text.lower()))

    if not jd_words:
        return 0.0
    return round((len(jd_words & resume_words) / len(jd_words)) * 100, 2)

# -------------------------
# Resume Suggestions
# -------------------------
def generate_resume_edits(jd_text: str, resume_text: str) -> str:
    if not HAS_GENAI or resume_chain is None:
        return "⚠️ Gemini/FAISS not configured."

    try:
        lang = detect(jd_text if jd_text.strip() else resume_text)
        context_docs = retriever.get_relevant_documents(jd_text) if retriever else []
        context = "\n".join([d.page_content for d in context_docs]) or "N/A"

        result = resume_chain.invoke({"job": jd_text, "resume": resume_text, "context": context, "lang": lang})
        return result["text"]

    except Exception as e:
        return f"⚠️ Error generating resume edits: {e}"

# -------------------------
# Interview Questions
# -------------------------
def generate_interview_qs(jd_text: str) -> str:
    if not HAS_GENAI or interview_chain is None:
        return "⚠️ Gemini/FAISS not configured."

    try:
        lang = detect(jd_text)
        context_docs = retriever.get_relevant_documents(jd_text) if retriever else []
        context = "\n".join([d.page_content for d in context_docs]) or "N/A"

        result = interview_chain.invoke({"job": jd_text, "context": context, "lang": lang})
        return result["text"]

    except Exception as e:
        return f"⚠️ Error generating interview questions: {e}"

# -------------------------
# Endpoints
# -------------------------
@app.post("/generate_resume_edits")
async def resume_edits_api(req: ResumeRequest):
    if not req.jd_text.strip():
        return {"error": "Job description is required."}
    if not req.resume_text.strip():
        return {"error": "Resume text is required."}

    return {
        "suggestions": generate_resume_edits(req.jd_text, req.resume_text),
        "ats_score": calculate_ats_score(req.jd_text, req.resume_text)
    }

@app.post("/generate_interview_qs")
async def interview_qs_api(req: JDRequest):
    if not req.jd_text.strip():
        return {"error": "Job description is required."}
    return {"response": generate_interview_qs(req.jd_text)}

@app.post("/gemini_chat")
async def gemini_chat_api(req: ChatRequest):
    if not HAS_GENAI:
        return {"answer": "Gemini not configured. Please set GOOGLE_API_KEY."}

    try:
        lang = detect(req.query)
        prompt = f"""
        You are a multilingual AI career assistant.
        - Only answer questions related to career guidance, resume tailoring, and interview preparation.
        - Respond in the SAME language as detected: {lang}.
        - If the query is unrelated to jobs, resumes, or interviews, politely say:
          "I'm here to help with career-related questions only."

        User query: {req.query}
        """
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return {"answer": response.text if hasattr(response, "text") else str(response)}

    except Exception as e:
        return {"answer": f"⚠️ Error calling Gemini: {e}"}

# -------------------------
# Server starter
# -------------------------
def run_uvicorn():
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    api_key = os.environ.get("AIzaSyC9iV96_5oEECqmxF3qURfApM-hHGc8L_k")
    if HAS_GENAI and api_key:
        genai.configure(api_key=api_key)

    with open("backend_url.txt", "w") as f:
        f.write("http://127.0.0.1:8000")

    print("🚀 Backend running at http://127.0.0.1:8000")

    t = threading.Thread(target=run_uvicorn, daemon=True)
    t.start()

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nServer stopped manually.")
____________________________________________________________________________________________________
%%writefile app.py
import streamlit as st
import requests
import PyPDF2
import os

# -------------------------
# Backend URL Loader
# -------------------------
def read_backend():
    try:
        with open("backend_url.txt", "r") as f:
            return f.read().strip()
    except:
        return ""

BACKEND_URL = read_backend()

# Streamlit Page Configuration
st.set_page_config(
    page_title="JobMate – AI Resume & Interview Coach",
    layout="wide",
    page_icon="💼"
)
st.title("💼 JobMate – AI Resume & Interview Coach")

# -------------------------
# Helper Functions
# -------------------------
def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF resume."""
    text = ""
    if uploaded_file is not None:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def safe_post(endpoint, payload):
    """Safely call backend APIs and handle errors."""
    if not BACKEND_URL:
        st.error("⚠️ Backend URL not found! Please start mcp_server.py.")
        return None
    try:
        url = BACKEND_URL.rstrip("/") + endpoint
        res = requests.post(url, json=payload, timeout=25)
        if res.status_code == 200:
            return res.json()
        st.error(f"❌ Backend error {res.status_code}: {res.text}")
        return None
    except Exception as e:
        st.error(f"🚨 Could not reach backend at {BACKEND_URL}\n\n{e}")
        return None

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.header("Choos Your Task")
mode = st.sidebar.radio("Choose Task", ["Resume Tailoring", "Interview Prep", "AI Career Q&A"])

# -------------------------
# Resume Tailoring
# -------------------------
if mode == "Resume Tailoring":
    st.subheader("📄 Optimize Your Resume")
    jd_text = st.text_area("Paste Job Description", height=200, placeholder="Paste the job description here...")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    resume_text = ""
    if uploaded_file:
        resume_text = extract_text_from_pdf(uploaded_file)
        st.success("✅ Resume uploaded successfully!")
        with st.expander("📂 View Extracted Resume Text"):
            st.write(resume_text)

    if st.button("✨ Generate Resume Suggestions & ATS Score"):
        if not jd_text.strip():
            st.error("⚠️ Please paste the Job Description before generating suggestions.")
        elif not resume_text.strip():
            st.error("⚠️ Please upload a resume before generating suggestions.")
        else:
            payload = {"jd_text": jd_text, "resume_text": resume_text}
            data = safe_post("/generate_resume_edits", payload)

            if data:
                if "error" in data:
                    st.error(data["error"])
                else:
                    # --- Display Suggestions in Clean Format ---
                    st.subheader("💡 Improvement Suggestions")
                    suggestions = data.get("suggestions", "")
                    if suggestions and suggestions != "✅ Resume already aligns well!":
                        st.markdown("<div style='background-color:#f9f9f9;padding:10px;border-radius:10px;'>", unsafe_allow_html=True)
                        for line in suggestions.split("\n"):
                            st.markdown(f"- {line}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.success("🎉 Resume already aligns well with the job description!")

                    # --- ATS Score Progress Bar ---
                    st.subheader("📊 ATS Score")
                    ats_percentage = float(data.get("ats_score", 0.0))  # Already in 0–100

                    # Clamp to [0.0, 1.0] for st.progress
                    progress_value = max(0.0, min(1.0, ats_percentage / 100))
                    st.progress(progress_value)
                    st.write(f"**Match Percentage:** {ats_percentage}%")

                    if ats_percentage < 50:
                        st.warning("⚠️ Low match score! Consider adding more relevant keywords and skills.")
                    elif ats_percentage < 80:
                        st.info("📈 Good match! A few improvements can make it even better.")
                    else:
                        st.success("🔥 Excellent match! Your resume is highly optimized.")

# -------------------------
# Interview Prep
# -------------------------
elif mode == "Interview Prep":
    st.subheader("🗂 Generate Interview Questions")
    jd_text = st.text_area("Paste Job Description", height=200, placeholder="Paste the job description here...")
    if st.button("🎤 Generate Interview Questions"):
        if not jd_text.strip():
            st.error("⚠️ Please paste the Job Description first.")
        else:
            payload = {"jd_text": jd_text}
            data = safe_post("/generate_interview_qs", payload)
            if data:
                if "error" in data:
                    st.error(data["error"])
                else:
                    st.subheader("Suggested Interview Questions")
                    st.markdown("<div style='background-color:#eef6ff;padding:10px;border-radius:10px;'>", unsafe_allow_html=True)
                    for q in data.get("response", "").split("\n"):
                        st.markdown(f"- {q}")
                    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# AI Career Q&A
# -------------------------
elif mode == "AI Career Q&A":
    st.subheader("🤖 AI Career Q&A")
    query = st.text_area("Ask a career-related question (e.g., interview tips, resume improvement, career growth):")

    if st.button("Get Answer"):
        if not query.strip():
            st.error("⚠️ Please type a career-related question.")
        else:
            banned_keywords = ["food", "recipe", "weather", "song", "movie", "politics"]
            if any(word in query.lower() for word in banned_keywords):
                st.error("⚠️ Please ask only career-related questions.")
            else:
                data = safe_post("/gemini_chat", {"query": query})
                if data:
                    st.subheader("AI Answer")
                    st.info(data.get("answer", "No answer returned."))
____________________________________________________________________________________________________
!fuser -k 8000/tcp || echo "No process on 8000"
!fuser -k 8501/tcp || echo "No process on 8501"
___________________________________________________________________________________________________
!nohup python mcp_server.py > mcp_server.log 2>&1 &
!sleep 5
!tail -n 20 mcp_server.log
___________________________________________________________________________________________________
!nohup streamlit run app.py --server.port 8501 --server.headless true > streamlit.log 2>&1 &
!sleep 5
!tail -n 20 streamlit.log
___________________________________________________________________________________________________
from pyngrok import ngrok

# Kill any existing ngrok tunnels (important!)
ngrok.kill()

# Set your auth token
ngrok.set_auth_token("305a1DdX0Ut3GGDLuwx43irVn3O_7Fs44mUzWzY6P1dr5aSQn")  # replace with your token

# Open a new tunnel for Streamlit (port 8501)
frontend_tunnel = ngrok.connect(8501)
print("🌍 Streamlit Public URL:", frontend_tunnel.public_url)
____________________________________________________________________________________________________
