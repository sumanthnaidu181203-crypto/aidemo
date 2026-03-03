from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import torch
import os
import shutil
from paddleocr import PaddleOCR
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import faiss
import re
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import io

# Initialize models
ocr = PaddleOCR(use_angle_cls=True, lang='en', rec=False, det_db_box_thresh=0.5, det_db_unclip_ratio=2)
flan_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
flan_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="cpu")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# FastAPI app
app = FastAPI()

# In-memory store for current image processing session
session_data = {"text": "", "chunks": [], "index": None}

# Utility functions
def extract_text_from_image(image_path):
    result = ocr.ocr(image_path, cls=True)
    text = [line[1][0] for line in result[0] if line[1][1] > 0.95]
    return ' '.join(text)

def preprocess_text(text):
    return re.sub(r"\n{2,}", "\n", text).strip()

def chunk_text(text, chunk_size=512, overlap=100):
    tokens = flan_tokenizer(text, return_tensors="pt")["input_ids"][0]
    return [
        flan_tokenizer.decode(tokens[i:i+chunk_size], skip_special_tokens=True)
        for i in range(0, len(tokens), chunk_size - overlap)
    ]

def build_faiss_index(text_chunks):
    embeddings = embed_model.encode(text_chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, text_chunks

def retrieve_relevant_chunks(query, index, text_chunks, top_k=1):
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, top_k)
    return [text_chunks[idx] for idx in indices[0] if 0 <= idx < len(text_chunks)]

def extract_answer(context, question):
    inputs = flan_tokenizer(f"context: {context} question: {question}", return_tensors="pt", truncation=True, max_length=1024)
    inputs = inputs.to(flan_model.device)
    outputs = flan_model.generate(**inputs, max_new_tokens=200)
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Routes
@app.post("/upload-Document/")
async def upload_document(file: UploadFile = File(...)):
    pdf_path = f"temp_{file.filename}"
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            image_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_np = np.array(image)

            page_text = extract_text_from_image(image_np)
            text += page_text + "\n"
    text = preprocess_text(text)
    chunks = chunk_text(text)
    index, chunks = build_faiss_index(chunks)

    session_data["text"] = text
    session_data["chunks"] = chunks
    session_data["index"] = index

    os.remove(pdf_path)
    return {
        "message": "PDF processed",
        "text_length": len(text),
        "ocr_text": text
    }

@app.post("/upload-Image/")
async def upload_image(file: UploadFile = File(...)):
    image_path = f"temp_{file.filename}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text_from_image(image_path)
    text = preprocess_text(text)
    chunks = chunk_text(text)
    index, chunks = build_faiss_index(chunks)

    session_data["text"] = text
    session_data["chunks"] = chunks
    session_data["index"] = index

    os.remove(image_path)
    return {
        "message": "Image processed",
        "text_length": len(text),
        "ocr_text": text
    }

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    if not session_data["text"]:
        return JSONResponse(status_code=400, content={"error": "No image uploaded or OCR failed."})

    relevant_chunks = retrieve_relevant_chunks(question, session_data["index"], session_data["chunks"])
    context = "\n".join(relevant_chunks)
    answer = extract_answer(context, question)
    return {"question": question, "answer": answer}



