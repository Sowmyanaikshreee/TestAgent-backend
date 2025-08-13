from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import re
import google.generativeai as genai
from utils import extract_text_from_pdf
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import os
from fastapi import FastAPI
from admin_routes import router as admin_router
from teacher_routes import router as teacher_router
from user_crud_routes import router as crud_router



app = FastAPI()

app.include_router(admin_router)
app.include_router(teacher_router)
app.include_router(crud_router)

# CORS setup
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["https://kbuddy.ai:8444"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
 
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
 
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
category_indexes = {} 
 
# Gemini API setup
genai.configure(api_key="AIzaSyDzf2NKO7x3ff28z542P_fwQvqOwgTgjB4")
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

from fastapi.staticfiles import StaticFiles
from pathlib import Path

Path("profile_photos").mkdir(exist_ok=True)
app.mount("/profile_photos", StaticFiles(directory="profile_photos"), name="profile_photos")

@app.post("/upload_profile_photo/")
async def upload_profile_photo(email: str = Form(...), photo: UploadFile = File(...)):
    try:
        safe_email = email.replace("@", "_").replace(".", "_")
        file_path = Path("profile_photos") / f"{safe_email}.jpg"
 
        with open(file_path, "wb") as f:
            f.write(await photo.read())
 
        return {"message": "Photo uploaded successfully.", "url": f"/profile_photos/{safe_email}.jpg"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Failed to upload photo: {str(e)}"})

from pathlib import Path
from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent



# Redirect to index.html
@app.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse(url="/frontend/index.html")

@app.get("/frontend/{page_name}", response_class=HTMLResponse)
async def serve_html_page(page_name: str):
    file_path = BASE_DIR.parent / "frontend" / page_name
    if file_path.exists():
        return FileResponse(file_path)
    return HTMLResponse(content="Page not found", status_code=404)


def get_or_create_index(category):
    if category not in category_indexes:
        category_indexes[category] = (faiss.IndexFlatL2(dimension), [])
    return category_indexes[category]
 
def chunk_and_index(text, category, chunk_size=500, overlap=100):
    index, chunk_store = get_or_create_index(category)
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size].strip().replace("\n", " ")
        if chunk:
            vector = embedding_model.encode([chunk])
            index.add(np.array(vector).astype("float32"))
            chunk_store.append(chunk)
 
@app.on_event("startup")
async def load_documents_on_startup():
    for root, _, files in os.walk(UPLOAD_DIR):
        for file in files:
            if file.endswith(".pdf"):
                file_path = Path(root) / file
                relative_category = Path(root).relative_to(UPLOAD_DIR).as_posix()

                try:
                    text = extract_text_from_pdf(str(file_path))
                    if text.strip():
                        chunk_and_index(text, relative_category)
                        print(f"✅ Indexed: {relative_category}/{file}")
                    else:
                        print(f"⚠️ Empty: {relative_category}/{file}")
                except Exception as e:
                    print(f"❌ Failed: {relative_category}/{file} - {e}")

 
# ✅ Upload Endpoint
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), category: str = Form(...)):
    try:
    
        category_path = UPLOAD_DIR / category
        category_path.mkdir(parents=True, exist_ok=True)

        # Save the file
        file_path = category_path / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        text = extract_text_from_pdf(str(file_path))
        if text.strip():
            chunk_and_index(text, category)
            print(f"✅ Indexed uploaded file: {category}/{file.filename}")
        else:
            print(f"⚠️ Uploaded file is empty: {category}/{file.filename}")

        return JSONResponse(content={"message": f"File uploaded to {category}."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Upload failed: {str(e)}"})

    
# ✅ Serve Uploaded Files
@app.get("/files/{category:path}/{filename}")
async def get_uploaded_file(category: str, filename: str):
    file_path = UPLOAD_DIR / category / filename
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"message": "File not found."})
    return FileResponse(file_path)
 
@app.post("/ask/")
async def ask_question(question: str = Form(...), category: str = Form(...)):
    if category not in category_indexes:
        return {"answer": "No documents indexed for this category. Please upload a PDF."}
 
    index, chunk_store = category_indexes[category]
    question_vec = embedding_model.encode([question]).astype("float32")
    D, I = index.search(question_vec, 5)
    relevant_chunks = [chunk_store[i] for i in I[0] if i < len(chunk_store)]
    context = "\n\n".join(relevant_chunks)
 
    prompt = f"""
You are a helpful assistant. Use only the following document content to answer the question below.
 
Answer clearly and cleanly using:
- Bullet points with "•"
- No Markdown (no *, **, or symbols like ` or >)
- No formatting characters, just plain readable text
 
### DOCUMENT START
{context}
### DOCUMENT END
 
Question: {question}
Answer:
"""
 
    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
 
        cleaned = re.sub(r"\*+", "", raw)
        cleaned = re.sub(r"`+", "", cleaned)
        cleaned = re.sub(r"_{2,}", "", cleaned)
        cleaned = re.sub(r"•\s*•", "•", cleaned)
        cleaned = cleaned.replace("- ", "• ")  # make all bullets consistent
        cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned)  # spacing between paragraphs
        cleaned = re.sub(r"^\s*•", "•", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*:\s*", ": ", cleaned)
        cleaned = re.sub(r"\s*•\s*", "\n• ", cleaned)  # force each bullet to new line
        cleaned = re.sub(r"(Answer:)", r"\1\n\n", cleaned)
 
        cleaned = cleaned.strip()
 
        return {"answer": cleaned}
    except Exception as e:
        return {"answer": f"Error calling Gemini: {e}"}

import random
user_answers = []       

# Store generated questions and their corresponding text
question_sources = {}
 
@app.post("/generate_question/")
async def generate_question(category: str = Form(...)):
    if category not in category_indexes:
        return {"question": "No documents indexed for this category. Please upload a PDF."}
    
    _, chunk_store = category_indexes[category]
    
    if not chunk_store:
        return {"question": "No content available for this category."}
    
    # Randomly pick chunks
    sampled_chunks = random.sample(chunk_store, min(3, len(chunk_store)))
    combined_text = "\n\n".join(sampled_chunks)
    combined_text = combined_text[-3000:]  # safety cutoff

    prompt = f"""
You are an educational AI tutor.
Generate **one** standalone question that tests understanding of the following material.
Do NOT start with 'Based on the document...' or similar.
Ensure that the answer to this question is clearly available within this content.
Do not ask multiple choice questions
=== CONTENT START ===
{combined_text}
=== CONTENT END ===
Question:"""
    
    try:
        response = model.generate_content(prompt)
        question_text = response.text.strip()
        question_text = re.sub(r'[*_`#~]', '', question_text)

        # Save the source text for this question
        question_sources[question_text] = combined_text

        return {"question": question_text}
    except Exception as e:
        return {"question": f"Error generating question: {e}"}


@app.post("/submit_answer/")
async def submit_answer(question: str = Form(...), answer: str = Form(...)):
    user_answers.append({"question": question, "answer": answer})
    return {"status": "saved"}


@app.post("/evaluate_answer/")
async def evaluate_answer(question: str = Form(...), answer: str = Form(...), category: str = Form(...)):
    # Use the exact text that was used to generate this question
    source_text = question_sources.get(question)
    
    if not source_text:
        return {"evaluation": "Error: No source text found for this question. Please generate a new question."}

    prompt = f"""
You are a teacher assistant. Evaluate the user's answer to the question below, using the provided content as the source of truth.
=== STUDY MATERIAL ===
{source_text}
=== END ===
Question: {question}
User's Answer: {answer}
Provide short, crisp, and clear feedback in one or two sentences only.
Do not mention or refer to the text or material explicitly in your feedback. Give feedback naturally and directly.
Feedback:"""
    
    try:
        response = model.generate_content(prompt)
        feedback = response.text.strip()
        # Clean special formatting
        cleaned_feedback = re.sub(r'[*_`#~]', '', feedback)
        return {"evaluation": cleaned_feedback}
    except Exception as e:
        return {"evaluation": f"Error evaluating: {e}"}
    

 
# ✅ List Uploaded Files (Group by class/subject)
@app.get("/uploaded_files/")
async def list_uploaded_files():
    files_by_category = {}
    for root, dirs, files in os.walk("uploads"):
        rel_path = os.path.relpath(root, "uploads")
        if rel_path == ".":
            continue
        key = rel_path.replace("\\", "/")
        files_by_category[key] = files
    return {"files_by_category": files_by_category}



from fastapi.responses import StreamingResponse
from fpdf import FPDF
import io
from collections import defaultdict
from fpdf import FPDF


# --- Utility Functions ---
def clean_extracted_text(text: str) -> str:
    text = re.sub(r"/[A-Z]+\d+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text

def sanitize_text(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")

# --- Main Endpoint ---
from fastapi.responses import JSONResponse
import base64

@app.post("/generate_pdf")
async def generate_pdf(data: dict):
    import difflib
    from collections import defaultdict
    import io
    from fpdf import FPDF

    def is_similar(q1, q2, threshold=0.85):
        ratio = difflib.SequenceMatcher(None, q1, q2).ratio()
        return ratio >= threshold

    category = data["category"]
    subject = category.replace("_", " ").title()
    main_title = f"Question Paper - {subject}"
    heading_name_line = "Name: ____________________         Date: ____________________"

    file_name = data["file"]
    level = data.get("level", "medium")

    counts = {
        "mcq": int(data.get("mcq", 0)),
        "fill": int(data.get("fill", 0)),
        "short": int(data.get("short", 0)),
        "long": int(data.get("long", 0)),
    }

    file_path = UPLOAD_DIR / category / file_name
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"error": "File not found."})

    raw_text = extract_text_from_pdf(str(file_path))
    if not raw_text.strip():
        return JSONResponse(status_code=400, content={"error": "No extractable content in PDF."})

    def clean_extracted_text(text: str) -> str:
        text = re.sub(r"/[A-Z]+\d+", "", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text

    def extract_main_content(text: str) -> str:
        lower_text = text.lower()
        start_keywords = ["chapter 1", "lesson 1", "unit 1"]
        start_index = -1
        for keyword in start_keywords:
            idx = lower_text.find(keyword)
            if idx != -1:
                if start_index == -1 or idx < start_index:
                    start_index = idx
        if start_index != -1:
            return text[start_index:]
        lines = text.split('\n')
        skip_phrases = ["acknowledgment", "copyright", "preface", "author",
                        "published", "isbn", "editor", "introduction", "dedication"]
        content_lines = [line.strip() for line in lines
                         if len(line.strip()) > 20 and not any(skip in line.lower() for skip in skip_phrases)]
        return "\n".join(content_lines)

    def split_text_into_chunks(text, chunk_size=3000, overlap=500):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def sanitize_text(text: str) -> str:
        return text.encode("latin-1", "replace").decode("latin-1")

    text = clean_extracted_text(raw_text)
    main_content = extract_main_content(text)
    chunks = split_text_into_chunks(main_content)

    mark_weights = {"MCQ": 1, "FILL": 1, "SHORT": 2, "LONG": 5}
    section_titles = {
        "MCQ": "Multiple Choice Questions",
        "FILL": "Fill in the Blanks",
        "SHORT": "Answer the following in 20 to 30 words:",
        "LONG": "Answer the following in 50 to 100 words:"
    }

    sectioned_questions = defaultdict(list)
    generated_questions = set()

    for qtype, count in counts.items():
        for i in range(count):
            chunk = random.choice(chunks)
            q_upper = qtype.upper()

            if qtype == "mcq":
                prompt = f"""You are an exam question generator.
Generate a MULTIPLE-CHOICE question at {level.upper()} difficulty.
Format:
<Question>
A. Option A
B. Option B
C. Option C
D. Option D

CONTENT:
{chunk}
Only return the question and options."""
            elif qtype == "fill":
                prompt = f"""Generate a FILL IN THE BLANKS question #{i+1} at {level.upper()} difficulty.
Insert one blank using '__________' in place of a key term.
CONTENT:
{chunk}
Only return the question."""
            elif qtype == "short":
                prompt = f"""Generate a SHORT ANSWER QUESTION #{i+1} at {level.upper()} difficulty.
CONTENT:
{chunk}
Only return the question."""
            elif qtype == "long":
                prompt = f"""Generate a LONG ANSWER QUESTION #{i+1} at {level.upper()} difficulty.
CONTENT:
{chunk}
Only return the question."""
            else:
                prompt = f"Generate a question from this:\n{chunk}"

            try:
                response = model.generate_content(prompt)
                question = response.text.strip()
                if q_upper == "FILL" and "____" not in question:
                    question += " __________"

                # Deduplication check
                if any(is_similar(question, prev_q) for prev_q in generated_questions):
                    continue
                generated_questions.add(question)
                sectioned_questions[q_upper].append(question)

            except Exception as e:
                sectioned_questions[q_upper].append(f"[Error]: {e}")

    # Prepare Text Format
    text_lines = []
    q_num = 1
    for qtype in ["MCQ", "FILL", "SHORT", "LONG"]:
        questions = sectioned_questions[qtype]
        if not questions:
            continue
        marks = mark_weights[qtype]
        title = f"{section_titles[qtype]} ({len(questions)} × {marks} = {len(questions)*marks})"
        text_lines.append(title)
        text_lines.append("")
        for q in questions:
            text_lines.append(f"{q_num}. {q}\n")
            if qtype == "SHORT":
                text_lines.append("-" * 50 + "\n")
            elif qtype == "LONG":
                text_lines.append("-" * 80 + "\n")
            q_num += 1
        text_lines.append("")

    final_text = "\n".join(text_lines)

    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, main_title, ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, heading_name_line, ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    question_number = 1

    for qtype in ["MCQ", "FILL", "SHORT", "LONG"]:
        questions = sectioned_questions.get(qtype, [])
        if not questions:
            continue
        marks_per_q = mark_weights[qtype]
        total_marks = len(questions) * marks_per_q
        section_title = f"{section_titles[qtype]} ({len(questions)} × {marks_per_q} = {total_marks})"
        pdf.set_font("Arial", "B", 12)
        pdf.multi_cell(0, 10, section_title)
        pdf.ln(2)

        pdf.set_font("Arial", "", 12)
        for q in questions:
            safe_text = sanitize_text(f"{question_number}. {q}")
            pdf.multi_cell(0, 10, safe_text)
            pdf.ln(2)
            if qtype == "SHORT":
                pdf.ln(20)
            elif qtype == "LONG":
                pdf.ln(40)
            question_number += 1
        pdf.ln(4)

    pdf_output = pdf.output(dest="S").encode("latin-1", "replace")
    pdf_base64 = base64.b64encode(pdf_output).decode()

    return JSONResponse(content={
        "pdf_data": pdf_base64,
        "text": final_text
    })





from fastapi import Query
 
# ✅ Delete Uploaded File
@app.delete("/delete_file")
async def delete_file(filename: str, category: str):
    file_path = UPLOAD_DIR / category / filename
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"message": "File not found."})
    try:
        file_path.unlink()
        return JSONResponse(content={"message": f"{filename} deleted from {category}."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error deleting file: {str(e)}"})
 
# Lesson Plan Builder
from fastapi import Request
from fastapi.responses import StreamingResponse
from fpdf import FPDF
import io
 
@app.post("/generate_lesson_plan")
async def generate_lesson_plan(data: dict):
    subject = data.get("subject", "")
    grade = data.get("grade", "")
    standard = data.get("standard", "")
    objective = data.get("objective", "")
    activities = data.get("activities", "")
    assessment = data.get("assessment", "")
    template = data.get("template", "detailed")
 
    prompt = f"""
You are an educational expert. Create a {template} lesson plan using the following information.
 
Subject: {subject}
Grade Level: {grade}
Curriculum Standard: {standard}
Learning Objective: {objective}
Teaching Activities: {activities}
Assessment Methods: {assessment}
 
The lesson plan should include labeled sections and use clear formatting.
"""
 
    try:
        response = model.generate_content(prompt)
        return {"plan": response.text.strip()}
    except Exception as e:
        return {"plan": f"Error: {e}"}
 
 
@app.post("/generate_lesson_plan_pdf")
async def generate_lesson_plan_pdf(request: Request):
    try:
        data = await request.json()
        plan_text = data.get("plan", "").strip()
 
        if not plan_text:
            return JSONResponse(status_code=400, content={"error": "No content to generate."})
 
        class LessonPlanPDF(FPDF):
            def header(self):
                self.set_font("Arial", "B", 14)
                self.cell(0, 10, "Lesson Plan", ln=True, align="C")
                self.ln(5)
 
            def chapter_title(self, title):
                clean_title = re.sub(r"[*\-•]+", "", title).strip()
                self.set_font("Arial", "B", 12)
                self.multi_cell(0, 10, clean_title)
                self.ln(2)

            def chapter_body(self, body):
                clean_body = re.sub(r"[*\-•]+", "", body).strip()
                safe_text = clean_body.encode("latin-1", "replace").decode("latin-1")
                self.set_font("Arial", "", 11)
                self.multi_cell(0, 8, safe_text)
                self.ln(3)

        pdf = LessonPlanPDF()
        pdf.add_page()

         # Clean up markdown formatting from the plan text
        plan_text = re.sub(r"\*\*(.*?)\*\*", r"\1", plan_text)  # remove bold markers
        plan_text = re.sub(r"\*(.*?)\*", r"\1", plan_text)      # remove italics
        plan_text = re.sub(r"__([^_]+)__", r"\1", plan_text)     # remove __ underline
        plan_text = re.sub(r"`([^`]+)`", r"\1", plan_text)       # remove code backticks
 
        for section in plan_text.split("\n\n"):
            if ":" in section:
                title, content = section.split(":", 1)
                pdf.chapter_title(title)
                pdf.chapter_body(content)
            else:
                pdf.chapter_body(section)
 
        # ✅ Fix: use `dest='S'` to get PDF output as bytes
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        buffer = io.BytesIO(pdf_bytes)
 
        return StreamingResponse(buffer, media_type="application/pdf", headers={
            "Content-Disposition": "attachment; filename=lesson_plan.pdf"
        })
 
    except Exception as e:
        print("❌ PDF generation error:", e)
        return JSONResponse(status_code=500, content={"error": f"PDF generation failed: {str(e)}"})

import requests
from fastapi import Query
 
# 🔐 Set your YouTube API key here
YOUTUBE_API_KEY = "AIzaSyBShu90YRJ-VT7ox_LyXyO19EqpGdkgHcY"
 
@app.get("/real_youtube_links/")
def get_real_youtube_links(q: str = Query(...), max_results: int = 7):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": q,
        "key": YOUTUBE_API_KEY,
        "maxResults": max_results,
        "type": "video",
        "safeSearch": "strict"
    }
 
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("items", [])
 
        links = [
            {
                "title": item["snippet"]["title"],
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            }
            for item in results
        ]
        return {"results": links}
    except Exception as e:
        return {"results": [], "error": str(e)}
 
from youtube_transcript_api import YouTubeTranscriptApi
import re
 
def extract_video_id(url):
    """
    Extracts the YouTube video ID from a standard YouTube URL.
    """
    match = re.search(r"(?:v=|youtu\.be/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None
 
@app.post("/summarize_youtube")
async def summarize_youtube(data: dict):
    link = data.get("link", "")
    video_id = extract_video_id(link)
 
    if not video_id:
        return {"summary": "❌ Invalid YouTube link format."}
 
    try:
        # ✅ Attempt to get transcript
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([item["text"] for item in transcript_data])[:3500]  # limit for Gemini
 
        # ✅ Summarization prompt
        prompt = f"""
You are an AI tutor. Summarize the key educational points from the transcript below.
 
=== BEGIN TRANSCRIPT ===
{transcript_text}
=== END TRANSCRIPT ===
 
Provide a clear, student-friendly summary (1-2 paragraphs):
"""
 
        response = model.generate_content(prompt)
        return {"summary": response.text.strip()}
 
    except Exception as e:
        return {"summary": f"⚠️ Unable to summarize video. Error: {str(e)}"}
 
 
@app.post("/summarize_text")
async def summarize_text(data: dict):
    input_text = data.get("text", "").strip()
    if not input_text:
        return {"summary": "⚠️ No input text provided."}
 
    prompt = f"""
You are a helpful assistant. Please summarize the following content in a simple, clear, and student-friendly way.
 
=== INPUT ===
{input_text}
=== END ===
 
Summary:
"""
    try:
        response = model.generate_content(prompt)
        return {"summary": response.text.strip()}
    except Exception as e:
        return {"summary": f"❌ Error during summarization: {str(e)}"}
 
 
from fastapi import UploadFile, File, Form
from PIL import Image
import io
 
@app.post("/analyze_image")
async def analyze_image(image: UploadFile = File(...), prompt: str = Form(...)):
    try:
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data))
 
        # Combine user prompt with image
        full_prompt = prompt if prompt.strip() else "Describe this image in simple terms for a student."
 
        response = model.generate_content([full_prompt, image_pil])
        return {"description": response.text.strip()}
 
    except Exception as e:
        return {"description": f"❌ Error analyzing image: {str(e)}"}
 
@app.post("/chat/")
async def general_chat(question: str = Form(...)):
    prompt = f"You are a helpful, friendly AI tutor. Answer the following question in a clear, student-friendly way:\n\n{question}\n\nAnswer:"

    try:
        response = model.generate_content(prompt)
        return {"answer": response.text.strip()}
    except Exception as e:
        return {"answer": f"Error: {e}"}





 # Assessment Evaluataion

from google.cloud import vision
from google.oauth2 import service_account
import tempfile
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

 
@app.post("/upload_student_assessment/")
async def upload_student_assessment(file: UploadFile = File(...), category: str = Form(...)):
    try:
        # ✅ Get Vision API key from environment
        VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")
        if not VISION_API_KEY:
            raise ValueError("Please set GOOGLE_VISION_API_KEY in environment.")

        # ✅ Save uploaded file
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        extracted_text = ""

        # ✅ Use Google Vision API with API key
        def extract_text_from_image(image_bytes):
            img_b64 = base64.b64encode(image_bytes).decode("utf-8")
            url = f"https://vision.googleapis.com/v1/images:annotate?key={VISION_API_KEY}"
            payload = {
                "requests": [
                    {
                        "image": {"content": img_b64},
                        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]
                    }
                ]
            }
            r = requests.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return data["responses"][0].get("fullTextAnnotation", {}).get("text", "")

        # ✅ Extract image(s) from PDF or use directly
        if file.filename.lower().endswith(".pdf"):
            doc = fitz.open(file_path)
            for page in doc:
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                extracted_text += extract_text_from_image(img_bytes)
        else:
            with open(file_path, "rb") as img_file:
                extracted_text += extract_text_from_image(img_file.read())

        if not extracted_text.strip():
            return {"error": "No readable text found in the uploaded file."}

        if category not in category_indexes:
            return {"error": "Category not found. Upload a document first."}

        _, chunk_store = category_indexes[category]
        material = "\n".join(chunk_store)[-3000:]

        # ✅ Prompt for Gemini
        prompt = f"""
You are a teacher assistant.

Evaluate the student's answers below using the study material provided. Return detailed output in this exact structure for each question:

---
Question: <insert question text>
Student Answer: <insert answer>
Marks: x/y
Feedback: <insert feedback>
---

Use only the material provided. Be objective and concise.

=== STUDENT ANSWERS ===
{extracted_text}
=== STUDY MATERIAL ===
{material}
"""
        response = model.generate_content(prompt)
        feedback = response.text.strip()

        # ✅ Compute total marks
        marks = re.findall(r"Marks:\s*(\d+)\s*/\s*(\d+)", feedback)
        total = sum(int(m[0]) for m in marks)
        out_of = sum(int(m[1]) for m in marks)
        summary_line = f"Total Marks: {total}/{out_of}\n\n"

        # ✅ Generate PDF feedback
        class FeedbackPDF(FPDF):
            def header(self):
                self.set_font("Arial", "B", 14)
                self.cell(0, 10, "Assessment Feedback", ln=True, align="C")
                self.ln(5)

            def add_feedback(self, text):
                safe_text = text.encode("latin-1", "replace").decode("latin-1")
                self.set_font("Arial", "", 12)
                self.multi_cell(0, 10, safe_text)
                self.ln(5)

        pdf = FeedbackPDF()
        pdf.add_page()
        pdf.add_feedback(summary_line + feedback)

        pdf_bytes = pdf.output(dest="S").encode("latin1")
        return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf", headers={
            "Content-Disposition": "attachment; filename=feedback.pdf"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
 
 # This is crucial for Cloud Run
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

 