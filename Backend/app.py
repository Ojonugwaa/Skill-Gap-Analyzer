# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from routes import skill_routes, course_routes  

# app = FastAPI()

# # ✅ Allow frontend access (adjust origins if needed)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],  # Replace "*" with your frontend URL in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Include routers
# app.include_router(skill_routes.router)
# app.include_router(course_routes.router)  # if you have curriculum grouping

# @app.get("/")
# def root():
#     return {"message": "Welcome to the Smart Career Recommender API"}





from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import PyPDF2
import docx
import pandas as pd
import logging
import asyncio
import time
from typing import Optional
from pdf2image import convert_from_bytes
import pytesseract

from Backend.services.gpt4all_job_skill_extractor import extract_job_title_and_skills_with_llama
from Backend.services.missing_skill_finder import identify_missing_skills
from Backend.services.course_recommender import recommend_courses
from Backend.routes.course_routes import router as course_router

import gdown
import zipfile
import os

# Replace these with your actual file IDs or public shareable URLs
# Example Google Drive file links or IDs
phi3_file_id = '1RSphUsdluwU8xNmCeiA_qlsnKxoaG_c2'
ner_model_zip_id = '1nsnXJpb2YGiNRMyK7vf_fMq_6cP8RRRI'

# Output paths
output_dir = 'models'
os.makedirs(output_dir, exist_ok=True)

# Download Phi-3 model
phi3_output = os.path.join(output_dir, 'Phi-3-mini-4k-instruct.Q4_0.gguf')
gdown.download(f'https://drive.google.com/uc?id={phi3_file_id}', phi3_output, quiet=False)

# Download ner_model.zip (if zipped)
ner_model_output = os.path.join(output_dir, 'ner_model.zip')
gdown.download(f'https://drive.google.com/uc?id={ner_model_zip_id}', ner_model_output, quiet=False)

# Extract ner_model.zip
with zipfile.ZipFile(ner_model_output, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(output_dir, 'ner_model'))

print('Models downloaded and extracted successfully!')


# Configure logging
logging.basicConfig(
    filename='fastapi_log.txt',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(course_router)
# Load dataset
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data/IT jobs for training.csv")

try:
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Dataset loaded successfully: {len(df)} entries")
except FileNotFoundError:
    logger.error(f"Dataset not found at: {DATA_PATH}")
    raise HTTPException(status_code=500, detail=f"Dataset not found at: {DATA_PATH}")


def extract_text_from_pdf(file):
    start_time = time.time()
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages[:5]:
            extracted = page.extract_text()
            if extracted:
                text += extracted

        if text.strip():
            logger.info(f"PDF text extraction took {time.time() - start_time:.2f} seconds")
            return text.strip()

        # Fallback to OCR if no embedded text
        logger.warning("No embedded text found, attempting OCR")
        file.seek(0)
        images = convert_from_bytes(file.read(), first_page=1, last_page=5)
        ocr_text = ""
        for img in images:
            ocr_text += pytesseract.image_to_string(img)
        if not ocr_text.strip():
            raise HTTPException(status_code=400, detail="Failed to read PDF")
        logger.info(f"OCR extraction took {time.time() - start_time:.2f} seconds")
        return ocr_text.strip()

    except Exception as e:
        logger.error(f"Failed to extract PDF text: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")


def extract_text_from_docx(file):
    start_time = time.time()
    try:
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        if not text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in DOCX")
        logger.info(f"DOCX extraction took {time.time() - start_time:.2f} seconds")
        return text.strip()
    except Exception as e:
        logger.error(f"Failed to extract DOCX text: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract DOCX text: {str(e)}")


async def extract_with_timeout(cv_text: str, timeout_seconds: int = 1000):
    try:
        logger.debug("Starting LLM extraction with timeout")
        result = await asyncio.wait_for(
            asyncio.to_thread(extract_job_title_and_skills_with_llama, cv_text[:3000]),
            timeout=timeout_seconds
        )
        return result
    except asyncio.TimeoutError:
        logger.error("GPT4All extraction timed out")
        raise HTTPException(status_code=504, detail="Skill extraction timed out")
    except Exception as e:
        logger.error(f"GPT4All extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Skill extraction failed: {str(e)}")


@app.post("/analyze_skills")
async def analyze_skills(
    job_title: Optional[str] = Form(None),
    cv_skills: Optional[str] = Form(None),
    cv_file: Optional[UploadFile] = File(None)
):
    start_time = time.time()
    logger.debug("Received /analyze_skills request")

    try:
        response = {
            "job_title": job_title or "Not detected",
            "extracted_skills": [],
            "missing_skills": [],
            "course_recommendations": {}
        }

        cv_text = ""
        if cv_file:
            logger.debug(f"Processing uploaded file: {cv_file.filename}")
            if cv_file.content_type == "application/pdf":
                cv_text = extract_text_from_pdf(cv_file.file)
            elif cv_file.content_type in [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword"
            ]:
                cv_text = extract_text_from_docx(cv_file.file)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")

            extraction_result = await extract_with_timeout(cv_text)
            # extraction_result = {
            #     "job_title": "Software Developer",
            #     "skills": ["Python", "APIs", "SQL"]
            # }

            response["job_title"] = extraction_result["job_title"]
            response["extracted_skills"] = extraction_result["skills"]

        if not cv_file and cv_skills:
            response["extracted_skills"] = [skill.strip() for skill in cv_skills.split(",") if skill.strip()]
            logger.info(f"Manual skills provided: {response['extracted_skills']}")

        logger.debug(f"Looking up job_title: {response['job_title']}")
        matched = df[df["job_title"].str.lower() == response["job_title"].lower()]
        if matched.empty:
            logger.warning(f"No dataset match for job_title: {response['job_title']}")
        else:
            job_skills = matched.iloc[0]["skills"]
            response["missing_skills"] = identify_missing_skills(response["extracted_skills"], job_skills)
            response["course_recommendations"] = recommend_courses(response["missing_skills"])

        logger.info(f"Request completed in {time.time() - start_time:.2f} seconds")
        return JSONResponse(content=response)

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}", exc_info=True)  # log full traceback
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    
@app.get("/")
async def health_check():
    return {"status": "FastAPI is running"}
