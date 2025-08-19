from flask import Flask, request, jsonify
from flask_cors import CORS  
import fitz
from PIL import Image
import traceback
import base64
import io
import json
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import tempfile
import google.generativeai as genai
from io import BytesIO
from pypdf import PdfReader
import uuid
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
from io import BytesIO
from zoneinfo import ZoneInfo

app = Flask(__name__)
CORS(app)
load_dotenv()

# PostgreSQL Connection String
pg_connection_string = f"""
    host={os.getenv("PG_HOST")}
    port={os.getenv("PG_PORT")}
    dbname={os.getenv("PG_NAME")}
    user={os.getenv("PG_USER")}
    password={os.getenv("PG_PASSWORD")}
"""

# OpenAI & Gemini Setup
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# PostgreSQL Connection
# conn = psycopg2.connect(pg_connection_string)
# cursor = conn.cursor(cursor_factory=RealDictCursor)

# Gemini OCR
def process_pdf_with_gemini_ocr(pdf_path, dpi=500):
    model = genai.GenerativeModel('gemini-2.0-flash')
    doc = fitz.open(pdf_path)
    images = []

    for page in doc:
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        img_bytes = BytesIO(pix.tobytes("png"))
        img = Image.open(img_bytes)
        images.append(img)

    doc.close()

    prompt = f"""
    you are an ocr tools that extract resume. dont add anything and only return the resume   
    """

    response = model.generate_content([prompt, *images])
    return response.text, response.usage_metadata.total_token_count

# Evaluation
def evaluate_resume(pdf_path, original_filename, job_desc, user_id, url, acceptance = 70):
    current_uuid = str(uuid.uuid4())
    # Open the PDF
    reader = PdfReader(pdf_path)
    
    accpetanceVal = acceptance

    gemini_token = 0

    # Collect text from all pages
    has_text = False
    all_text = ''

    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():  # Check if text exists and is non-empty
            has_text = True
            all_text += text

    if not has_text:
        all_text, gemini_token = process_pdf_with_gemini_ocr(pdf_path)

    resume = all_text

    prompt = f"""
    You are a resume evaluator. You are given the following inputs:
    Job Description: {job_desc}
    Resume: {resume}
    Analyze the resume against the job description and return your evaluation in raw JSON format only (without any markdown formatting or labels).
    Return the following fields in the JSON:
    name: Full name of the candidate
    title: the job title
    job_desription: the job description
    email: Email address
    phone_number: Phone number
    percentage_match: An integer percentage (0–100) representing how well the resume matches the job description
    short_description: A 1–2 sentence summary of the candidate relevant to the job and you must include whether 
    <b>Ai Suggestion : <span class='text-success'>Can be hired </span> </b> if percentage_match above {accpetanceVal} or
    <b>Ai Suggestion : <span class='text-danger'>Not recommended to be hired</span></b> below
    do <br> before the ai suggestion html script
    """

    # Second response: Evaluation
    response2 = client.responses.create(
        model="gpt-5-nano",
        input=prompt
    )

    if not response2.output_text:
        raise ValueError("Evaluation response is empty or invalid.")

    try:
        data = json.loads(response2.output_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {response2.output_text}") from e

    data["LOG_HISTORY_ID"] = current_uuid
    data["user_id"] = user_id
    data["file_url"] = url
    data["date"] = datetime.now(ZoneInfo("Asia/Kuala_Lumpur"))
    data["total_token_gemini"] = gemini_token
    data["total_token_openai"] = response2.usage.total_tokens
    data["pdf_name"] = original_filename
    data["match_acceptence"] = accpetanceVal 


    values = (
    data["LOG_HISTORY_ID"],
    data["user_id"],
    data["date"],
    data["title"],
    data["job_desription"],
    data["file_url"],
    data["name"],
    data["email"],
    data["phone_number"],
    data["percentage_match"],
    data["short_description"],
    0,
    data["total_token_openai"],
    data["total_token_gemini"],
    data["match_acceptence"]
    )

    insert_query = """
    INSERT INTO LOG_HISTORY (
        "LOG_HISTORY_ID",
        "user_id",
        "date_run",
        "title",
        "job_description",
        "file_url",
        "name",
        "email",
        "phone_no",
        "match_percentage",
        "short_desc",
        "is_shortlisted",
        "gpt_token",
        "gemini_token",
        "match_acceptance"
    )
     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    with psycopg2.connect(pg_connection_string) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(insert_query, values)
            conn.commit()


    return data

@app.route('/evaluate-resume', methods=['POST'])
def upload_resume():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    job_desc = request.form.get('job_desc')
    user_id = request.form.get('user_id')
    acceptance = request.form.get('acceptance')

    if not job_desc:
        return jsonify({'error': 'No job description provided'}), 400

    try:
        # ✅ Read file content into memory
        file_bytes = file.read()

        # 🟢 Upload to remote API
        upload_response = requests.post(
            "http://webapifileupload.aiscreenmax.my/api/ResumeUpload/UploadFile",
            files={'file_url': (file.filename, BytesIO(file_bytes))}
        )

        # Extract URL string from upload_response if needed (you currently assign the full response object to `url`)
        url = upload_response.text  # or `.json().get('url')` depending on API

        # 🟢 Save file locally from memory
        original_filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            result = evaluate_resume(tmp.name, original_filename, job_desc, user_id=user_id, url=url, acceptance=acceptance)

        return jsonify(result)

    except Exception as e:
        print(f"❌ Backend error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000, debug=True)