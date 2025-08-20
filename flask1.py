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

    no_description = job_desc == "!##NO DESCRIPTION##!"

    if no_description:
        prompt = f"""
    You are a resume evaluator. You are given the following inputs:
    Job Description: {job_desc}
    Resume: {resume}
    Analyze the resume and return in raw JSON format only (without any markdown formatting or labels).
    Return the following fields in the JSON:
    name: Full name of the candidate
    title: current job title
    job_desription: the job description
    email: Email address
    company: current company
    past_company: list of all past companies as a valid JSON array of strings (e.g. ["Company A", "Company B"]) without the key (i.e. 0, 1, 2)  
    description: list of all past description of the candidate work as a valid JSON array of strings (e.g. ["Company A", "Company B"]) without the key (i.e. 0, 1, 2)  
    past_title: job title held before as a valid JSON array of strings (e.g. ["Company A", "Company B"]) without the key (i.e. 0, 1, 2)  
    current_description: current company description
    current_comp_year: current company start year must be in integer
    current_comp_month: current company start month must be in integer in range of 1 to 12 if not just do null
    start_year: list of all start year of all past company and title
    start_month: list of all start month of all past company and title must be in integer in range of 1 to 12 if not just do null
    end_year: list of all end year of all past company and title
    end_month: list of all end year of all past company and title must be in integer in range of 1 to 12 if not just do null
    employment_type: permanent or contract
    location: current company location
    phone_number: Phone number
    skill: list down 5 skill that the candidate have
    proficiency: proficiency of the 5 skill you listed
    years_experience: years experience for the 5 skill you listed
    last_used_year: last used year for 5 skill u listed
    percentage_match: put 0 only
    short_description: A 1–2 sentence summary of the about candidate
    """
    else:
        prompt = f"""
    You are a resume evaluator. You are given the following inputs:
    Job Description: {job_desc}
    Resume: {resume}
    Analyze the resume against the job description and return your evaluation in raw JSON format only (without any markdown formatting or labels).
    Return the following fields in the JSON:
    name: Full name of the candidate
    title: current job title
    job_desription: the job description
    email: Email address
    company: current company
    past_company: list of all past companies as a valid JSON array of strings (e.g. ["Company A", "Company B"]) without the key (i.e. 0, 1, 2)
    description: list of all past description of the candidate work as a valid JSON array of strings (e.g. ["Company A", "Company B"]) without the key (i.e. 0, 1, 2)
    past_title: job title held before as a valid JSON array of strings (e.g. ["Company A", "Company B"]) without the key (i.e. 0, 1, 2)
    current_description: current company description
    current_comp_year: current company start year must be in integer
    current_comp_month: current company start month must be in integer in range of 1 to 12 if not just do null
    start_year: list of all start year of all past company and title
    start_month: list of all start month of all past company and title must be in integer in range of 1 to 12 if not just do null
    end_year: list of all end year of all past company and title
    end_month: list of all end year of all past company and title must be in integer in range of 1 to 12 if not just do null
    employment_type: permanent or contract
    location: current company location
    phone_number: Phone number
    skill: list down 5 skill that the candidate have
    proficiency: proficiency of the 5 skill you listed
    years_experience: years experience for the 5 skill you listed
    last_used_year: last used year for 5 skill u listed
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

    data["user_id"] = user_id
    data["file_url"] = url
    data["date"] = datetime.now(ZoneInfo("Asia/Kuala_Lumpur"))
    data["total_token_gemini"] = gemini_token
    data["total_token_openai"] = response2.usage.total_tokens
    data["pdf_name"] = original_filename
    data["match_acceptence"] = accpetanceVal

    if no_description:
        data["percentage_match"] = 0

    with psycopg2.connect(pg_connection_string) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cand_id = str(uuid.uuid4())
            email = user_id

            if not no_description:
                values = (
                current_uuid,
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
                cursor.execute(insert_query,values)

            # 1️⃣ Check if candidate already exists
            cursor.execute("SELECT candidate_id FROM candidates WHERE owner_email = %s AND candidate_email =%s", (email, data.get("email")))
            existing = cursor.fetchone()

            if existing:
                can_id = existing["candidate_id"]
                # If exists, UPDATE current experience
                #CANDIDATE TABLEEEEEEEEEE
                cursor.execute("""
                    UPDATE candidates
                    SET current_company = %s,
                        current_title = %s,
                        updated_at = now()
                    WHERE candidate_id = %s
                """, (
                    data.get("company"),
                    data.get("title"),
                    can_id
                ))

                #CANDIDATE EXPERIENCE TABLEEEEEEEEEE
                cursor.execute("""
                    UPDATE candidate_experience
                    SET company = %s,
                        description = %s,
                        title = %s,
                        start_year = %s,
                        start_month = %s,
                        employment_type = %s,
                        is_current = %s,
                        updated_at = now()
                    WHERE candidate_id = %s
                    AND company = %s
                """, (
                    data.get("company"),
                    data.get("current_description"),
                    data.get("title"),
                    data.get("current_comp_year"),
                    data.get("current_comp_month"),
                    data.get("employment_type"),
                    True,
                    can_id,
                    data.get("company")
                ))


            else:
                # If not exists, INSERT new current experience
                skills = data.get("skill", [])
                proficiency = data.get("proficiency", [])
                years_exp = data.get("years_experience", [])
                last_used = data.get("last_used_year", [])

                insert_current = """
                    INSERT INTO candidate_skills (
                        candidate_id, skill_name, proficiency, years_experience,
                        last_used_year, created_at
                    ) VALUES (%s, %s, %s, %s, %s, now())
                """

                for i in range(len(skills)):
                    cursor.execute(insert_current, (
                        cand_id,
                        skills[i],
                        proficiency[i],
                        years_exp[i],
                        last_used[i]
                    ))

                insert_current = """
                    INSERT INTO candidates (
                        candidate_id, owner_email, full_name, current_company, notes, current_title,
                        location, candidate_email, phone, resume_url, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                """
                
                cursor.execute(insert_current, (
                    cand_id,
                    email,
                    data.get("name"),                    
                    data.get("company"),
                    data.get("current_description"),
                    data.get("title"),
                    data.get("location"),
                    data.get("email"),
                    data.get("phone_number"),
                    url
                ))

                insert_current = """
                    INSERT INTO candidate_experience (
                        candidate_id, company, description, title,
                        start_year, start_month, employment_type, is_current
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                cursor.execute(insert_current, (
                    cand_id,
                    data.get("company"),
                    data.get("current_description"),
                    data.get("title"),
                    data.get("current_comp_year"),
                    data.get("current_comp_month"),
                    data.get("employment_type"),
                    True
                ))

            # 2️⃣ Insert past experiences (only if not duplicate company for same candidate)
            past_companies   = data.get("past_company", [])
            past_titles      = data.get("past_title", [])
            past_descriptions = data.get("description", [])
            start_years      = data.get("start_year", [])
            start_months     = data.get("start_month", [])
            end_years        = data.get("end_year", [])
            end_months       = data.get("end_month", [])

            for i, company in enumerate(past_companies):
                # Check if this past company already exists for this candidate
                cursor.execute("""
                    SELECT 1 FROM candidate_experience 
                    WHERE candidate_id = %s AND company = %s AND is_current = false
                """, (cand_id, company))
                
                if cursor.fetchone():
                    continue  # skip duplicate past company
                
                cursor.execute("""
                    INSERT INTO candidate_experience (
                        candidate_id, company, title, description,
                        start_year, start_month, end_year, end_month, is_current
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, false)
                """, (
                    cand_id,
                    company,
                    past_titles[i] if i < len(past_titles) else None,
                    past_descriptions[i] if i < len(past_descriptions) else None,
                    start_years[i] if i < len(start_years) else None,
                    start_months[i] if i < len(start_months) else None,
                    end_years[i] if i < len(end_years) else None,
                    end_months[i] if i < len(end_months) else None
                ))

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
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)