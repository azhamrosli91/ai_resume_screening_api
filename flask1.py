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
import random
from faker import Faker

fake = Faker()
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

def dummy_data():
    data = {
        "LOG_HISTORY_ID": str(uuid.uuid4()),
        "name": fake.name(),
        "title": random.choice(["Software Engineer", "Retail Sales Executive", "Data Analyst", "Project Manager"]),
        "job_description": fake.text(max_nb_chars=300),
        "email": fake.email(),
        "company": fake.company(),
        "past_company": [fake.company() for _ in range(3)],
        "description": [fake.text(max_nb_chars=100) for _ in range(3)],
        "past_title": [random.choice(["Sales Assistant", "Manager", "Trainer", "Executive"]) for _ in range(3)],
        "current_description": fake.text(max_nb_chars=120),
        "current_comp_year": random.randint(2010, 2024),
        "current_comp_month": random.randint(1, 12),
        "start_year": [random.randint(2000, 2015) for _ in range(3)],
        "start_month": [random.randint(1, 12) for _ in range(3)],
        "end_year": [random.randint(2016, 2024) for _ in range(3)],
        "end_month": [random.randint(1, 12) for _ in range(3)],
        "employment_type": random.choice(["permanent", "contract", "internship"]),
        "location": fake.city(),
        "phone_number": fake.phone_number(),
        "skill": random.sample(["Sales", "Python", "Excel", "Customer Service", "Project Management", "Team Leadership"], 3),
        "proficiency": [random.randint(5, 10) for _ in range(3)],
        "years_experience": [random.randint(1, 15) for _ in range(3)],
        "last_used_year": [random.randint(2018, 2024) for _ in range(3)],
        "percentage_match": random.randint(0, 100),
        "short_description": fake.sentence(nb_words=15)
    }
    
    return jsonify(data)


# Evaluation
def evaluate_resume(pdf_path, original_filename, job_desc, user_id, url, acceptance = 70, is_dummy = False, job_title = '', id_MM_user = ''):

    if is_dummy:
        return dummy_data()
    
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
    if they dont put any year or month just assume they dont have a current company and put all in past company and title
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
    current_comp_year: current company start year must be in integer if null must return 0 and dont make up any answer must be based on resume
    current_comp_month: current company start month must be in integer in range of 1 to 12 if null must return 0 and dont make up any answer must be based on resume
    start_year: list of all start year of all past company and title and dont make up any answer must be based on resume if null must return 0
    start_month: list of all start month of all past company and title must be in integer in range of 1 to 12 if null must return 0 and dont make up any answer must be based on resume
    end_year: list of all end year of all past company and title and dont make up any answer must be based on resume if null must return 0
    end_month: list of all end year of all past company and title must be in integer in range of 1 to 12 if null must return 0 and dont make up any answer must be based on resume
    employment_type: permanent or contract
    location: current company location
    phone_number: Phone number
    skill: list down 5 skill that the candidate have
    proficiency: proficiency of the 5 skill you listed
    years_experience: years experience for the 5 skill you listed must return in list for each skill or return null if you dont know
    last_used_year: last used year for 5 skill u listed or return null if you dont know
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
    current_comp_month: current company start month must be in integer in range of 1 to 12
    start_year: list of all start year of all past company and title
    start_month: list of all start month of all past company and title must be in integer in range of 1 to 12
    end_year: list of all end year of all past company and title must be null if for current company
    end_month: list of all end year of all past company and title must be in integer in range of 1 to 12 must be null if for current company
    employment_type: permanent or contract
    location: current company location
    phone_number: Phone number
    skill: list down 5 skill that the candidate have
    proficiency: proficiency of the 5 skill you listed
    years_experience: years experience for the 5 skill you listed must return in list for each skill
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
    data["match_acceptence" ] = accpetanceVal
    data["LOG_HISTORY_ID"] = current_uuid
    
    if job_title:  # this is True if job_title is not None or not empty
        data["title"] = job_title
    
    if no_description:
        data["percentage_match"] = 0

    with psycopg2.connect(pg_connection_string) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cand_id = str(uuid.uuid4())

            # --- LOG_HISTORY insert (unchanged) ---
            if not no_description:
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
                    "LOG_HISTORY_ID", user_id, date_run, title, job_description,
                    file_url, name, email, phone_no, match_percentage,
                    short_desc, is_shortlisted, gpt_token, gemini_token, match_acceptance
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(insert_query, values)

            # --- find existing candidate ---
            cursor.execute(
                "SELECT candidate_id FROM candidates WHERE owner_email = %s AND candidate_email = %s",
                (user_id, data.get("email"))
            )
            existing = cursor.fetchone()

            # unify candidate_id variable so later code can always use it
            if existing:
                candidate_id = existing["candidate_id"]

                # update candidate top-level fields
                cursor.execute("""
                    UPDATE candidates
                    SET current_company = %s,
                        current_title = %s,
                        updated_at = now()
                    WHERE candidate_id = %s
                """, (
                    data.get("company"),
                    data.get("title"),
                    candidate_id
                ))

                # Handle current company experience: update if exists, else insert
                current_year = data.get("current_comp_year")
                current_month = data.get("current_comp_month")

                # Only consider a "current job" entry when we have both start year & month (and valid > 0)
                if (current_year is not None and current_month is not None
                        and isinstance(current_year, int) and isinstance(current_month, int)
                        and current_year > 0 and current_month > 0):

                    # Try updating an existing candidate_experience row for this company
                    cursor.execute("""
                        UPDATE candidate_experience
                        SET company = %s,
                            description = %s,
                            title = %s,
                            start_year = %s,
                            start_month = %s,
                            employment_type = %s,
                            end_year = NULL,
                            end_month = NULL,
                            is_current = TRUE,
                            updated_at = now()
                        WHERE candidate_id = %s AND company = %s
                    """, (
                        data.get("company"),
                        data.get("current_description"),
                        data.get("title"),
                        current_year,
                        current_month,
                        data.get("employment_type"),
                        candidate_id,
                        data.get("company")
                    ))

                    # If no row was updated, insert it (explicit end_year/end_month NULL, is_current TRUE)
                    if cursor.rowcount == 0:
                        cursor.execute("""
                            INSERT INTO candidate_experience (
                                candidate_id, company, title, description,
                                start_year, start_month, end_year, end_month, employment_type, is_current, created_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, NULL, NULL, %s, TRUE, now())
                        """, (
                            candidate_id,
                            data.get("company"),
                            data.get("title"),
                            data.get("current_description"),
                            current_year,
                            current_month,
                            data.get("employment_type")
                        ))

            else:
                # New candidate path
                candidate_id = cand_id

                # insert skills
                skills = data.get("skill", [])
                proficiency = data.get("proficiency", [])
                years_exp = data.get("years_experience", [])
                last_used = data.get("last_used_year", [])

                insert_skill_q = """
                    INSERT INTO candidate_skills (
                        candidate_id, skill_name, proficiency, years_experience, last_used_year, created_at
                    ) VALUES (%s, %s, %s, %s, %s, now())
                """
                for i in range(len(skills)):
                    cursor.execute(insert_skill_q, (
                        candidate_id,
                        skills[i],
                        proficiency[i] if i < len(proficiency) else None,
                        years_exp[i] if i < len(years_exp) else None,
                        last_used[i] if i < len(last_used) else None
                    ))

                # insert candidate master row
                insert_candidate_q = """
                    INSERT INTO candidates (
                        candidate_id, owner_email, full_name, current_company, notes, current_title,
                        location, candidate_email, phone, resume_url,user_id, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                """
                cursor.execute(insert_candidate_q, (
                    candidate_id,
                    user_id,
                    data.get("name"),
                    data.get("company"),
                    data.get("current_description"),
                    data.get("title"),
                    data.get("location"),
                    data.get("email"),
                    data.get("phone_number"),
                    url,
                    user_id
                ))

                # Insert current experience row if start date available
                current_year = data.get("current_comp_year")
                current_month = data.get("current_comp_month")
                if (current_year is not None and current_month is not None
                        and isinstance(current_year, int) and isinstance(current_month, int)
                        and current_year > 0 and current_month > 0):
                    cursor.execute("""
                        INSERT INTO candidate_experience (
                            candidate_id, company, title, description,
                            start_year, start_month, end_year, end_month, employment_type, is_current, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, NULL, NULL, %s, TRUE, now())
                    """, (
                        candidate_id,
                        data.get("company"),
                        data.get("title"),
                        data.get("current_description"),
                        current_year,
                        current_month,
                        data.get("employment_type")
                    ))

            # --- Insert past experiences ---
            past_companies = data.get("past_company", [])
            past_titles = data.get("past_title", [])
            past_descriptions = data.get("description", [])
            start_years = data.get("start_year", [])
            start_months = data.get("start_month", [])
            end_years = data.get("end_year", [])
            end_months = data.get("end_month", [])

            for i, company in enumerate(past_companies):
                # skip duplicate past company rows for this candidate where is_current = false
                cursor.execute("""
                    SELECT 1 FROM candidate_experience
                    WHERE candidate_id = %s AND company = %s AND is_current = FALSE
                """, (candidate_id, company))
                if cursor.fetchone():
                    continue

                # extract safely
                start_year = start_years[i] if i < len(start_years) else None
                start_month = start_months[i] if i < len(start_months) else None
                end_year = end_years[i] if i < len(end_years) else None
                end_month = end_months[i] if i < len(end_months) else None

                # normalize inconsistent data:
                # if end_year is None, force end_month to None as well (month without year is meaningless)
                if end_year is None:
                    end_month = None

                # determine is_current: only True when both end_year and end_month are None
                is_current = (end_year is None and end_month is None)

                # If end_year exists then it's a past job -> is_current False
                if end_year is not None:
                    is_current = False

                cursor.execute("""
                    INSERT INTO candidate_experience (
                        candidate_id, company, title, description,
                        start_year, start_month, end_year, end_month, is_current, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                """, (
                    candidate_id,
                    company,
                    past_titles[i] if i < len(past_titles) else None,
                    past_descriptions[i] if i < len(past_descriptions) else None,
                    start_year,
                    start_month,
                    end_year,
                    end_month,
                    is_current
                ))

            # Track the candidate_track, user_id, and resume_id is exsit or not
            cursor.execute("""
                SELECT 1 FROM candidate_track
                WHERE user_id = %s AND resume_id = %s
                LIMIT 1
            """, (id_MM_user, candidate_id))

            track_exists = cursor.fetchone()

            print(id_MM_user)

            if not track_exists:
                cursor.execute("""
                    INSERT INTO candidate_track (user_id, resume_id)
                    VALUES (%s, %s)
                """,(
                    id_MM_user,
                    candidate_id
                ))

            conn.commit()


    return data

@app.route('/evaluate-resume', methods=['POST'])
def upload_resume():
    job_desc = request.form.get('job_desc')
    user_id = request.form.get('user_id')
    acceptance = request.form.get('acceptance')
    is_dummy = request.form.get('is_dummy')

    if not job_desc:
        return jsonify({'error': 'No job description provided'}), 400

    # ✅ Handle dummy mode first
    if is_dummy and str(is_dummy).lower() == "true":
        result = evaluate_resume(
            pdf_path=None,
            original_filename=None,
            job_desc=job_desc,
            user_id=user_id,
            url=None,
            acceptance=acceptance,
            is_dummy=True
        )
        return result  # already jsonify() inside dummy_data()

    # === Normal flow with file ===
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # ✅ Read file content into memory
        file_bytes = file.read()

        # 🟢 Upload to remote API
        upload_response = requests.post(
            "http://webapifileupload.aiscreenmax.my/api/ResumeUpload/UploadFile",
            files={'file_url': (file.filename, BytesIO(file_bytes))},
            data={"foldername": "dev"}
        )

        # Extract URL string
        url = upload_response.text  # or upload_response.json().get("url")

        # 🟢 Save file locally
        original_filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            result = evaluate_resume(
                tmp.name,
                original_filename,
                job_desc,
                user_id=user_id,
                url=url,
                acceptance=acceptance,
                is_dummy=False
            )

        return jsonify(result)

    except Exception as e:
        print(f"❌ Backend error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)