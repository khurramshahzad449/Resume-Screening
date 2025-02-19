import gradio as gr
import pdfplumber
from docx import Document
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import spacy
import pandas as pd

# Load Spacy NLP model for name extraction
nlp = spacy.load("en_core_web_sm")

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text.strip()


# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([p.text for p in doc.paragraphs]).strip()

# Function to extract text from Excel (XLSX)
def extract_text_from_xlsx(xlsx_path, num_resumes=3):
    df = pd.read_excel(xlsx_path)

    # Check if there's an unnamed column at index 32
    if df.shape[1] > 32 and df.columns[32].startswith("Unnamed"):
        resume_column = df.columns[32]  # Get the correct column name
        resumes = df[resume_column].dropna().head(num_resumes).tolist()
        return resumes
    elif "Resume Text" in df.columns:
        resumes = df["Resume Text"].dropna().head(num_resumes).tolist()
        return resumes
    else:
        print("❌ No resume text column found in the file.")
        return []


# Function to extract candidate name
def extract_name_from_text(text):
    lines = text.strip().split("\n")

    # First-line heuristic
    first_line = lines[0].strip()
    if len(first_line.split()) > 1 and "resume" not in first_line.lower():
        return first_line

    # Use NLP as fallback
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text

    return "Not Found"


# Function to extract contact information
def extract_contact_info(text):
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    phone_pattern = r"\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"

    email_match = re.findall(email_pattern, text)
    phone_match = re.findall(phone_pattern, text)

    email = email_match[0] if email_match else "Not Found"
    phone = phone_match[0] if phone_match else "Not Found"

    return email, phone


# Function to extract education details
def extract_education(text):
    education_keywords = ["Bachelor", "Master", "PhD", "Degree", "University", "B.Sc", "M.Sc", "B.E", "M.E"]
    lines = text.split("\n")
    education = [line.strip() for line in lines if any(word in line for word in education_keywords)]
    return education if education else ["Not Found"]

def extract_education_score(education_list):
    if not education_list or education_list == ["Not Found"]:
        return 0  # No education found
    elif any(degree in " ".join(education_list) for degree in ["PhD", "Doctorate"]):
        return 100  # Highest qualification
    elif any(degree in " ".join(education_list) for degree in ["Master", "M.Sc", "M.E"]):
        return 80  # Mid-level qualification
    elif any(degree in " ".join(education_list) for degree in ["Bachelor", "B.Sc", "B.E"]):
        return 60  # Basic qualification
    else:
        return 40


def extract_skills(text):
    prompt = f"Extract all professional skills from the following text:\n{text}\nReturn skills as a comma-separated list."
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an AI that extracts skills from resumes and job descriptions."},
                  {"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0
    )
    
    skills_text = response.choices[0].message.content.strip()
    
    # Debugging: Print response to check format
    # print("Extracted Skills Response:", skills_text)
    
    skills = [skill.strip() for skill in skills_text.split(",") if skill]
    
    return set(skills)


# Function to extract projects
def extract_projects(text):
    prompt = f"""
    Extract all projects from the following resume text. For each project, provide a summary highlighting technical skills, technologies, and key contributions.
    
    Resume Text:
    {text}
    
    Return the projects in the following format:
    - Project Name: [Project Name]
      Summary: [Brief description including technical skills & tools used]
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an AI that extracts and summarizes projects from resumes."},
                  {"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0
    )

    projects_text = response.choices[0].message.content.strip()
    
    # print("Extracted Projects Response:", projects_text)

    return projects_text if projects_text else "Not Found"


def extract_experience(text):
    years_pattern = r"(\d+)\s*(?:years?|yrs?)"
    matches = re.findall(years_pattern, text, re.IGNORECASE)
    experience_years = max(map(int, matches)) if matches else 0
    return min(100, (experience_years / 10) * 100) 

# Function to get OpenAI embeddings
def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return np.array(response.data[0].embedding)


# # Function to compare Resume and JD using cosine similarity
# def compare_resume_with_jd(resume_text, jd_text):
#     if not resume_text or not jd_text:
#         return 0  # Avoid empty text errors

#     resume_embedding = get_embedding(resume_text)
#     jd_embedding = get_embedding(jd_text)

#     similarity = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
#     return similarity * 100  # Convert to percentage

def update_weightage(skills, experience, education):
    global WEIGHTAGE
    WEIGHTAGE = {"skills": skills, "experience": experience, "education": education}

WEIGHTAGE = {"skills": 5, "experience": 3, "education": 2}

# Gradio function to process multiple resumes against one JD
# Gradio function to process multiple resumes against one JD with ranking
def process_resumes(xlsx_file, resume_files, jd_file, skills_weight, experience_weight, education_weight):
    # Extract text from XLSX file if provided
    extracted_resumes = []
    if xlsx_file:
        extracted_resumes = extract_text_from_xlsx(xlsx_file.name, num_resumes=5)

    # Extract JD text
    if jd_file.name.endswith(".pdf"):
        jd_text = extract_text_from_pdf(jd_file.name)
    elif jd_file.name.endswith(".docx"):
        jd_text = extract_text_from_docx(jd_file.name)
    else:
        return "❌ Unsupported JD file format"

    jd_skills = extract_skills(jd_text)
    results = []
    candidate_data = []  
    update_weightage(skills_weight, experience_weight, education_weight)

    # Combine resumes from XLSX and uploaded files
    all_resumes = []
    
    if resume_files:
        all_resumes.extend(resume_files)  # Add uploaded resume files
    
    for i, resume_text in enumerate(extracted_resumes):
        all_resumes.append((f"Extracted Resume {i+1}.txt", resume_text))  # Append extracted resume texts with a fake name

    if not all_resumes:
        return "❌ No resumes found. Please upload resumes or provide an XLSX file."

    for resume_entry in all_resumes:
        if isinstance(resume_entry, tuple):  
            resume_name, resume_text = resume_entry  # XLSX extracted resume
        else:
            resume_name = os.path.basename(resume_entry.name)  # Uploaded file
            if resume_entry.name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(resume_entry.name)
            elif resume_entry.name.endswith(".docx"):
                resume_text = extract_text_from_docx(resume_entry.name)
            else:
                results.append(f"❌ {resume_name}: Unsupported file format")
                continue

        # Extract candidate details
        candidate_name = extract_name_from_text(resume_text)
        email, phone = extract_contact_info(resume_text)
        education = extract_education(resume_text)
        resume_skills = extract_skills(resume_text)
        projects = extract_projects(resume_text)
        matched_skills = jd_skills.intersection(resume_skills)
        missed_skills = jd_skills.difference(resume_skills)

        skills_score = (len(matched_skills) / len(jd_skills)) * 100 if jd_skills else 0
        education_score = extract_education_score(education)
        experience_score = extract_experience(resume_text)

        weighted_score = (
            (skills_score * WEIGHTAGE["skills"]) +
            (education_score * WEIGHTAGE["education"]) +
            (experience_score * WEIGHTAGE["experience"])
        ) / sum(WEIGHTAGE.values())

        candidate_data.append({
            'name': candidate_name,
            'resume_name': resume_name,
            'email': email,
            'phone': phone,
            'education': education,
            'matched_skills': matched_skills,
            'missed_skills' : missed_skills,
            'projects': projects,
            'skills_count': len(matched_skills),  
            'score': weighted_score,
            'skills_score' : skills_score,
            'education_score' : education_score,
            'experience_score' : experience_score
        })

    ranked_candidates = sorted(candidate_data, key=lambda x: x['score'], reverse=True)

    for i, candidate in enumerate(ranked_candidates):
        score = candidate['score']
        skills_count = candidate['skills_count']
    
        results.append(f"""
{candidate['resume_name']}
Name: {candidate['name']}
Email: {candidate['email']}
Phone: {candidate['phone']}
Education: {', '.join(candidate['education'])}
Matched Skills: {', '.join(candidate['matched_skills']) if candidate['matched_skills'] else 'Not Found'}
Missing Skills: {', '.join(candidate['missed_skills']) if candidate['missed_skills'] else 'Not Found'}
Projects:\n {candidate['projects']}
Match Score: {candidate['score']:.2f}% - {"✅ Compatible" if score >= 80 else "❌ Not Compatible"}

Explanation:
- Skills Match: {skills_count} matching skills
- Skills Score: {candidate['skills_score']:.2f}%
- Education Score: {candidate['education_score']:.2f}%
- Experience Score: {candidate['experience_score']:.2f}%
""")
    
    return "\n\n".join(results)




# Gradio UI
interface = gr.Interface(
    fn=process_resumes,
    inputs=[
        gr.File(label="Upload XLSX (Resumes)", file_types=[".xlsx"]),
        gr.Files(label="Upload Resumes (PDF/DOCX)", file_types=[".pdf", ".docx"]),
        gr.File(label="Upload Job Description (PDF/DOCX)", file_types=[".pdf", ".docx"]),
        gr.Slider(1, 10, value=5, label="Skills Weightage"),
        gr.Slider(1, 10, value=3, label="Experience Weightage"),
        gr.Slider(1, 10, value=2, label="Education Weightage")
    ],
    outputs="text",
    title="AI Resume Screening System",
    description="Upload multiple resumes and a job description to check compatibility. The AI will analyze each resume and provide a percentage score.\n\nA resume is considered compatible if the similarity is **80% or higher**.",
)

# Launch Gradio app
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
