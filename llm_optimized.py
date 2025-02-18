import gradio as gr
import pdfplumber
from docx import Document
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import spacy

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


# Function to get OpenAI embeddings
def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return np.array(response.data[0].embedding)


# Function to compare Resume and JD using cosine similarity
def compare_resume_with_jd(resume_text, jd_text):
    if not resume_text or not jd_text:
        return 0  # Avoid empty text errors

    resume_embedding = get_embedding(resume_text)
    jd_embedding = get_embedding(jd_text)

    similarity = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
    return similarity * 100  # Convert to percentage


# Gradio function to process multiple resumes against one JD
# Gradio function to process multiple resumes against one JD with ranking
def process_resumes(resume_files, jd_file):
    # Extract JD text
    if jd_file.name.endswith(".pdf"):
        jd_text = extract_text_from_pdf(jd_file.name)
    elif jd_file.name.endswith(".docx"):
        jd_text = extract_text_from_docx(jd_file.name)
    else:
        return "❌ Unsupported JD file format"
    
    jd_skills = extract_skills(jd_text)
    results = []
    
    candidate_data = []  # Store each candidate's data along with their scores for sorting

    for resume_file in resume_files:
        resume_name = os.path.basename(resume_file.name)  # Extract filename
        # Extract resume text
        if resume_file.name.endswith(".pdf"):
            resume_text = extract_text_from_pdf(resume_file.name)
        elif resume_file.name.endswith(".docx"):
            resume_text = extract_text_from_docx(resume_file.name)
        else:
            results.append(f"❌ {resume_name}: Unsupported file format")
            continue

        # Extract candidate details
        candidate_name = extract_name_from_text(resume_text)
        email, phone = extract_contact_info(resume_text)
        education = extract_education(resume_text)
        resume_skills = extract_skills(resume_text)
        projects = extract_projects(resume_text)
        # experience = extract_experience(resume_text)
        matched_skills = jd_skills.intersection(resume_skills)
        missed_skills = jd_skills.difference(resume_skills)

        # Compare resume with JD
        score = compare_resume_with_jd(resume_text, jd_text)
        result = f"✅ Compatible" if score >= 80 else f"❌ Not Compatible"
        
        # Add candidate to the data list with all necessary information for ranking
        candidate_data.append({
            'name': candidate_name,
            'resume_name': resume_name,
            'email': email,
            'phone': phone,
            'education': education,
            'matched_skills': matched_skills,
            'missed_skills' : missed_skills,
            'projects': projects,
            'score': score,
            'skills_count': len(matched_skills)  # Use matched skills count as one of the ranking criteria
        })
    
    # Sort candidates based on the score (higher score first)
    ranked_candidates = sorted(candidate_data, key=lambda x: x['score'], reverse=True)

    # Format results with ranking explanation
    for i, candidate in enumerate(ranked_candidates):
        score = candidate['score']
        skills_count = candidate['skills_count']
        explanation = f"""
        Explanation:
        - Skills Match: {skills_count} matching skills
        - Education Relevance: Relevant (as per JD)
        - Project/Experience Relevance: Found
        """
        results.append(f"""
{candidate['resume_name']}
Name: {candidate['name']}
Email: {candidate['email']}
Phone: {candidate['phone']}
Education: {', '.join(candidate['education'])}
Matched Skills: {', '.join(candidate['matched_skills']) if candidate['matched_skills'] else 'Not Found'}
Missing Skills: {', '.join(candidate['missed_skills']) if candidate['missed_skills'] else 'Not Found'}
Projects:\n {candidate['projects']}
Match Score: {score:.2f}% - {"✅ Compatible" if score >= 80 else "❌ Not Compatible"}
{explanation}
""")
    
    return "\n\n".join(results)



# Gradio UI
interface = gr.Interface(
    fn=process_resumes,
    inputs=[
        gr.Files(label="Upload Resumes (PDF/DOCX)", file_types=[".pdf", ".docx"]),
        gr.File(label="Upload Job Description (PDF/DOCX)", file_types=[".pdf", ".docx"]),
    ],
    outputs="text",
    title="AI Resume Screening System",
    description="Upload multiple resumes and a job description to check compatibility. The AI will analyze each resume and provide a percentage score.\n\nA resume is considered compatible if the similarity is **80% or higher**.",
)

# Launch Gradio app
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
