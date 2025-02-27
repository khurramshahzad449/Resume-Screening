import gradio as gr
from docx import Document
import fitz 
import sqlite3
from openai import OpenAI
import os
import pandas as pd

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
db_name = 'jd_database.db'

def setup_db():
    conn = sqlite3.connect(db_name, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS job_descriptions (
        id INTEGER PRIMARY KEY,
        filename TEXT UNIQUE,
        content TEXT
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS jd_results (
        id INTEGER PRIMARY KEY,
        jd_filename TEXT,
        skill TEXT,
        weightage str,
        hr_comment TEXT,
        rationale TEXT,
        FOREIGN KEY (jd_filename) REFERENCES job_descriptions(filename)
    )''')
    conn.commit()
    conn.close()

def save_to_db(filename, content):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    filename = os.path.basename(filename)
    cursor.execute("SELECT * FROM job_descriptions WHERE filename=?", (filename,))
    existing_job = cursor.fetchone()
    
    if not existing_job:  
        cursor.execute("INSERT INTO job_descriptions (filename, content) VALUES (?, ?)", (filename, content))
        conn.commit()
    
    conn.close()

def save_results_to_db(jd_filename, results):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM jd_results WHERE jd_filename=?", (jd_filename,))  # Overwrite previous results
    
    for result in results:
        jd_filename = str(result[0])
        skill = str(result[1])
        weightage = str(result[2]) if result[2] else ""  # Handle empty values safely
        hr_comment = str(result[3]) if result[3] else ""  # New Column
        rationale = str(result[4]) if result[4] else ""
        
        cursor.execute(
            "INSERT INTO jd_results (jd_filename, skill, weightage, hr_comment, rationale) VALUES (?, ?, ?, ?, ?)", 
            (jd_filename, skill, weightage, hr_comment, rationale)
        )
    
    conn.commit()
    conn.close()

def get_existing_job_descriptions():
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM job_descriptions")
    job_descs = cursor.fetchall()
    conn.close()
    return [jd[0] for jd in job_descs]

def get_saved_results(jd_filename):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT skill, weightage, '', rationale FROM jd_results WHERE jd_filename=?", (jd_filename,))
    results = cursor.fetchall()
    conn.close()
    return results


def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file.name)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_doc(doc_file):
    doc = Document(doc_file.name)
    text = ""
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text

# def extract_skills_from_text(text):
#     prompt = f"Please extract the top 10 most important skills from the following job description, considering the context and importance of each skill:\n\n{text}\n\nProvide a list of skills in bullet points, only listing the skill names, no extra details."
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[{
#             "role": "system", "content": "You are a helpful assistant that extracts top skills from job descriptions. Please provide only the skill names."
#         }, {
#             "role": "user", "content": prompt
#         }],
#         max_tokens=150,
#         n=1,
#         stop=None,
#         temperature=0.5
#     )
#     skills = response.choices[0].message.content.strip().split("\n")
#     return skills

def assign_weightage_to_skills(content):
    prompt =  f"""
You are an expert Prompt Engineer and Skill Weightage Analyst. Your task is to:

1. Thoroughly review the Job Description (JD) below:
   {content}

2. Identify the *top 10 essential skills* required for the role, taking into account:
   - How frequently each skill is mentioned in the JD.
   - Its direct impact on the role’s key responsibilities.
   - Its significance to overall job performance and success.

3. Assign a *unique weightage* (1–10) to each skill using the following **exact definitions**:

   - *10*  
     A “must-have” skill, cited frequently and integral to overall success. The role cannot function without it.

   - *9*  
     A highly crucial skill, strongly emphasized and vital to performance, yet slightly less all-encompassing than a 10.

   - *8*  
     An essential skill that is central to many responsibilities, though not the single most defining factor.

   - *7*  
     An important skill with clear impact on the role’s success but not always front and center.

   - *6*  
     A supportive skill that significantly contributes to efficiency and quality, though not a strict core requirement.

   - *5*  
     A moderately important skill, beneficial in multiple areas but not indispensable.

   - *4*  
     A useful skill that offers tangible value but remains secondary to higher-priority competencies.

   - *3*  
     A minor skill, rarely mentioned or situationally relevant, though still potentially helpful.

   - *2*  
     A peripheral skill that appears to have minimal direct impact on core tasks, referenced sparingly if at all.

   - *1*  
     A “nice-to-have” or bonus skill mentioned briefly or potentially implied, but not essential for day-to-day work.

4. *Justify each weight* in a concise paragraph:
   - State how often the skill appears in the JD (frequency).
   - Explain how it ties into the main duties (relevance).
   - Demonstrate whether it is indispensable, important, supportive, or supplementary (alignment).

5. Present your findings as a *numbered list* of the 10 skills with their *unique* weightages and justifications, for example:

   1. Skill Name - 10  
      Reasoning: [Highlight frequency, relevance, and why it's indispensable]

   2. Skill Name - 9  
      Reasoning: [Highlight frequency, relevance, and why it's crucial but slightly less than 10]

   ...

*Important Notes:*
- *No two skills* should receive the *same weight*.
- Ensure each justification is *clear, well-structured*, and directly tied to the JD.

When done, provide your final output in the specified format.

"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "Extract exactly 10 key skills and assign weightages from a job description."},
                  {"role": "user", "content": prompt}],
        temperature=0
    )

    if not response.choices or not response.choices[0].message.content.strip():
        print("Warning: Empty response from API")
        return []

    response_text = response.choices[0].message.content.strip()
    # print("Debug: API Response ->", response_text)  

    # Process response
    weighted_skills = []
    lines = response_text.split("\n")

    skill_name, weightage, rationale = None, None, ""
    for line in lines:
        if " - " in line:
            try:
                skill_name, weight = line.rsplit(" - ", 1)
                weightage = int(weight.strip())
                skill_name = skill_name.split(". ", 1)[1]  
            except ValueError:
                continue  
        elif "Reasoning:" in line:
            rationale = line.replace("Reasoning:", "").strip()
            if skill_name and weightage:
                weighted_skills.append({
                    "skill": skill_name,
                    "weightage": weightage,
                    "rationale": rationale
                    })
            skill_name, weightage, rationale = None, None, ""  
        elif skill_name and weightage:
         rationale += " " + line.strip()  

    return weighted_skills

def get_existing_job_descriptions():
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM job_descriptions")
    job_descs = cursor.fetchall()
    conn.close()
    return [jd[0] for jd in job_descs]

def process_job_description(file, existing_jd):
    if existing_jd:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM job_descriptions WHERE filename=?", (existing_jd,))
        content = cursor.fetchone()[0]
        conn.close()
        
        results = get_saved_results(existing_jd)
        if results:
            return results, get_existing_job_descriptions(), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
    else:
        if file.name.endswith('.pdf'):
            content = extract_text_from_pdf(file)
        elif file.name.endswith('.docx'):
            content = extract_text_from_doc(file)
        if file is not None:
            filename = os.path.basename(file.name)  
            save_to_db(filename, content)
    
    weighted_skills = assign_weightage_to_skills(content)
    weighted_skills_with_extra_column = [[item["skill"], item["weightage"], "", item["rationale"]] for item in weighted_skills]
    
    return weighted_skills_with_extra_column, get_existing_job_descriptions(), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("### Job Description Skill Extraction & Weightage")
        
        with gr.Column(visible=True) as input_section:
            dropdown_jd = gr.Dropdown(
                label="Select a Previously Uploaded Job Description",
                choices=get_existing_job_descriptions(),
                value=None,
                allow_custom_value=False,
            )
            file_input = gr.File(label="Or Upload a New Job Description (PDF/DOCX)")
            submit_button = gr.Button("Submit")
        
        output_table = gr.DataFrame(
            label="Skills, Weightage and HR Comments", 
            headers=["Skill", "Weightage", "HR Comments", "Rationale"], 
            interactive=True,
            visible=False
        )

        with gr.Column(visible=False) as resume_section:
            gr.Markdown("### Upload Resumes for Screening")
            resume_files = gr.Files(label="Upload Resumes (PDF/DOCX/XLSX)", file_types=[".pdf", ".docx", ".xlsx"])
            process_resumes_button = gr.Button("Process Resumes")
            # resume_output = gr.DataFrame(label="Resume Compatibility Results", headers=["Candidate", "Score", "Remarks"], visible=False)
        
        save_button = gr.Button("Save Results", visible=False)
        
        def process_and_toggle_visibility(file, existing_jd):
            results, updated_jd_list, input_hide, table_show, save_show = process_job_description(file, existing_jd)
            return results, gr.update(choices=updated_jd_list, value=existing_jd), input_hide, table_show, save_show
        
        def extract_data_from_gradio_df(gradio_df, filename):

            if isinstance(gradio_df, pd.DataFrame):  
                # Convert pandas DataFrame to list of lists
                rows = gradio_df.values.tolist()
                print("DEBUG: Extracted Rows (from DataFrame) ->", rows)

            elif isinstance(gradio_df, dict):  
                # Convert dictionary format to list of lists (row-wise)
                rows = list(zip(*gradio_df.values())) 
                print("rows dic : ", rows) 

            elif isinstance(gradio_df, list):  
                # Ensure it's a list of lists format
                rows = gradio_df  
                print("rows list : ", rows) 

            else:
                print("Unexpected Gradio DataFrame format:", type(gradio_df))
                return []

            extracted_data = []
            for row in rows:
                if len(row) == 4:  # Ensure we have exactly 4 columns
                    skill, weightage, hr_comment, rationale = row
                    print(skill, weightage, hr_comment, rationale)
                    
                    extracted_data.append({
                        "filename": str(filename),
                        "skill": str(skill),
                        "weightage": int(weightage) if str(weightage).isdigit() else 0,  # Handle numeric conversion safely
                        "hr_comment": str(hr_comment),
                        "rationale": str(rationale)
                        })
    
            return extracted_data  # Returns a structured list of dicts
        
        def save_results(jd_filename, gradio_df):
            extracted_results = extract_data_from_gradio_df(gradio_df, jd_filename)
            print("DEBUG: Extracted Data ->", extracted_results, jd_filename)  # Debugging output
            results = [[item["filename"],item["skill"], item["weightage"], item["hr_comment"], item["rationale"]] for item in extracted_results]
            save_results_to_db(jd_filename, results)  # Save to DB
            return (
                gr.update(value="Results Saved Successfully!"),  # Update button text
                gr.update(visible=True),   # Show resume upload section
                gr.update(visible=False),  # Hide JD output table
                gr.update(visible=False)   # Hide "Save Results" button
                )
        
        submit_button.click(
            fn=process_and_toggle_visibility, 
            inputs=[file_input, dropdown_jd], 
            outputs=[output_table, dropdown_jd, input_section, output_table, save_button]
        )
        
        save_button.click(
            fn=save_results,
            inputs=[dropdown_jd, output_table],
            outputs=[save_button, resume_section, output_table, save_button]
        )



    
    demo.launch(debug=True, server_name="0.0.0.0", server_port=7860)

setup_db()
create_gradio_interface()
