import PyPDF2
from transformers import pipeline
import spacy
import gradio as gr
import pandas as pd
import string
import os
import concurrent.futures  # For parallel processing
import logging  # For better fault tolerance
from resume_extractor import extract_resume_data

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the log directory exists
LOG_DIR = 'log'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Load models once globally to avoid reloading them in each function
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
nlp = spacy.load('en_core_web_sm')

# Preprocess text for tokenization with caching for multi-word phrases
def preprocess_text(text, multi_word_phrases):
    text = text.lower()
    for phrase in multi_word_phrases:
        text = text.replace(phrase, phrase.replace(' ', '_'))
    doc = nlp(text)
    tokens = [token.text for token in doc if token.text not in spacy.lang.en.stop_words.STOP_WORDS and token.text not in string.punctuation and len(token.text) > 1]
    return tokens

# Keyword search optimized with set lookup
def keyword_search(tokens, requirements):
    keyword_set = set(requirements)
    keyword_scores = [1 if req in tokens else 0 for req in requirements]
    return keyword_scores

# Calculate LLM score asynchronously for each requirement
def calculate_llm_score(text, requirement):
    try:
        classification_score = classifier(text, [requirement])
        return classification_score['scores'][0]  # First score for the given label
    except Exception as e:
        logging.error(f"Error calculating LLM score for {requirement}: {e}")
        return 0

# Combine scores for all requirements in a field and return dataframe
def calculate_combined_score(text, requirements, keyword_scores):
    llm_scores = []
    
    # Use ThreadPoolExecutor for parallel processing of LLM scoring
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_req = {executor.submit(calculate_llm_score, text, req): req for req in requirements}
        for future in concurrent.futures.as_completed(future_to_req):
            req = future_to_req[future]
            try:
                llm_scores.append(future.result())
            except Exception as e:
                logging.error(f"Error processing LLM score for {req}: {e}")
                llm_scores.append(0)

    llm_avg_score = sum(llm_scores) / len(llm_scores) if llm_scores else 0
    normalized_keyword_score = sum(keyword_scores) / len(requirements) if requirements else 0
    combined_score = (llm_avg_score + normalized_keyword_score) / 2

    # Create DataFrame for the results
    results = {
        'Keyword': requirements,
        'LLM Probability (%)': [round(score * 100, 2) for score in llm_scores],
        'Keyword Found (1/0)': keyword_scores
    }
    df = pd.DataFrame(results)

    return llm_avg_score, normalized_keyword_score, combined_score, df

# Save results to Excel file in log directory
def save_to_excel(name, job_title, skills, experience, llm_score, keyword_score, overall_score):
    filename = os.path.join(LOG_DIR, "resume_analysis.xlsx")
    new_entry = {
        'Name': name,
        'Job Title': job_title.lower(),
        'Skills': skills.lower(),
        'Experience': experience.lower(),
        'LLM Score (%)': llm_score,
        'Keyword Score (%)': keyword_score,
        'Overall Score (%)': overall_score
    }
    
    try:
        if os.path.exists(filename):
            existing_df = pd.read_excel(filename)
            new_df = pd.DataFrame([new_entry])
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_excel(filename, index=False)
        else:
            pd.DataFrame([new_entry]).to_excel(filename, index=False)
        logging.info(f"Results saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving to Excel: {e}")

# Process the resume and calculate all scores
def process_resume(pdf_file, job_title, skills, experience):
    pdf_data = extract_resume_data(pdf_file)

    fields = {
        'Job Title': job_title,
        'Skills': skills,
        'Experience': experience
    }

    all_llm_scores = []
    all_keyword_scores = []
    combined_dfs = {}

    # Process each field in parallel
    for field_name, field_value in fields.items():
        requirements = [req.strip().lower() for req in field_value.split(',')]
        multi_word_phrases = [req.lower() for req in requirements if ' ' in req]
        tokens = preprocess_text(pdf_data, multi_word_phrases)
        tokens = [token.replace('_', ' ') for token in tokens]

        keyword_scores = keyword_search(tokens, requirements)
        llm_avg_score, keyword_avg_score, combined_score, df = calculate_combined_score(pdf_data, requirements, keyword_scores)

        all_llm_scores.append(llm_avg_score)
        all_keyword_scores.append(keyword_avg_score)

        combined_dfs[field_name] = df

    final_llm_avg_score = sum(all_llm_scores) / len(all_llm_scores)
    final_keyword_avg_score = sum(all_keyword_scores) / len(all_keyword_scores)
    final_combined_score = (final_llm_avg_score + final_keyword_avg_score) / 2

    name = os.path.splitext(os.path.basename(pdf_file.name))[0]

    # Save results to Excel
    save_to_excel(
        name,
        job_title,
        skills,
        experience,
        round(final_llm_avg_score * 100, 2),
        round(final_keyword_avg_score * 100, 2),
        round(final_combined_score * 100, 2)
    )

    return round(final_llm_avg_score * 100, 2), round(final_keyword_avg_score * 100, 2), round(final_combined_score * 100, 2), combined_dfs, pdf_data

## GUI =============================================================================

def gradio_interface(pdf_file, job_title, skills, experience):
    llm_score, keyword_score, overall_score, combined_dfs, extracted_data = process_resume(pdf_file, job_title, skills, experience)
    return (
        llm_score, keyword_score, overall_score,
        combined_dfs['Job Title'], combined_dfs['Skills'], combined_dfs['Experience'],
        extracted_data  # Show extracted data in the interface
    )

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("## Resume Analyzer with LLM and Keyword Matching")

    with gr.Tab("Input & Scores"):
        with gr.Row():
            pdf_input = gr.File(label="Upload Resume (PDF)")
            llm_output = gr.Label(label="Average LLM Score (%)")
            keyword_output = gr.Label(label="Average Keyword Score (%)")
            overall_output = gr.Label(label="Overall Average Score (%)")
        
        job_title_input = gr.Textbox(label="Job Title Requirements (comma-separated)", placeholder="e.g., software engineer, data scientist")
        skills_input = gr.Textbox(label="Skills Requirements (comma-separated)", placeholder="e.g., python, deep learning, java")
        experience_input = gr.Textbox(label="Experience Requirements (comma-separated)", placeholder="e.g., 3 years, 5+ years experience")

        submit_button = gr.Button("Analyze Resume")

    with gr.Tab("Tables"):
        table_job_title_output = gr.DataFrame(headers=["Keyword", "LLM Probability (%)", "Keyword Found (1/0)"], label="Job Title - Keyword Probabilities")
        table_skills_output = gr.DataFrame(headers=["Keyword", "LLM Probability (%)", "Keyword Found (1/0)"], label="Skills - Keyword Probabilities")
        table_experience_output = gr.DataFrame(headers=["Keyword", "LLM Probability (%)", "Keyword Found (1/0)"], label="Experience - Keyword Probabilities")

    with gr.Tab("Extracted Resume"):
        extracted_resume_output = gr.Textbox(label="Extracted Resume Text", lines=20, placeholder="The extracted resume text will be displayed here.")

    # Link the extracted resume output to the process_resume function
    submit_button.click(gradio_interface, inputs=[pdf_input, job_title_input, skills_input, experience_input],
                        outputs=[llm_output, keyword_output, overall_output,
                                 table_job_title_output, table_skills_output, table_experience_output,
                                 extracted_resume_output])  # Include extracted resume text output

# Run the Gradio app
demo.launch(share=False)
