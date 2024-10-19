
# 🚀 Resume Analyzer with LLM & Keyword Matching

A Python-based tool to analyze resumes using advanced **Natural Language Processing (NLP)** techniques, combining **Zero-Shot Classification** from HuggingFace and custom keyword matching to calculate scores for job requirements, skills, and experience. Results are saved in an Excel file for easy viewing.

---

## 🎯 Key Features
- **LLM-Powered Analysis:** Uses HuggingFace’s zero-shot classification model to score resumes.
- **Keyword Matching:** Custom keyword search to assess skills and experience.
- **Parallel Processing:** Efficient scoring using concurrent processing.
- **Docker & Ollama Integration:** Runs LLM models via Docker for better isolation.
- **Gradio Interface:** A user-friendly web interface for uploading and analyzing resumes.

---

## 🗂️ Project Structure

```bash
resume_analyzer/
│
├── main_advanced.py             # Main script to run the resume analysis
├── resume_extractor.py          # Contains helper functions for extracting data
├── requirements.txt             # Required pip packages
├── data/                        # Directory to store resume PDFs
├── log/                         # Directory to save analysis results (Excel files)
├── extracted_data.txt           # File where extracted resume data is saved
└── README.md                    # You're here!
```

---

## 🛠️ Dependencies

This project uses **both pip and conda** for package management due to specific library requirements.

### 📦 Pip Dependencies (from `requirements.txt`)
- **transformers**
- **gradio**
- **pandas**
- **spacy**
- **concurrent.futures** (built-in)
- **PyPDF2**
- **string**

### 🐍 Conda Dependencies
- **poppler** (required by `pdftotext` which is a sub-dependency)
- **Ollama** for running LLM models

### ⚙️ Additional Requirements
- **Docker Desktop**: Needed to run Ollama's LLM models in a containerized environment.

---

## 💻 Installation & Setup

### Step 1: Clone the Repository
```bash
git clone 
cd resume-analyzer
```

### Step 2: Set up the Virtual Environments

#### 1️⃣ Create Pip Environment
You can set up the `pip` environment first:
```bash
python -m venv env
source env/bin/activate  # For Mac/Linux
# OR
env\Scripts\activate  # For Windows

pip install -r requirements.txt
```

#### 2️⃣ Install Conda Packages
Some packages need to be installed with `conda`:
```bash
conda install -c conda-forge poppler
```

### Step 3: Install Docker & Ollama

#### 🐋 Install Docker Desktop
Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop).

#### 🧠 Install Ollama LLM
Run the following commands to pull and start the Ollama Docker image:
```bash
docker pull ollama/ollama
```
#### Ollama Setup Instructions
Download and install [ollama setup](https://hub.docker.com/r/ollama/ollama)

---

## 🚀 Running the Project

### 1️⃣ Prepare Resumes

Place the PDF resumes you want to analyze in the `data/` directory.

### 2️⃣ Execute the Main Script

```bash
python main_advanced.py
```

This will:
- Extract the text from the PDF resumes.
- Analyze the job title, skills, and experience using both keyword matching and LLM scoring.
- Save the results as an Excel file in the `log/` directory.

### 3️⃣ View Results

Check the generated Excel file in the `log/` directory for detailed scores, and refer to `extracted_data.txt` for raw extracted data from the resume.

---

## 📝 How It Works

### main_advanced.py
- **pdf_to_text**: Extracts text from PDF resumes.
- **preprocess_text**: Prepares and tokenizes the resume text.
- **calculate_llm_score**: Uses the LLM model for scoring.
- **gradio_interface**: Launches the Gradio web app for uploading resumes and displaying results.

### resume_extractor.py
- Contains helper functions such as `extract_resume_data` for text extraction and interacting with Ollama's LLM model via Docker.

## 💡 Troubleshooting & Tips

- Ensure Docker Desktop is running when using the Ollama model.
- Check the logs for any errors related to missing dependencies or Docker issues.

