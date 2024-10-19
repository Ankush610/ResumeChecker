import subprocess
from langchain_community.llms import Ollama
from PyPDF2 import PdfReader  # Importing from PyPDF2

# Define a class to handle input data and LLM interactions
class InputData:
    @staticmethod
    def input_data(text):
        # Prepare the input for the LLM
        input_text = f"""Extract relevant information from the following resume text and format it for plain text output. Ensure all specified fields are present in the output, even if the value is empty or unknown. If a specific piece of information is not found in the text, use 'Not provided' as the value.

        Resume text:
        {text}

        Instructions:
        Please extract the following fields and format the output:
        - Name
        - Email
        - Phone 1
        - Phone 2
        - Address
        - City
        - LinkedIn
        - Professional Experience (in years)
        - Highest Education
        - Is Fresher (yes/no)
        - Is Student (yes/no)
        - Skills (comma-separated)
        - Applied For Profile
        - Education (Institute Name, Year of Passing, Score)
        - Projects (Project Name, Description)
        - Professional Experience (Organisation Name, Duration, Profile)

        Ensure to format the output as:
        Name: [Value]
        Email: [Value]
        Phone 1: [Value]
        ...
        
        """
        
        return input_text
    
    @staticmethod
    def llm():
        # Initialize the Ollama model
        return Ollama(model="llama3")

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()  # Extract text from each page
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

# Function to run Docker commands
def run_docker_command(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout, result.stderr

def extract_resume_data(pdf_path):
    # Start the Ollama Docker container
    start_command = ["docker", "run", "-d", "-v", "ollama:/root/.ollama", "-p", "11434:11434", "--name", "ollama", "ollama/ollama"]
    stdout, stderr = run_docker_command(start_command)

    if stderr:
        print(f"Error starting container: {stderr}")
        return None

    print("Docker container started.")

    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)

    # Check if text extraction was successful
    if not text:
        print("No text extracted from PDF. Exiting program.")
        return None

    # Initialize the LLM
    llm = InputData.llm()

    # Invoke the LLM with the input data
    try:
        extracted_data = llm.invoke(InputData.input_data(text))
        print("Extracted Data:")
        print(extracted_data)
        
        # Save the extracted data to a text file
        with open('extracted_data.txt', 'w') as file:
            file.write(extracted_data)

        print("Data saved to extracted_data.txt")
        return extracted_data
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return None
    finally:
        # Stop and remove the Docker container after execution
        stop_command = ["docker", "stop", "ollama"]
        remove_command = ["docker", "rm", "ollama"]

        # Stop the container
        stdout, stderr = run_docker_command(stop_command)
        if stderr:
            print(f"Error stopping container: {stderr}")
        else:
            print("Docker container stopped.")

        # Remove the container
        stdout, stderr = run_docker_command(remove_command)
        if stderr:
            print(f"Error removing container: {stderr}")
        else:
            print("Docker container removed.")

