import os
from dotenv import load_dotenv
from llama_index.core import Document, Settings
from llama_index.readers.file import PDFReader
from llama_index.llms.openai import OpenAI
import pickle 

#--------------------------------#
#--------- INITIAL SETUP --------#
#--------------------------------#

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("LLAMA_CLOUD_API_KEY")
if not api_key:
    raise ValueError("LLAMA_CLOUD_API_KEY not found in environment variables")

# Initialize OpenAI settings
Settings.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize PDF Reader
reader = PDFReader()

#--------------------------------#
#--------- PARSING PDF ----------#
#--------------------------------#

# Choose your PDF files
pdf_paths = [
            "PDF_files/Swap_KYC_Procedures.pdf",
            "PDF_files/Swap_EnhancedDD.pdf",
            "PDF_files/Swap_LeadAndParticipantMgmt.pdf",
            "PDF_files/Swap_SanctionsPolicy.pdf",
            "PDF_files/Swap_ABCPolicy.pdf",
            "PDF_files/Swap_AML&CTF.pdf",
            "PDF_files/Swap_DataProtection.pdf",
            "PDF_files/Swap_SuspiciousAcitivity.pdf",
            "PDF_files/Swap_Whistleblow2024.pdf"
]

# Parse all PDFs and combine the results
pdf_docs = []
for path in pdf_paths:
    # Check if file exists before trying to parse
    if not os.path.exists(path):
        print(f"Warning: File {path} not found")
        continue
        
    try:
        docs = reader.load_data(path)
        pdf_docs.extend(docs)
    except Exception as e:
        print(f"Error parsing {path}: {str(e)}")
        continue

# Print the parsed data to inspect it
print(f"Number of documents parsed: {len(pdf_docs)}")

# Verify we have documents before saving
if len(pdf_docs) == 0:
    raise ValueError("No documents were successfully parsed")

# Save the parsed data to a pickle file
if not os.path.exists("data"):
    os.makedirs("data")

try:
    with open('data/parsed_docs.pkl', 'wb') as f:
        pickle.dump(pdf_docs, f)
    print("\n✅ Parsed data saved to data/parsed_docs.pkl")
except Exception as e:
    print(f"Error saving parsed data: {str(e)}")

# Show parsed output (uncomment to see preview)
# for i, doc in enumerate(pdf_docs):
#     print(f"\n--- Document {i+1} ---\n")
#     print(doc.text[:3000])  # Preview first 3000 chars
#     print("..." if len(doc.text) > 3000 else "")