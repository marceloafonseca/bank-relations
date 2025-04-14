import os
import pickle
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser

# Ensure the "data" directory exists
if not os.path.exists("data"):
    os.makedirs("data")

# Load parsed markdown documents saved from 1.parse.py
with open("data/parsed_docs.pkl", "rb") as f:
    documents = pickle.load(f)

print(f"✅ Loaded {len(documents)} parsed documents.")

# Initialize the Markdown parser for semantic chunking
parser = MarkdownNodeParser()

# Process each document and extract semantic chunks
all_chunks = []
for doc in documents:
    # If needed, you can check the type, but ideally LlamaParse returns Document instances.
    if not isinstance(doc, Document):
        print("Warning: Document is not an instance of the expected Document type.")
    
    # Extract source information from the document
    source_file = ""
    page_number = "unknown"
    
    # Check if the document has metadata
    if hasattr(doc, 'metadata'):
        # Try to get the file path from metadata
        if 'file_path' in doc.metadata:
            source_file = doc.metadata['file_path']
        elif 'file_name' in doc.metadata:
            source_file = doc.metadata['file_name']
        
        # Try to get the page number from metadata
        if 'page_label' in doc.metadata:
            page_number = doc.metadata['page_label']
        elif 'page_number' in doc.metadata:
            page_number = doc.metadata['page_number']
    
    # If source_file is still empty, try to get it from direct attributes
    if not source_file and hasattr(doc, 'file_path'):
        source_file = doc.file_path
    elif not source_file and hasattr(doc, 'file_name'):
        source_file = doc.file_name
    
    # If page_number is still unknown, try to get it from direct attributes
    if page_number == "unknown" and hasattr(doc, 'page_label'):
        page_number = doc.page_label
    elif page_number == "unknown" and hasattr(doc, 'page_number'):
        page_number = doc.page_number
    
    # Get chunks from the document
    chunks = parser.get_nodes_from_documents([doc])
    
    # Add source information to each chunk's metadata
    for chunk in chunks:
        # Initialize metadata if it doesn't exist
        if not hasattr(chunk, 'metadata') or chunk.metadata is None:
            chunk.metadata = {}
        
        # Add source information to metadata
        chunk.metadata['source'] = source_file
        chunk.metadata['page'] = page_number
    
    all_chunks.extend(chunks)

print(f"✅ Total semantic chunks created: {len(all_chunks)}")

# Save the chunks for later embedding
with open("data/chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

print("✅ Saved chunks to data/chunks.pkl")