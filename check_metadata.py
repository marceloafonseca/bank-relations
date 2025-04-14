import os
import pickle
from pprint import pprint

# Load the parsed documents from the pickle file
print("Loading parsed documents...")
with open("data/parsed_docs.pkl", "rb") as f:
    documents = pickle.load(f)

print(f"✅ Loaded {len(documents)} documents.")

# Inspect the first document
if documents:
    print("\n=== First Document ===")
    doc = documents[0]
    
    # Print document type
    print(f"Document type: {type(doc)}")
    
    # Print available attributes
    print("\nAvailable attributes:")
    for attr in dir(doc):
        if not attr.startswith('_'):  # Skip private attributes
            print(f"- {attr}")
    
    # Print metadata if it exists
    if hasattr(doc, 'metadata'):
        print("\nMetadata content:")
        pprint(doc.metadata)
    else:
        print("\nNo metadata attribute found.")
    
    # Print text preview
    if hasattr(doc, 'text'):
        print(f"\nText preview: {doc.text[:200]}...")
    
    # Check for other potential source information
    print("\nChecking for source information in other attributes:")
    for attr in ['source', 'file_path', 'file_name', 'page_label', 'page_number']:
        if hasattr(doc, attr):
            print(f"- {attr}: {getattr(doc, attr)}")
        elif hasattr(doc, 'metadata') and attr in doc.metadata:
            print(f"- metadata['{attr}']: {doc.metadata[attr]}")
    
    # Print all documents' metadata keys to see what's available
    print("\n=== Metadata Keys Across All Documents ===")
    all_keys = set()
    for doc in documents:
        if hasattr(doc, 'metadata'):
            all_keys.update(doc.metadata.keys())
    
    print(f"Found {len(all_keys)} unique metadata keys:")
    for key in sorted(all_keys):
        print(f"- {key}")
else:
    print("No documents found.") 