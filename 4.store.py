import os
import pickle
from dotenv import load_dotenv
from tqdm import tqdm
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import json
import uuid

# Load environment variables
load_dotenv()

# Load your precomputed chunk embeddings from file
with open("data/chunks_with_embeddings.pkl", "rb") as f:
    chunk_embeddings = pickle.load(f)

print(f"✅ Loaded {len(chunk_embeddings)} chunk embeddings.")

# Prepare Document objects and a list of corresponding embeddings
documents = []
precomputed_embeddings = []

for item in tqdm(chunk_embeddings, desc="Preparing Documents"):
    chunk = item["chunk"]
    # Create a Document from the chunk; ensure your chunk has a .text attribute and optionally metadata
    doc = Document(text=chunk.text, metadata=getattr(chunk, "metadata", {}))
    documents.append(doc)
    precomputed_embeddings.append(item["embedding"])

# Initialize ChromaDB client - change to PersistentClient to save data
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Check if collection exists and delete it if it does
try:
    existing_collection = chroma_client.get_collection("rag_collection")
    if existing_collection:
        print(f"Collection 'rag_collection' already exists with {existing_collection.count()} documents.")
        print("Deleting existing collection...")
        chroma_client.delete_collection("rag_collection")
        print("Existing collection deleted.")
except Exception as e:
    # Collection doesn't exist, which is fine
    pass

# Create a fresh collection
chroma_collection = chroma_client.create_collection("rag_collection")
print("Created new 'rag_collection' collection.")

# Manually add documents to ChromaDB with their embeddings
print("Adding documents to ChromaDB...")
ids = []
texts = []
metadatas = []
embeddings = []

for i, (doc, embedding) in enumerate(zip(documents, precomputed_embeddings)):
    doc_id = str(uuid.uuid4())
    ids.append(doc_id)
    texts.append(doc.text)
    metadatas.append(doc.metadata)
    embeddings.append(embedding)
    
    # Add in batches of 100 to avoid memory issues
    if len(ids) >= 100 or i == len(documents) - 1:
        chroma_collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        # Clear the lists for the next batch
        ids = []
        texts = []
        metadatas = []
        embeddings = []

# Initialize the Chroma vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index from storage context
index = VectorStoreIndex(
    nodes=documents,
    storage_context=storage_context,
    embed_model=None  # Since we're using pre-computed embeddings
)

print("✅ Added documents to the vector store.")
print("✅ Vector store initialized with collection 'rag_collection'")

# Print the first 10 entries in the Chroma database
print("\n=== First 10 entries in Chroma database ===")

# Check if the collection has any documents
count = chroma_collection.count()
print(f"Collection contains {count} documents")

# Get the first 10 document IDs
all_ids = chroma_collection.get(limit=3)['ids']
if all_ids:
    # Get details for these IDs - explicitly include embeddings
    collection_data = chroma_collection.get(
        ids=all_ids,
        include=["embeddings", "documents", "metadatas"]
    )
    
    print(f"Retrieved {len(all_ids)} documents")
    print(f"Available keys in collection_data: {collection_data.keys()}")
    
    # Process each document
    for i in range(len(all_ids)):
        print(f"\nEntry {i+1}:")
        print(f"ID: {collection_data['ids'][i]}")
        
        # Handle embeddings - check if the key exists and has values
        if 'embeddings' in collection_data and collection_data['embeddings'] is not None:
            embedding = collection_data['embeddings'][i]
            if embedding is not None and len(embedding) > 0:
                print(f"Embedding (first 5 values): {embedding[:5]}...")
            else:
                print("Embedding: None or empty")
        else:
            print("Embeddings not available in response")
            
        # Handle metadata - check if the key exists
        if 'metadatas' in collection_data and collection_data['metadatas'] is not None:
            metadata = collection_data['metadatas'][i]
            print(f"Metadata: {json.dumps(metadata, indent=2) if metadata else '{}'}")
        else:
            print("Metadata not available in response")
        
        # Handle document text - check if the key exists
        if 'documents' in collection_data and collection_data['documents'] is not None:
            document = collection_data['documents'][i]
            if document:
                print(f"Document preview: {document[:100]}..." if len(document) > 100 else f"Document: {document}")
            else:
                print("Document: None")
        else:
            print("Documents not available in response")
            
        print("-" * 50)
else:
    print("No documents found in the collection")