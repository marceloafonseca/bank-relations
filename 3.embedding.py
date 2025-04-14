import os
import pickle
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from tqdm import tqdm  # For progress bar

# Load environment variables from .env file
load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Load the previously saved chunks from 2.chunk.py
with open("data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"✅ Loaded {len(chunks)} chunks.")

# Initialize the embedding model
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",  # Using the new embedding model
    api_key=os.getenv("OPENAI_API_KEY"),
    dimensions=1536  # Specify dimensions for the embedding model
)

# Set the embed model as the default for LlamaIndex
Settings.embed_model = embed_model

# Create a list to store embeddings along with each chunk's data
chunk_embeddings = []

# Process chunks with progress bar
for chunk in tqdm(chunks, desc="Computing embeddings"):
    try:
        # Get the text from the chunk
        text = chunk.text if hasattr(chunk, 'text') else str(chunk)
        
        # Compute the embedding vector for the chunk
        embedding = embed_model.get_text_embedding(text)
        
        # Store the result along with the original chunk
        chunk_embeddings.append({
            "chunk": chunk,
            "embedding": embedding
        })
    except Exception as e:
        print(f"Error processing chunk: {e}")
        continue

print("✅ Computed embeddings for all chunks.")

# Save the resulting list (chunks with embeddings)
with open("data/chunks_with_embeddings.pkl", "wb") as f:
    pickle.dump(chunk_embeddings, f)

print("✅ Saved chunks with embeddings to data/chunks_with_embeddings.pkl")