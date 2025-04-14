import pandas as pd
import pickle
from llama_index.embeddings.openai import OpenAIEmbedding
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Load CSV
df = pd.read_csv("guidance/2025_AMLCR_Guidance.csv")

# Initialize embedding model
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small", 
    api_key=os.getenv("OPENAI_API_KEY"),
    dimensions=1536  # Specify dimensions for the embedding model
)

# Compute embeddings for guidance questions
embeddings = []

# Use the correct column name for questions
for question in df["Questions"]:  # Changed from "Question" to "Guidance"
    vector = embed_model.get_text_embedding(question)
    embeddings.append(vector)

# Save for later use
with open("data/guidance_embeddings.pkl", "wb") as f:
    pickle.dump({"df": df, "embeddings": embeddings}, f)

print("✅ Guidance CSV embedded and saved.")