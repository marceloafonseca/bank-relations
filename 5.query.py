import os
from dotenv import load_dotenv
import openai
from tqdm import tqdm
import chromadb
import tiktoken
import numpy as np
import pickle

# Load environment variables and set the OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize the embedding model (using OpenAIEmbedding from the updated llama-index)
from llama_index.embeddings.openai import OpenAIEmbedding
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",  # Using the new embedding model
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize ChromaDB client and get the collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("rag_collection")

# Load the persistent Chroma vector store
from llama_index.vector_stores.chroma import ChromaVectorStore
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)

# Function to count tokens in a string
def count_tokens(text, model="gpt-4o-2024-08-06"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


# Function to embed the user query
def embed_query(query_text):
    return embed_model.get_text_embedding(query_text)

# Function to find the best guidance row in the guidance CSV file
def find_guidance(query_embedding, embed_model, similarity_threshold=0.7):
    # Load guidance embeddings
    with open("data/guidance_embeddings.pkl", "rb") as f:
        data = pickle.load(f)

    df = data["df"]
    guidance_vectors = data["embeddings"]

    # Compute cosine similarity
    similarities = [np.dot(query_embedding, v) / (np.linalg.norm(query_embedding) * np.linalg.norm(v)) for v in guidance_vectors]

    # Get the index of the most similar question
    best_idx = int(np.argmax(similarities))
    best_similarity = similarities[best_idx]

    # Check if the best similarity is above the threshold
    if best_similarity >= similarity_threshold:
        best_row = df.iloc[best_idx]
        return best_row["Best Practices"], best_row["Red Flags"]
    else:
        return None, None  # Or handle this case as needed

def query_llm(query_text, best_practice, red_flag):
    
    # Get the embedding for the query
    query_embedding = embed_query(query_text)

    # Create a VectorStoreQuery object
    from llama_index.core.vector_stores.types import VectorStoreQuery
    
    vector_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=5 # ADJUST NUMBER OF CONTEXT CHUNKS ACCORDINGLY    
    )
    
    # Query the vector store using the documented method
    results = vector_store.query(vector_query)
    
    # Print the number of retrieved nodes for debugging
    #print(f"\n=== Retrieved {len(results.nodes)} context chunks ===")
    
    # Build a context string that includes source references for each retrieved chunk
    context = ""
    # Track sources for the final answer
    sources = {}
    
    for i, node in enumerate(results.nodes):  # Access nodes from the VectorStoreQueryResult
        # Each result is a Node; try to obtain a 'source' field from metadata if available
        source = node.metadata.get("source", "unknown")
        
        # Extract just the filename from the source path
        if isinstance(source, str):
            # Handle both full paths and just filenames
            filename = os.path.basename(source) if os.path.sep in source else source
        else:
            filename = "unknown"
            
        # Get page number from metadata
        page = node.metadata.get("page", "unknown page")
        
        # Format source info for display
        source_info = f"{filename} (Page: {page})"
        
        # Print each chunk with its index and source info for debugging
        # print(f"\nChunk {i+1}:")
        # print(f"Source: {source_info}")
        # print(f"Text preview: {node.text[:150]}..." if len(node.text) > 150 else f"Text: {node.text}")
        
        # Add the chunk with its source info to the context
        context += f"[Source: {source_info}] {node.text}\n\n"
        
        # Track sources for the final answer
        if filename not in sources:
            sources[filename] = set()
        sources[filename].add(page)
    
    # Handle best_practice and red_flag if they are None
    best_practice_text = best_practice if best_practice is not None else "No relevant best practices found."
    red_flag_text = red_flag if red_flag is not None else "No relevant red flags found."
    
    # Construct the LLM prompt.
    prompt_system = (
        """You are a professional KYC and Compliance expert assisting fintech companies in responding to questionnaires from regulators and external stakeholders.

        Your role is to provide clear, accurate, and concise answers based strictly on internal company policies such as Sanctions, AML, and KYC policies.

        Instructions for answering:
        - Use only the provided **context**, which includes excerpts from internal policy documents.
        - If the context does not contain enough information to answer the question, state clearly that you do not have sufficient information.
        - You are also provided with **Best Practices** and **Red Flags** that are relevant to the user question. These are not drawn from the company policies or context, and should **not** be treated as an evaluation of the content provided.
        - Instead, present the Best Practices and Red Flags as **general industry guidance** for the user to consider when preparing their answer. Do not refer to or compare them with the provided context.
        - If no Best Practices or Red Flags are found, indicate that no relevant guidance was available.
        - If the user question is outside the scope of KYC, AML, Sanctions, or other compliance topics, politely state that the question is out of scope.
        - Respond as if you are a representative of the company answering on its behalf in a formal questionnaire or compliance form.
        - The person reading your response will not have access to the internal policies provided as context, so your answer must be self-contained.
        - Do not reference policy names or say things like “as mentioned in our AML policy.” Instead, present the information as factual statements.
        - Maintain a professional and concise tone at all times. Your answer may be directly copied into a compliance form, so clarity and brevity are essential.

        Your response format should be:

        Suggested answer:
        [Your answer here]
        
        Best practices:
        [Summarized best practices, if any]

        Red flags:
        [Summarized red flags, if any]

        """
    )

    prompt_user = (
        f"""Answer the question below using only the provided context:

        Question: {query_text}

        Context:
        {context}

        Best Practices:
        {best_practice_text}

        Red Flags:
        {red_flag_text}
        """
    )
    
    # Check if the context fits within the model's context window
    # GPT-4o has a context window of approximately 128,000 tokens
    MAX_CONTEXT_TOKENS = 128000
    system_tokens = count_tokens(prompt_system)
    user_tokens = count_tokens(prompt_user)
    total_tokens = system_tokens + user_tokens
    
    if total_tokens > MAX_CONTEXT_TOKENS:
        print(f"⚠️ Warning: Total tokens ({total_tokens}) exceed the model's context window ({MAX_CONTEXT_TOKENS})")
        print(f"System prompt: {system_tokens} tokens")
        print(f"User prompt (including context): {user_tokens} tokens")
        # You could implement context truncation here if needed
    else:
        pass
        # print()
        # print(f"✅ Context size is within limits: {total_tokens}/{MAX_CONTEXT_TOKENS} tokens")
    
    # Call OpenAI's ChatCompletion API (using GPT-4 in this example)
    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",  
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user}
        ],
        temperature=0.0  # Set to 0 for deterministic output; adjust if needed
    )
    answer = response.choices[0].message.content
    
    # Format the sources for display
    sources_text = "\n\nSources:"
    for filename, pages in sources.items():
        pages_list = sorted(list(pages), key=lambda x: str(x))
        sources_text += f"\n- {filename}: pages {', '.join(str(p) for p in pages_list)}"
    
    # Append sources to the answer
    answer_with_sources = answer + sources_text
    
    return answer_with_sources, context

def chat():
    print()
    print("Enter your question (type 'exit' to quit):")
    print()
    while True:
        query_text = input(">> ")
        print()
        if query_text.lower() == "exit":
            break
        
        print("Processing your query...\n")
        query_embedding = embed_query(query_text)
        best_practice, red_flag = find_guidance(query_embedding, embed_model)   
        answer, context_used = query_llm(query_text, best_practice, red_flag)   
        
        print()
        print("==================== Answer ====================")
        print() 
        print(answer)
        print()
        print("================================================")
        print()

if __name__ == "__main__":
    chat()
