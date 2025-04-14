# Fix for ChromaDB SQLite version issue on Streamlit Cloud
import os
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Continue with your regular imports
import streamlit as st
import os
from dotenv import load_dotenv
import openai
import numpy as np
import pickle
import chromadb
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery
import tiktoken

# Page configuration
st.set_page_config(
    page_title="Compliance Assistant",
    page_icon="📁",
    layout="centered"
)

# Initialize session state for authentication and chat history
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to authenticate user
def authenticate(password):
    if password == "coffeeisgood":
        st.session_state.authenticated = True
        return True
    return False

# Authentication page
if not st.session_state.authenticated:
    st.title("Banking Relations Assistant")
    st.markdown("Please enter the password to access the application.")
    
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")
    
    if login_button:
        if authenticate(password):
            st.session_state.authenticated = True
            st.success("Authentication successful!")
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")
    
    # Stop execution here if not authenticated
    st.stop()

# If we get here, the user is authenticated
# Load environment variables and set the OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OPENAI_API_KEY not found in environment variables")
    st.stop()

# Initialize the embedding model
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize ChromaDB client and get the collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("rag_collection")

# Load the persistent Chroma vector store
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
    vector_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=5 # ADJUST NUMBER OF CONTEXT CHUNKS ACCORDINGLY    
    )
    
    # Query the vector store using the documented method
    results = vector_store.query(vector_query)
    
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
        - Do not reference policy names or say things like "as mentioned in our AML policy." Instead, present the information as factual statements.
        - Maintain a professional and concise tone at all times. Your answer may be directly copied into a compliance form, so clarity and brevity are essential.
        - You might receive more than one question at once, process and answer which question separately.

        Your response format should be:

        **Suggested answer:**
        [Your answer here]
        
        **Best practices when answering this question:**
        [Summarized best practices in bullet points instructing the user on how to answer the question, if any]

        **Red flags when answering this question:**
        [Summarized red flags in bullet points instructing the user on how to answer the question, if any]
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
    MAX_CONTEXT_TOKENS = 128000
    system_tokens = count_tokens(prompt_system)
    user_tokens = count_tokens(prompt_user)
    total_tokens = system_tokens + user_tokens
    
    if total_tokens > MAX_CONTEXT_TOKENS:
        print(f"⚠️ Warning: Total tokens ({total_tokens}) exceed the model's context window ({MAX_CONTEXT_TOKENS})")
        # You could implement context truncation here if needed
    
    # Call OpenAI's ChatCompletion API
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
    sources_text = "\n\n**Sources:**"
    for filename, pages in sources.items():
        pages_list = sorted(list(pages), key=lambda x: str(x))
        sources_text += f"\n- {filename}: pages {', '.join(str(p) for p in pages_list)}"
    
    # Append sources to the answer
    answer_with_sources = answer + sources_text
    
    return answer_with_sources, context

# App title
st.title("Banking Relations Assistant")
st.markdown("Ask questions about KYC and Compliance and get accurate answers based on your internal documents")
st.markdown("**Instructions:** Enter one question at a time")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a container for the input area
input_container = st.container()

# Create two columns in the input area
col1, col2 = input_container.columns([1, 5])

# Add the clear button to the first column
with col1:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# User input in the second column
with col2:
    # We can't directly place the chat_input in a column, so we'll use a placeholder
    # The actual chat_input will be outside the columns
    st.write("") # Add some space

# User input (this has to be outside the columns due to Streamlit limitations)
if prompt := st.chat_input("Ask a question about your internal documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response in chat
    with st.chat_message("assistant"):
        with st.spinner("Processing your query..."):
            try:
                # Get query embedding
                query_embedding = embed_query(prompt)
                
                # Find guidance
                best_practice, red_flag = find_guidance(query_embedding, embed_model)
                
                # Get answer from LLM
                answer, _ = query_llm(prompt, best_practice, red_flag)
                
                # Display the answer
                st.markdown(answer)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {str(e)}")