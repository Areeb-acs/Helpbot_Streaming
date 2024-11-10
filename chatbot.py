# Import necessary libraries
import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage, Document
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, JSONLoader, CSVLoader
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings  # Using OpenAI embeddings for vectorization
import pinecone  # Pinecone client for vector storage
import os
from dotenv import load_dotenv  # Loading environment variables
import pandas as pd
from langchain.schema import Document
from tqdm import tqdm
import multiprocessing
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
# Load environment variables from a .env file (GROQ API key, OpenAI API key, Pinecone API key, and environment)
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Replace HuggingFace embeddings with OpenAI embeddings
embeddings = OpenAIEmbeddings()
# Importing Pinecone-specific classes to set up connection and configuration
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone API key and environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")

# Initializing Pinecone client with API key and specifying environment via ServerlessSpec
# Initialize Pinecone with the correct API key and environment
app = FastAPI()

# Initialize Pinecone client with the specified API key and environment
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
# Define or create an index if it doesn't already exist
index_name = "rcm-app-3"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Embedding dimension must match the OpenAI embedding model dimension (1536 for text-embedding-ada-002)
        metric="euclidean"  # Using cosine similarity as the distance metric
    )

# Retrieve the index for further operations
index = pc.Index(index_name)  # Retrieve or interact with the specified index

# Set up Streamlit user interface with title and introductory text
st.set_page_config(page_title="Onasi RCM Chatbot")
st.title("Onasi RCM Chatbot")
st.write('Welcome, happy to ask any questions you may have!')

# Initialize ChatGroq model using the provided API key
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# Document loading and processing (only if 'final_documents' is not already stored in session state)
if 'final_documents' not in st.session_state:
    # Load PDF file using PyPDFLoader and store it with metadata for identifying document type
    st.session_state.loader_pdf = PyPDFLoader("./Onasi_RCM.pdf")
    st.session_state.docs_pdf = [
        Document(page_content=doc.page_content, metadata={"file_type": "pdf", "source": "Onasi_RCM.pdf"})
        for doc in st.session_state.loader_pdf.load()
    ]

    

    # Load JSON data file and batch-process it as needed
    jq_schema = '.[]'
        
    # Define file paths and batch size
    file_paths = ["./Medical_coding.json", "./Nphies_validation.json"]  # Combine file paths into a single list
    batch_size = 400

    # Initialize list for Document objects
    combined_json_documents = []

    # Process each JSON file using pandas
    for file_path in file_paths:
        try:
            data = pd.read_json(file_path)
            for _, row in data.iterrows():
                combined_json_documents.append(
                    Document(
                        page_content=str(row.to_dict()),
                        metadata={"file_type": "json", "source": file_path.split('/')[-1]}  # Add source dynamically
                    )
                )
        except ValueError as e:
            st.error(f"Error processing JSON file {file_path}: {e}")

    # Combine PDF and JSON documents
    st.session_state.docs = st.session_state.docs_pdf +  combined_json_documents

                
    
    # Split documents into chunks for embedding, using specified chunk size and overlap
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

    # Initialize OpenAI embeddings
    # st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.pinecone_index = Pinecone(index_name=index_name)

    chunks = [doc.page_content for doc in st.session_state.final_documents]
    embedding_batches = []

    # Generate embeddings using threads
    def generate_embeddings_threading(chunks, embeddings, batch_size=200):
        embeddings_list = []

        def embed_chunk(chunk):
            return embeddings.embed_query(chunk)

        with ThreadPoolExecutor() as executor:
            for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
                batch = chunks[i:i + batch_size]
                embeddings_list.extend(executor.map(embed_chunk, batch))

        return embeddings_list

    # Call the function
    embedding_batches = generate_embeddings_threading(chunks, embeddings)
    # Prepare documents for insertion into Pinecone
    # Prepare documents for insertion into Pinecone
    # Prepare documents for Pinecone upsert

    docs_to_index = [
        {
            "id": doc.metadata.get("id", str(hash(doc.page_content))),
            "values": embedding_batches[idx],
            "metadata": {"page_content": doc.page_content, **doc.metadata}
        }
        for idx, doc in enumerate(st.session_state.final_documents)
    ]

    # Batch upsert (in case of a large number of vectors, batch them into smaller chunks)
    batch_size = 200  # Choose an appropriate batch size
    for i in range(0, len(docs_to_index), batch_size):
        batch = docs_to_index[i:i + batch_size]
        index.upsert(vectors=batch)  # Use the index object for upserting

# Function to retrieve relevant chunks of documents based on a user query, with an optional filter for document type
def retrieve_relevant_chunks(question, num_chunks=10, file_type=None):
    import re
    
    # Extract numbers and Rule IDs from the query
    numbers_in_query = re.findall(r'\d+', question)  # Extract numbers
    rule_id_match = re.search(r'\bBV-\d+\b', question)  # Match Rule IDs like "BV-00027"

    # Check for exact match search based on Rule ID
    if rule_id_match:
        rule_id_to_search = rule_id_match.group(0)  # Extract matched Rule ID
        matches = [
            doc.page_content
            for doc in st.session_state.final_documents
            if rule_id_to_search in doc.page_content  # Check if the Rule ID exists in the page content
        ]
        if matches:
            return "\n".join(matches[:num_chunks])  # Return top 'num_chunks' matches

    # Check for exact match search based on numbers
    if numbers_in_query:
        number_to_search = numbers_in_query[0]  # Assume we're searching for the first number found
        matches = [
            doc.page_content
            for doc in st.session_state.final_documents
            if number_to_search in doc.page_content  # Check if the number exists in the page content
        ]
        if matches:
            return "\n".join(matches[:num_chunks])  # Return top 'num_chunks' matches

    # Fall back to embedding-based search for more complex queries
    question_embedding = embeddings.embed_query(question)
    filter_dict = {"file_type": file_type} if file_type else {}
    similar_docs = index.query(
        vector=question_embedding,
        top_k=num_chunks,
        filter=filter_dict,
        include_values=True,
        include_metadata=True,
    )
    return "\n".join([match["metadata"].get("page_content", "") for match in similar_docs["matches"]])


# Define the template for generating prompts with context and input placeholders
prompt_template = ChatPromptTemplate.from_template(
    """
    You are friendly conversational chatbot that remembers names, answer the question based on the provided context and general knowledge.
    Please provide the most accurate response. You will first understand what the user is asking, and reply in plain english for doctor to understand.
    

    
    Intructions:
    
    Instructions:
    1. Always read the context and provide responsefor the doctor in simple english and explain it to the doctor using your knowledge.
    2. Based on the context, understand the context and output your response in plain english for doctor to understand.
    3. Do not please say "in simple english", always say in other words or to put it simply, use phrases like these.

    <context>
    {context}
    <context>
    
    Question: {input}
    """
)



# prompt_template = ChatPromptTemplate.from_template(
#     """
#     You are friendly conversational chatbot that remembers names, answer the question based on the provided context and general knowledge.
#     Please provide the most accurate response. You will first understand what the user is asking, and reply based on that accurately from the context.
    
#     You are an expert in knowing about the RCM application, medical coding and nphies validation codes.
#     You are an expert in reading .json data, so you know how to read information from json files.
    
#     If what the user asks does not exist in knowledge base, like code values or anything, just say you do not know, do not make up things.
    
#     Intructions:
#     1. When you respond, do not show the context you are searching, just give a medium answer, summarize in bullet points or whatever appropriate, to the point, if the user ask what is the code value, just answer with number etc.
#     2. If the answer is not in the given context, mention I cannot find it, out of my knowledge base.
#     3. If you cannot find any relevant information say for example the code value is not present, just say I cannot find it, out of my knowledge base
#     4. You can read json files easily.
#     5. Wherever appropriate, give answer in bullet points or table format or couple of sentences.

#     <context>
#     {context}
#     <context>
    
#     Question: {input}
#     """
# )

# Function to generate chatbot response

# Function to generate a response based on the user query and context
async def get_chatgroq_response(question, file_type=None):
    # Retrieve relevant context chunks based on question and optional file_type filter
    context = retrieve_relevant_chunks(question, file_type=file_type)
    print("==============================================")
    print(context)
    # Format the prompt with context and question
    formatted_prompt = prompt_template.format(context=context, input=question)
    
    # Create a conversational flow with message history for context
    flow_messages = [SystemMessage(content="You are a conversational AI assistant.")]
    # Initialize session state variables
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []  # Initialize chat history as an empty list

    for entry in st.session_state['chat_history']:
        flow_messages.append(HumanMessage(content=entry['question']))
        flow_messages.append(AIMessage(content=entry['answer']))
    
    # Add the formatted prompt as the user's current question
    flow_messages.append(HumanMessage(content=formatted_prompt))
    
    # Get the response from the ChatGroq model
    answer = llm(flow_messages)
    
    return answer.content

@app.get("/")
async def get_root():
    return {"message": "Welcome to the RCM Chatbot!"}


@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    question = data.get("question")
    chat_history = data.get("chat_history", [])
    
    async def response_generator():
        # Stream each part of the response as it's ready
        response = await get_chatgroq_response(question, chat_history)
        for chunk in response:
            yield chunk
    
    return StreamingResponse(response_generator(), media_type="text/plain")
