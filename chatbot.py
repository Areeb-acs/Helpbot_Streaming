from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables and API keys
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Load and split PDF document into chunks if not already done
loader = PyPDFLoader("./Onasi_RCM.pdf")
docs = loader.load()

# Use RecursiveCharacterTextSplitter for chunking and FAISS for vector storage
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs[:50])
embeddings = OpenAIEmbeddings()
vectors = FAISS.from_documents(final_documents, embeddings)

# Function to retrieve relevant context chunks
def retrieve_relevant_chunks(question, num_chunks=5):
    # Perform similarity search to get the most relevant chunks for the question
    similar_docs = vectors.similarity_search(question, k=num_chunks)
    return "\n".join([doc.page_content for doc in similar_docs])

# Define the prompt template with placeholders for context and input
prompt_template = ChatPromptTemplate.from_template(
    """
    You are a helpful conversational chatbot. Answer the questions based on the provided context and the previous conversation.
    Please provide the most accurate response. You will first understand what the user is asking, and reply based on that accurately from the context and if not 
    then use common sense.
    
    Like at the start, you need to gather more information from the user, and if the user does not ask a question, then tell him or her to, dont give too long responses.
    If the user asks something you do not know, then just say please give me more information, dont give too long 
    descriptive answers.
    
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Function to generate chatbot response
async def get_chatgroq_response(question, chat_history):
    context = retrieve_relevant_chunks(question)
    formatted_prompt = prompt_template.format(context=context, input=question)

    # Build the flow messages
    flow_messages = [SystemMessage(content="You are a conversational AI assistant.")]
    for entry in chat_history:
        flow_messages.append(HumanMessage(content=entry['question']))
        flow_messages.append(AIMessage(content=entry['answer']))

    flow_messages.append(HumanMessage(content=formatted_prompt))
    # Get the response from the LLM
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
