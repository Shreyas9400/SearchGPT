import os
import json
import re
import gradio as gr
import requests
from duckduckgo_search import DDGS
from typing import List, Dict
from pydantic import BaseModel, Field
from tempfile import NamedTemporaryFile
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_parse import LlamaParse
from huggingface_hub import InferenceClient
import inspect
import logging
import shutil


# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables and configurations
huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
llama_cloud_api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
ACCOUNT_ID = os.environ.get("CLOUDFARE_ACCOUNT_ID")
API_TOKEN = os.environ.get("CLOUDFLARE_AUTH_TOKEN")
API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/a17f03e0f049ccae0c15cdcf3b9737ce/ai/run/"

print(f"ACCOUNT_ID: {ACCOUNT_ID}")
print(f"CLOUDFLARE_AUTH_TOKEN: {API_TOKEN[:5]}..." if API_TOKEN else "Not set")

MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "@cf/meta/llama-3.1-8b-instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "duckduckgo/gpt-4o-mini",
    "duckduckgo/claude-3-haiku",
    "duckduckgo/llama-3.1-70b",
    "duckduckgo/mixtral-8x7b"
]

# Initialize LlamaParse
llama_parser = LlamaParse(
    api_key=llama_cloud_api_key,
    result_type="markdown",
    num_workers=4,
    verbose=True,
    language="en",
)

def load_document(file: NamedTemporaryFile, parser: str = "llamaparse") -> List[Document]:
    """Loads and splits the document into pages."""
    if parser == "pypdf":
        loader = PyPDFLoader(file.name)
        return loader.load_and_split()
    elif parser == "llamaparse":
        try:
            documents = llama_parser.load_data(file.name)
            return [Document(page_content=doc.text, metadata={"source": file.name}) for doc in documents]
        except Exception as e:
            print(f"Error using Llama Parse: {str(e)}")
            print("Falling back to PyPDF parser")
            loader = PyPDFLoader(file.name)
            return loader.load_and_split()
    else:
        raise ValueError("Invalid parser specified. Use 'pypdf' or 'llamaparse'.")

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="avsolatorio/GIST-Embedding-v0")

# Add this at the beginning of your script, after imports
DOCUMENTS_FILE = "uploaded_documents.json"

def load_documents():
    if os.path.exists(DOCUMENTS_FILE):
        with open(DOCUMENTS_FILE, "r") as f:
            return json.load(f)
    return []

def save_documents(documents):
    with open(DOCUMENTS_FILE, "w") as f:
        json.dump(documents, f)

# Replace the global uploaded_documents with this
uploaded_documents = load_documents()

# Modify the update_vectors function
def update_vectors(files, parser):
    global uploaded_documents
    logging.info(f"Entering update_vectors with {len(files)} files and parser: {parser}")
    
    if not files:
        logging.warning("No files provided for update_vectors")
        return "Please upload at least one PDF file.", display_documents()
    
    embed = get_embeddings()
    total_chunks = 0
    
    all_data = []
    for file in files:
        logging.info(f"Processing file: {file.name}")
        try:
            data = load_document(file, parser)
            if not data:
                logging.warning(f"No chunks loaded from {file.name}")
                continue
            logging.info(f"Loaded {len(data)} chunks from {file.name}")
            all_data.extend(data)
            total_chunks += len(data)
            if not any(doc["name"] == file.name for doc in uploaded_documents):
                uploaded_documents.append({"name": file.name, "selected": True})
                logging.info(f"Added new document to uploaded_documents: {file.name}")
            else:
                logging.info(f"Document already exists in uploaded_documents: {file.name}")
        except Exception as e:
            logging.error(f"Error processing file {file.name}: {str(e)}")
    
    logging.info(f"Total chunks processed: {total_chunks}")
    
    if not all_data:
        logging.warning("No valid data extracted from uploaded files")
        return "No valid data could be extracted from the uploaded files. Please check the file contents and try again.", display_documents()
    
    try:
        if os.path.exists("faiss_database"):
            logging.info("Updating existing FAISS database")
            database = FAISS.load_local("faiss_database", embed, allow_dangerous_deserialization=True)
            database.add_documents(all_data)
        else:
            logging.info("Creating new FAISS database")
            database = FAISS.from_documents(all_data, embed)
        
        database.save_local("faiss_database")
        logging.info("FAISS database saved")
    except Exception as e:
        logging.error(f"Error updating FAISS database: {str(e)}")
        return f"Error updating vector store: {str(e)}", display_documents()

    # Save the updated list of documents
    save_documents(uploaded_documents)

    # Return a tuple with the status message and the updated document list
    return f"Vector store updated successfully. Processed {total_chunks} chunks from {len(files)} files using {parser}.", display_documents()


def delete_documents(selected_docs):
    global uploaded_documents
    
    if not selected_docs:
        return "No documents selected for deletion.", display_documents()
    
    embed = get_embeddings()
    database = FAISS.load_local("faiss_database", embed, allow_dangerous_deserialization=True)
    
    deleted_docs = []
    docs_to_keep = []
    for doc in database.docstore._dict.values():
        if doc.metadata.get("source") not in selected_docs:
            docs_to_keep.append(doc)
        else:
            deleted_docs.append(doc.metadata.get("source", "Unknown"))
    
    # Print debugging information
    logging.info(f"Total documents before deletion: {len(database.docstore._dict)}")
    logging.info(f"Documents to keep: {len(docs_to_keep)}")
    logging.info(f"Documents to delete: {len(deleted_docs)}")
    
    if not docs_to_keep:
        # If all documents are deleted, remove the FAISS database directory
        if os.path.exists("faiss_database"):
            shutil.rmtree("faiss_database")
        logging.info("All documents deleted. Removed FAISS database directory.")
    else:
        # Create new FAISS index with remaining documents
        new_database = FAISS.from_documents(docs_to_keep, embed)
        new_database.save_local("faiss_database")
        logging.info(f"Created new FAISS index with {len(docs_to_keep)} documents.")
    
    # Update uploaded_documents list
    uploaded_documents = [doc for doc in uploaded_documents if doc["name"] not in deleted_docs]
    save_documents(uploaded_documents)
    
    return f"Deleted documents: {', '.join(deleted_docs)}", display_documents()

def generate_chunked_response(prompt, model, max_tokens=10000, num_calls=3, temperature=0.2, should_stop=False):
    print(f"Starting generate_chunked_response with {num_calls} calls")
    full_response = ""
    messages = [{"role": "user", "content": prompt}]
    
    if model == "@cf/meta/llama-3.1-8b-instruct":
        # Cloudflare API
        for i in range(num_calls):
            print(f"Starting Cloudflare API call {i+1}")
            if should_stop:
                print("Stop clicked, breaking loop")
                break
            try:
                response = requests.post(
                    f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/meta/llama-3.1-8b-instruct",
                    headers={"Authorization": f"Bearer {API_TOKEN}"},
                    json={
                        "stream": true,
                        "messages": [
                            {"role": "system", "content": "You are a friendly assistant"},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    },
                    stream=true
                )
                
                for line in response.iter_lines():
                    if should_stop:
                        print("Stop clicked during streaming, breaking")
                        break
                    if line:
                        try:
                            json_data = json.loads(line.decode('utf-8').split('data: ')[1])
                            chunk = json_data['response']
                            full_response += chunk
                        except json.JSONDecodeError:
                            continue
                print(f"Cloudflare API call {i+1} completed")
            except Exception as e:
                print(f"Error in generating response from Cloudflare: {str(e)}")
    else:
        # Original Hugging Face API logic
        client = InferenceClient(model, token=huggingface_token)
        
        for i in range(num_calls):
            print(f"Starting Hugging Face API call {i+1}")
            if should_stop:
                print("Stop clicked, breaking loop")
                break
            try:
                for message in client.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                ):
                    if should_stop:
                        print("Stop clicked during streaming, breaking")
                        break
                    if message.choices and message.choices[0].delta and message.choices[0].delta.content:
                        chunk = message.choices[0].delta.content
                        full_response += chunk
                print(f"Hugging Face API call {i+1} completed")
            except Exception as e:
                print(f"Error in generating response from Hugging Face: {str(e)}")
    
    # Clean up the response
    clean_response = re.sub(r'<s>\[INST\].*?\[/INST\]\s*', '', full_response, flags=re.DOTALL)
    clean_response = clean_response.replace("Using the following context:", "").strip()
    clean_response = clean_response.replace("Using the following context from the PDF documents:", "").strip()
    
    # Remove duplicate paragraphs and sentences
    paragraphs = clean_response.split('\n\n')
    unique_paragraphs = []
    for paragraph in paragraphs:
        if paragraph not in unique_paragraphs:
            sentences = paragraph.split('. ')
            unique_sentences = []
            for sentence in sentences:
                if sentence not in unique_sentences:
                    unique_sentences.append(sentence)
            unique_paragraphs.append('. '.join(unique_sentences))
    
    final_response = '\n\n'.join(unique_paragraphs)
    
    print(f"Final clean response: {final_response[:100]}...")
    return final_response

def chatbot_interface(message, history, model, temperature, num_calls):
    if not message.strip():
        return "", history

    history = history + [(message, "")]

    try:
        for response in respond(message, history, model, temperature, num_calls):
            history[-1] = (message, response)
            yield history
    except gr.CancelledError:
        yield history
    except Exception as e:
        logging.error(f"Unexpected error in chatbot_interface: {str(e)}")
        history[-1] = (message, f"An unexpected error occurred: {str(e)}")
        yield history

def retry_last_response(history, model, temperature, num_calls):
    if not history:
        return history
    
    last_user_msg = history[-1][0]
    history = history[:-1]  # Remove the last response
    
    return chatbot_interface(last_user_msg, history, model, temperature, num_calls)

def truncate_context(context, max_length=16000):
    """Truncate the context to a maximum length."""
    if len(context) <= max_length:
        return context
    return context[:max_length] + "..."

def get_response_from_duckduckgo(query, model, context, num_calls=1, temperature=0.2):
    logging.info(f"Using DuckDuckGo chat with model: {model}")
    ddg_model = model.split('/')[-1]  # Extract the model name from the full string
    
    # Truncate the context to avoid exceeding input limits
    truncated_context = truncate_context(context)
    
    full_response = ""
    for _ in range(num_calls):
        try:
            # Include truncated context in the query
            contextualized_query = f"Using the following context:\n{truncated_context}\n\nUser question: {query}"
            results = DDGS().chat(contextualized_query, model=ddg_model)
            full_response += results + "\n"
            logging.info(f"DuckDuckGo API response received. Length: {len(results)}")
        except Exception as e:
            logging.error(f"Error in generating response from DuckDuckGo: {str(e)}")
            yield f"An error occurred with the {model} model: {str(e)}. Please try again."
            return

    yield full_response.strip()

class ConversationManager:
    def __init__(self):
        self.history = []
        self.current_context = None

    def add_interaction(self, query, response):
        self.history.append((query, response))
        self.current_context = f"Previous query: {query}\nPrevious response summary: {response[:200]}..."

    def get_context(self):
        return self.current_context

conversation_manager = ConversationManager()

def get_web_search_results(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    try:
        results = list(DDGS().text(query, max_results=max_results))
        if not results:
            print(f"No results found for query: {query}")
        return results
    except Exception as e:
        print(f"An error occurred during web search: {str(e)}")
        return [{"error": f"An error occurred during web search: {str(e)}"}]

def rephrase_query(original_query: str, conversation_manager: ConversationManager) -> str:
    context = conversation_manager.get_context()
    if context:
        prompt = f"""You are a highly intelligent conversational chatbot. Your task is to analyze the given context and new query, then decide whether to rephrase the query with or without incorporating the context. Follow these steps:

        1. Determine if the new query is a continuation of the previous conversation or an entirely new topic.
        2. If it's a continuation, rephrase the query by incorporating relevant information from the context to make it more specific and contextual.
        3. If it's a new topic, rephrase the query to make it more appropriate for a web search, focusing on clarity and accuracy without using the previous context.
        4. Provide ONLY the rephrased query without any additional explanation or reasoning.
        
        Context: {context}
        
        New query: {original_query}
        
        Rephrased query:"""
        response = DDGS().chat(prompt, model="llama-3.1-70b")
        rephrased_query = response.split('\n')[0].strip()
        return rephrased_query
    return original_query

def summarize_web_results(query: str, search_results: List[Dict[str, str]], conversation_manager: ConversationManager) -> str:
    try:
        context = conversation_manager.get_context()
        search_context = "\n\n".join([f"Title: {result['title']}\nContent: {result['body']}" for result in search_results])

        prompt = f"""You are a highly intelligent & expert analyst and your job is to skillfully articulate the web search results about '{query}' and considering the context: {context}, 
        You have to create a comprehensive news summary FOCUSING on the context provided to you. 
        Include key facts, relevant statistics, and expert opinions if available. 
        Ensure the article is well-structured with an introduction, main body, and conclusion, IF NECESSARY. 
        Address the query in the context of the ongoing conversation IF APPLICABLE.
        Cite sources directly within the generated text and not at the end of the generated text, integrating URLs where appropriate to support the information provided:

        {search_context}

        Article:"""

        summary = DDGS().chat(prompt, model="llama-3.1-70b")
        return summary
    except Exception as e:
        return f"An error occurred during summarization: {str(e)}"

# Modify the existing respond function to handle both PDF and web search
def respond(message, history, model, temperature, num_calls, use_web_search, selected_docs):
    logging.info(f"User Query: {message}")
    logging.info(f"Model Used: {model}")
    logging.info(f"Selected Documents: {selected_docs}")
    logging.info(f"Use Web Search: {use_web_search}")

    if use_web_search:
        original_query = message
        rephrased_query = rephrase_query(message, conversation_manager)
        logging.info(f"Original query: {original_query}")
        logging.info(f"Rephrased query: {rephrased_query}")

        final_summary = ""
        for _ in range(num_calls):
            search_results = get_web_search_results(rephrased_query)
            if not search_results:
                final_summary += f"No search results found for the query: {rephrased_query}\n\n"
            elif "error" in search_results[0]:
                final_summary += search_results[0]["error"] + "\n\n"
            else:
                summary = summarize_web_results(rephrased_query, search_results, conversation_manager)
                final_summary += summary + "\n\n"

        if final_summary:
            conversation_manager.add_interaction(original_query, final_summary)
            yield final_summary
        else:
            yield "Unable to generate a response. Please try a different query."
    else:
        # Existing PDF search logic
        try:
            embed = get_embeddings()
            if os.path.exists("faiss_database"):
                database = FAISS.load_local("faiss_database", embed, allow_dangerous_deserialization=True)
                retriever = database.as_retriever(search_kwargs={"k": 20})
                
                all_relevant_docs = retriever.get_relevant_documents(message)
                relevant_docs = [doc for doc in all_relevant_docs if doc.metadata["source"] in selected_docs]
                
                if not relevant_docs:
                    yield "No relevant information found in the selected documents. Please try selecting different documents or rephrasing your query."
                    return
    
                context_str = "\n".join([doc.page_content for doc in relevant_docs])
                logging.info(f"Context length: {len(context_str)}")
            else:
                context_str = "No documents available."
                yield "No documents available. Please upload PDF documents to answer questions."
                return
            
            if model.startswith("duckduckgo/"):
                # Use DuckDuckGo chat with context
                for partial_response in get_response_from_duckduckgo(message, model, context_str, num_calls, temperature):
                    yield partial_response
            elif model == "@cf/meta/llama-3.1-8b-instruct":
                # Use Cloudflare API
                for partial_response in get_response_from_cloudflare(prompt="", context=context_str, query=message, num_calls=num_calls, temperature=temperature, search_type="pdf"):
                    yield partial_response
            else:
                # Use Hugging Face API
                for partial_response in get_response_from_pdf(message, model, selected_docs, num_calls=num_calls, temperature=temperature):
                    yield partial_response
        except Exception as e:
            logging.error(f"Error with {model}: {str(e)}")
            if "microsoft/Phi-3-mini-4k-instruct" in model:
                logging.info("Falling back to Mistral model due to Phi-3 error")
                fallback_model = "mistralai/Mistral-7B-Instruct-v0.3"
                yield from respond(message, history, fallback_model, temperature, num_calls, selected_docs)
            else:
                yield f"An error occurred with the {model} model: {str(e)}. Please try again or select a different model."
        
logging.basicConfig(level=logging.DEBUG)

def get_response_from_cloudflare(prompt, context, query, num_calls=3, temperature=0.2, search_type="pdf"):
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    model = "@cf/meta/llama-3.1-8b-instruct"

    if search_type == "pdf":
        instruction = f"""Using the following context from the PDF documents:
{context}
Write a detailed and complete response that answers the following user question: '{query}'"""
    else:  # web search
        instruction = f"""Using the following context:
{context}
Write a detailed and complete research document that fulfills the following user request: '{query}'
After writing the document, please provide a list of sources used in your response."""

    inputs = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": query}
    ]

    payload = {
        "messages": inputs,
        "stream": True,
        "temperature": temperature,
        "max_tokens": 32000
    }

    full_response = ""
    for i in range(num_calls):
        try:
            with requests.post(f"{API_BASE_URL}{model}", headers=headers, json=payload, stream=True) as response:
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            try:
                                json_response = json.loads(line.decode('utf-8').split('data: ')[1])
                                if 'response' in json_response:
                                    chunk = json_response['response']
                                    full_response += chunk
                                    yield full_response
                            except (json.JSONDecodeError, IndexError) as e:
                                logging.error(f"Error parsing streaming response: {str(e)}")
                                continue
                else:
                    logging.error(f"HTTP Error: {response.status_code}, Response: {response.text}")
                    yield f"I apologize, but I encountered an HTTP error: {response.status_code}. Please try again later."
        except Exception as e:
            logging.error(f"Error in generating response from Cloudflare: {str(e)}")
            yield f"I apologize, but an error occurred: {str(e)}. Please try again later."
    
    if not full_response:
        yield "I apologize, but I couldn't generate a response at this time. Please try again later."

def create_web_search_vectors(search_results):
    embed = get_embeddings()
    
    documents = []
    for result in search_results:
        if 'body' in result:
            content = f"{result['title']}\n{result['body']}\nSource: {result['href']}"
            documents.append(Document(page_content=content, metadata={"source": result['href']}))
    
    return FAISS.from_documents(documents, embed)

def get_response_from_pdf(query, model, selected_docs, num_calls=3, temperature=0.2):
    logging.info(f"Entering get_response_from_pdf with query: {query}, model: {model}, selected_docs: {selected_docs}")
    
    embed = get_embeddings()
    if os.path.exists("faiss_database"):
        logging.info("Loading FAISS database")
        database = FAISS.load_local("faiss_database", embed, allow_dangerous_deserialization=True)
    else:
        logging.warning("No FAISS database found")
        yield "No documents available. Please upload PDF documents to answer questions."
        return

    # Pre-filter the documents
    filtered_docs = []
    for doc_id, doc in database.docstore._dict.items():
        if isinstance(doc, Document) and doc.metadata.get("source") in selected_docs:
            filtered_docs.append(doc)
    
    logging.info(f"Number of documents after pre-filtering: {len(filtered_docs)}")

    if not filtered_docs:
        logging.warning(f"No documents found for the selected sources: {selected_docs}")
        yield "No relevant information found in the selected documents. Please try selecting different documents or rephrasing your query."
        return

    # Create a new FAISS index with only the selected documents
    filtered_db = FAISS.from_documents(filtered_docs, embed)
    
    retriever = filtered_db.as_retriever(search_kwargs={"k": 10})
    logging.info(f"Retrieving relevant documents for query: {query}")
    relevant_docs = retriever.get_relevant_documents(query)
    logging.info(f"Number of relevant documents retrieved: {len(relevant_docs)}")

    for doc in relevant_docs:
        logging.info(f"Document source: {doc.metadata['source']}")
        logging.info(f"Document content preview: {doc.page_content[:100]}...")  # Log first 100 characters of each document

    context_str = "\n".join([doc.page_content for doc in relevant_docs])
    logging.info(f"Total context length: {len(context_str)}")

    if model == "@cf/meta/llama-3.1-8b-instruct":
        logging.info("Using Cloudflare API")
        # Use Cloudflare API with the retrieved context
        for response in get_response_from_cloudflare(prompt="", context=context_str, query=query, num_calls=num_calls, temperature=temperature, search_type="pdf"):
            yield response
    else:
        logging.info("Using Hugging Face API")
        # Use Hugging Face API
        prompt = f"""Using the following context from the PDF documents:
{context_str}
Write a detailed and complete response that answers the following user question: '{query}'"""
        
        client = InferenceClient(model, token=huggingface_token)
        
        response = ""
        for i in range(num_calls):
            logging.info(f"API call {i+1}/{num_calls}")
            for message in client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20000,
                temperature=temperature,
                stream=True,
            ):
                if message.choices and message.choices[0].delta and message.choices[0].delta.content:
                    chunk = message.choices[0].delta.content
                    response += chunk
                    yield response  # Yield partial response
        
        logging.info("Finished generating response")

def vote(data: gr.LikeData):
    if data.liked:
        print(f"You upvoted this response: {data.value}")
    else:
        print(f"You downvoted this response: {data.value}")

css = """
/* Fine-tune chatbox size */
.chatbot-container {
    height: 600px !important;
    width: 100% !important;
}
.chatbot-container > div {
    height: 100%;
    width: 100%;
}
"""

uploaded_documents = []

def display_documents():
    return gr.CheckboxGroup(
        choices=[doc["name"] for doc in uploaded_documents],
        value=[doc["name"] for doc in uploaded_documents if doc["selected"]],
        label="Select documents to query or delete"
    )

def initial_conversation():
    return [
        (None, "Welcome! I'm your AI assistant for web search and PDF analysis. Here's how you can use me:\n\n"
                "1. Set the toggle for Web Search and PDF Search from the checkbox in Additional Inputs drop down window\n"
                "2. Use web search to find information\n"
                "3. Upload the documents and ask questions about uploaded PDF documents by selecting your respective document\n"
                "4. For any queries feel free to reach out @desai.shreyas94@gmail.com or discord - shreyas094\n\n"
                "To get started, upload some PDFs or ask me a question!")
    ]
# Add this new function
def refresh_documents():
    global uploaded_documents
    uploaded_documents = load_documents()
    return display_documents()

# Define the checkbox outside the demo block
document_selector = gr.CheckboxGroup(label="Select documents to query")

use_web_search = gr.Checkbox(label="Use Web Search", value=False)

custom_placeholder = "Ask a question (Note: You can toggle between Web Search and PDF Chat in Additional Inputs below)"

# Update the demo interface
# Update the Gradio interface
demo = gr.ChatInterface(
    respond,
    additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=True, render=False),
    additional_inputs=[
        gr.Dropdown(choices=MODELS, label="Select Model", value=MODELS[3]),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.2, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Number of API Calls"),
        gr.Checkbox(label="Use Web Search", value=True),
        gr.CheckboxGroup(label="Select documents to query")        
    ],
    title="AI-powered PDF Chat and Web Search Assistant",
    description="Chat with your PDFs or use web search to answer questions.",
    theme=gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="amber",
        neutral_hue="gray",
        font=[gr.themes.GoogleFont("Exo"), "ui-sans-serif", "system-ui", "sans-serif"]
    ).set(
        body_background_fill_dark="#0c0505",
        block_background_fill_dark="#0c0505",
        block_border_width="1px",
        block_title_background_fill_dark="#1b0f0f",
        input_background_fill_dark="#140b0b",
        button_secondary_background_fill_dark="#140b0b",
        border_color_accent_dark="#1b0f0f",
        border_color_primary_dark="#1b0f0f",
        background_fill_secondary_dark="#0c0505",
        color_accent_soft_dark="transparent",
        code_background_fill_dark="#140b0b"
    ),
    css=css,
    examples=[
        ["Tell me about the contents of the uploaded PDFs."],
        ["What are the main topics discussed in the documents?"],
        ["Can you summarize the key points from the PDFs?"],
        ["What's the latest news about artificial intelligence?"]
    ],
    cache_examples=False,
    analytics_enabled=False,
    textbox=gr.Textbox(placeholder="Ask a question about the uploaded PDFs or any topic", container=False, scale=7),
    chatbot = gr.Chatbot(  
        show_copy_button=True,
        likeable=True,
        layout="bubble",
        height=400,
        value=initial_conversation()
    )
)

# Add file upload functionality
# Add file upload functionality
with demo:
    gr.Markdown("## Upload and Manage PDF Documents")
    with gr.Row():
        file_input = gr.Files(label="Upload your PDF documents", file_types=[".pdf"])
        parser_dropdown = gr.Dropdown(choices=["pypdf", "llamaparse"], label="Select PDF Parser", value="llamaparse")
        update_button = gr.Button("Upload Document")
        refresh_button = gr.Button("Refresh Document List")
    
    update_output = gr.Textbox(label="Update Status")
    delete_button = gr.Button("Delete Selected Documents")
    
    # Update both the output text and the document selector
    update_button.click(
        update_vectors, 
        inputs=[file_input, parser_dropdown], 
        outputs=[update_output, demo.additional_inputs[-1]]  # Use the CheckboxGroup from additional_inputs
    )
    
    # Add the refresh button functionality
    refresh_button.click(
        refresh_documents, 
        inputs=[], 
        outputs=[demo.additional_inputs[-1]]  # Use the CheckboxGroup from additional_inputs
    )
    
    # Add the delete button functionality
    delete_button.click(
        delete_documents,
        inputs=[demo.additional_inputs[-1]],  # Use the CheckboxGroup from additional_inputs
        outputs=[update_output, demo.additional_inputs[-1]]
    )

    gr.Markdown(
    """
    ## How to use
    1. Upload PDF documents using the file input at the top.
    2. Select the PDF parser (pypdf or llamaparse) and click "Upload Document" to update the vector store.
    3. Select the documents you want to query using the checkboxes.
    4. Ask questions in the chat interface. 
    5. Toggle "Use Web Search" to switch between PDF chat and web search.
    6. Adjust Temperature and Number of API Calls to fine-tune the response generation.
    7. Use the provided examples or ask your own questions.
    """
    )

if __name__ == "__main__":
    demo.launch(share=True)
