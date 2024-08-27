1. Clone the repository: git clone https://github.com/Shreyas9400/SearchGPT.git
cd SearchGPT

2. Create a virtual environment: python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

3. Install the required packages: pip install -r requirements.txt

4. Set up environment variables:
Create a `.env` file in the project root and add the following:
HUGGINGFACE_TOKEN=your_huggingface_token
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
CLOUDFARE_ACCOUNT_ID=your_cloudflare_account_id
CLOUDFLARE_AUTH_TOKEN=your_cloudflare_auth_token

5. Run the application: python main.py


# AI-powered Web Search and PDF Chat Assistant

This project combines the power of large language models with web search capabilities and PDF document analysis to create a versatile chat assistant. Users can interact with their uploaded PDF documents or leverage web search to get informative responses to their queries. Product available for testing here: https://shreyas094-searchgpt.hf.space

## Features

* **PDF Document Chat**: Upload and interact with multiple PDF documents.
* **Web Search Integration**: Option to use web search for answering queries.
* **Multiple AI Models**: Choose from a selection of powerful language models.
* **Customizable Responses**: Adjust temperature and API call settings for fine-tuned outputs.
* **User-friendly Interface**: Built with Gradio for an intuitive chat experience.
* **Document Selection**: Choose which uploaded documents to include in your queries.
* **Document Management**: Allows uploading, deleting, and refreshing document lists. Persistent storage of uploaded documents using JSON

## How It Works

1. **Document Processing**:
   - Upload PDF documents using either PyPDF or LlamaParse.
   - Documents are processed and stored in a FAISS vector database for efficient retrieval.

2. **Embedding**:
   - Utilizes HuggingFace embeddings (default: 'sentence-transformers/all-mpnet-base-v2') for document indexing and query matching.

3. **Query Processing**:
   - For PDF queries, relevant document sections are retrieved from the FAISS database.
   - For web searches, results are fetched using the DuckDuckGo search API.

4. **Response Generation**:
   - Queries are processed using the selected AI model (options include Mistral, Mixtral, and others).
   - Responses are generated based on the retrieved context (from PDFs or web search).

5. **User Interaction**:
   - Users can chat with the AI, asking questions about uploaded documents or general queries.
   - The interface allows for adjusting model parameters and switching between PDF and web search modes.

6. **Document Management**:
   - The application maintains a list of uploaded documents in a JSON file (uploaded_documents.json).
   - It provides functions to update, delete, and refresh the document list. 

## Setup and Usage

1. Install the required dependencies (list of dependencies to be added).
2. Set up the necessary API keys and tokens in your environment variables.
3. Run the main script to launch the Gradio interface.
4. Upload PDF documents using the file input at the top of the interface.
5. Select documents to query using the checkboxes.
6. Toggle between PDF chat and web search modes as needed.
7. Adjust temperature and number of API calls to fine-tune responses.
8. Start chatting and asking questions!

## Models

The project supports multiple AI models, including:
* mistralai/Mistral-7B-Instruct-v0.3
* mistralai/Mixtral-8x7B-Instruct-v0.1
* meta/llama-3.1-8b-instruct
* mistralai/Mistral-Nemo-Instruct-2407

## Future Improvements

* Integration of more embedding models for improved performance.
* Enhanced PDF parsing capabilities.
* Support for additional file formats beyond PDF.
* Improved caching for faster response times.

## Contribution

Contributions to this project are welcome! Please feel free to submit issues or pull requests on the project's GitHub repository.

## Contact

For any queries feel free to reach out @desai.shreyas94@gmail.com or discord - shreyas094



