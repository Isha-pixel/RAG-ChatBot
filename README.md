# Retrieval-Augmented Generation (RAG) Chatbot using LangChain, OpenAI, Google Generative AI, and Hugging Face

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-App-red.svg" alt="Streamlit"></a>
  <a href="https://python.langchain.com/"><img src="https://img.shields.io/badge/LangChain-RAG%20Pipeline-green.svg" alt="LangChain"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/Hugging%20Face-Models-yellow.svg" alt="Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License"></a>
  <a href="#contributing"><img src="https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg" alt="Contributions"></a>
</p>

---

<div align="center">
  <img src="https://github.com/AlaGrine/RAG_chatabot_with_Langchain/blob/main/data/docs/RAG_architecture.png" >
  <figcaption>RAG architecture with Langchain components.</figcaption>
</div>

---

## Project Overview

Large Language Models (LLMs) like GPT and Gemini are powerful at generating human-like responses, but they are limited by the static data they were trained on — meaning they can sometimes produce outdated or inaccurate answers.

To overcome this limitation, this project implements a **Retrieval-Augmented Generation (RAG)** system using **LangChain**.  
RAG allows the chatbot to dynamically retrieve relevant information from uploaded documents and augment the LLM’s response with up-to-date, context-specific data.

This project integrates APIs from  
- **OpenAI** ([platform.openai.com](https://platform.openai.com/))  
- **Google Generative AI** ([ai.google.dev](https://ai.google.dev/))  
- **Hugging Face** ([huggingface.co](https://huggingface.co/))  

Users can upload `.txt`, `.pdf`, `.csv`, or `.docx` files, and the chatbot will retrieve the most relevant chunks of information before generating an accurate, context-aware response.  
A simple and interactive **Streamlit** interface is also included for real-time conversations with your data.

---

## Key Features
- **Multi-provider LLM support**: OpenAI, Gemini, and Hugging Face models  
- **Document-aware retrieval** using Chroma vectorstore  
- **RAG pipeline built with LangChain** (loader → splitter → embeddings → retriever → LLM chain)  
- **Streamlit UI** for easy local deployment and interaction  
- **Secure API handling** using `.env` (no keys in code)

---

## ⚙️ Installation

### Prerequisites
- Python **3.9+**
- GPU optional (for Hugging Face local models)
- Internet access for API-based models

### Required Libraries
Install all dependencies via `requirements.txt`:
```bash
pip install -r requirements.txt
```

This project requires Python 3 and the following Python libraries installed:

`langchain`,`langchain-openai`, `langchain-google-genai`, `chromadb`, `streamlit`, `streamlit`

The full list of requirements can be found in `requirements.txt`

## Instructions <a name="instructions"></a>

Option 1 — Run the Streamlit App locally:

1. Create a virtual environment: `python -m venv langchain_env`
2. Activate the virtual environment : `.\langchainenv\Scripts\activate` on Windows.
3. Run the following command in the directory: `cd RAG_Chatabot_Langchain`
4. Install the required dependencies `pip install -r requirements.txt`
5. Start the app: `streamlit run RAG_app.py`
6. In the sidebar, select the LLM provider (OpenAI, Google Generative AI or HuggingFace), choose an LLM (GPT-3.5, GPT-4, Gemini-pro or Mistral-7B-Instruct-v0.2), adjust its parameters, and insert your API keys.
7. Create or load a Chroma vectorstore.
8. Chat with your documents: ask questions and get AI answers.


Option 2 — Run the RAG Script or Notebook:

1. Open run_RAG_code.ipynb in Jupyter or Colab and run all cells sequentially.
2. Create a .env file in the root directory:
```bash
  OPENAI_API_KEY=your_openai_key
  GOOGLE_API_KEY=your_google_key
  HUGGINGFACEHUB_API_TOKEN=your_hf_token
  COHERE_API_KEY=your_cohere_key 
  ```
3. Run the notebook.

---

##Architecture Summary

Document Loader → Reads and parses files in multiple formats
Text Splitter → Divides text into manageable semantic chunks
Embeddings → Converts text chunks into dense vectors
Vector Store (ChromaDB) → Stores embeddings for fast similarity search
Retriever → Finds relevant chunks for the user query
LLM Chain → Generates context-aware answers
UI Layer (Streamlit) → Enables user interaction and configuration
