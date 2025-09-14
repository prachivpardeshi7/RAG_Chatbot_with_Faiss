# RAG Chatbot 

A Retrieval-Augmented Generation (RAG) based chatbot built with **LangChain**, **FAISS**, **HuggingFace Embeddings**, and **Groq LLM**.  
The bot answers user queries by retrieving relevant context from documents and generating accurate responses.

## Features
- Loads and indexes PDF/text documents
- Embedding generation using `all-MiniLM-L6-v2`
- Vector search with **FAISS**
- Query answering with **Groq LLM**
- Source document traceability


## Usage 
-1.Add your documents inside the data/ folder.
-2.Run the chatbot:python -m streamlit run app.py
-3.Enter your query and get answers with cited sources.
   -Eg: Enter your query: What is the objective of the HR policy?
       Answer: The objective of the HR policy is to provide support to employees through continuity, communication, orientation, and mentoring.

