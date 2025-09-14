# RAG Chatbot 

A Retrieval-Augmented Generation (RAG) based chatbot built with **LangChain**, **FAISS**, **HuggingFace Embeddings**, and **Groq LLM**.  
The bot answers user queries by retrieving relevant context from documents and generating accurate responses.

## Features
- Loads and indexes PDF document
- Embedding generation using `all-MiniLM-L6-v2`
- Vector search with **FAISS**
- Query answering with **Groq LLM**
- Source document traceability

 ##Installation

1.Clone the repository:
git clone https://github.com/prachivpardeshi7/rag_chatbot.git
cd rag_chatbot<br>

2.Create a virtual environment and activate it:<br>
python -m venv venv<br>
source venv/bin/activate   # For Linux/Mac <br>
venv\Scripts\activate      # For Windows < <br>

3.Install dependencies: <br>
pip install -r requirements.txt <br>

4.Add your API keys in .env file:  <br>
GROQ_API_KEY=your_groq_api_key  <br>


## Usage 

   1.Add your documents inside the data/ folder. <br>
   2.Run the chatbot:python -m streamlit run app.py <br>
   3.Enter your query and get answers with cited sources. <br>
      Example:- Enter your query: What is the objective of the HR policy? <br>
          Answer: The objective of the HR policy is to provide support to employees through continuity, communication, orientation, and mentoring. <br>

