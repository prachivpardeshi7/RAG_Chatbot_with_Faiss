import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert GROQ_API_KEY is not None, "GROQ_API_KEY missing from .env"


def load_llm():
    return ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",   
        temperature=0.5,
        max_tokens=100
    )

# Prompt Template
PROMPT_TEMPLATE = """
Use the following context to answer the question.
If you don't know the answer, just say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

def clean_documents(docs):
    cleaned = []
    for doc in docs:
        text = " ".join([line.strip() for line in doc.page_content.split("\n") if line.strip()])
        cleaned.append(Document(page_content=text, metadata=doc.metadata))
    return cleaned

# Load vector DB 
VECTOR_DB_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGroq(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Run the chain
query = input("Enter your query: ")
response = qa_chain.invoke({"query": query})

# Display result
print("\n Answer:\n", response["result"])
print("\n Source Documents:")
for i, doc in enumerate(response["source_documents"], 1):
    print(f"\n--- Document {i} ---\n{doc.page_content[:18]}...\n")
