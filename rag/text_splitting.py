# RAG system for reinsurance data exploration using OpenAI

# === 1. Imports ===
import os
import glob
import json
import re
import openai
import faiss
import numpy as np
from typing import List, Tuple
from langchain.document_loaders import UnstructuredFileLoader
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings as SemanticEmbeddings
from dotenv import load_dotenv

load_dotenv()
# === 2. Environment ===
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API_KEY")


# === 3. Load and preprocess documents ===
def preprocess_contract(text: str) -> str:
    # Normalize line breaks and merge lines that are part of the same sentence/paragraph
    lines = text.splitlines()
    grouped_lines = []
    current_line = ""
    for line in lines: 
        line = line.strip()
        if not line:
            continue  # skip empty lines
        if current_line and not re.search(r'[\.;!?]$', current_line):
            current_line += " " + line  # append line if previous one didn't end with punctuation
        else:
            if current_line:
                grouped_lines.append(current_line.strip())
            current_line = line
    if current_line:
        grouped_lines.append(current_line.strip())

    # Join back with single newlines
    return "\n".join(grouped_lines).strip()

def load_documents(directory: str) -> List[Document]:
    file_paths = glob.glob(os.path.join(directory, '*'))
    docs = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[-1].lower()
        if ext in ['.ndjson', '.json']:  # skip unsupported formats for now
            print(f"Skipping unsupported file format: {file_path}")
            continue
        try:
            loader = UnstructuredFileLoader(file_path)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = os.path.basename(file_path)
                doc.page_content = preprocess_contract(doc.page_content)
            docs.extend(loaded_docs)
        except Exception as e:
            print(f"⚠️ Failed to load {file_path}: {e}")
    return docs

# === 4. Semantic chunking ===
def chunk_documents(docs: List[Document]) -> List[Document]:
    embedder = SemanticEmbeddings()
    chunker = SemanticChunker(embeddings=embedder, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=90)
    all_chunks = []
    for doc in docs:
        try:
            chunks = chunker.split_text(doc.page_content)
            if not chunks:
                raise ValueError("No semantic chunks returned")
            for chunk_text in chunks:
                chunk_doc = Document(page_content=chunk_text, metadata=doc.metadata)
                all_chunks.append(chunk_doc)
        except Exception as e:
            print(f"⚠️ Failed semantic chunking for {doc.metadata.get('source')}: {e}")
    return all_chunks

# === 5. Embed and store in FAISS ===
def build_vectorstore(chunks: List[Document]) -> FAISS:
    if not chunks:
        raise ValueError("No chunks to index. Check if documents are loaded and chunked properly.")
    embedding_model = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embedding_model)

# === 6. Set up the LLM + Retrieval QA chain ===
def create_qa_chain(vectorstore: FAISS) -> RetrievalQA:
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4")
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )

# === 7. Ask questions with filtering based on year ===
def ask_question(qa_chain: RetrievalQA, query: str) -> Tuple[str, List[Document]]:
    # Extract year from query
    year_match = re.search(r"20\d{2}", query)
    year = year_match.group() if year_match else None

    if year:
        print(f"🔎 Filtering chunks based on year: {year}")
        original_docs = qa_chain.retriever.vectorstore.docs
        filtered_docs = [doc for doc in original_docs if year in doc.metadata.get("source", "")]

        if not filtered_docs:
            print(f"⚠️ No documents matched the year {year}. Proceeding without filter.")
            result = qa_chain({"query": query})
        else:
            filtered_vectorstore = FAISS.from_documents(filtered_docs, OpenAIEmbeddings())
            retriever = filtered_vectorstore.as_retriever(search_kwargs={"k": 5})
            chain = RetrievalQA.from_chain_type(
                llm=qa_chain.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            result = chain({"query": query})
    else:
        result = qa_chain({"query": query})

    return result['result'], result['source_documents']

# === 8. Main flow ===
if __name__ == "__main__":
    docs = load_documents("./data/submissions/florida")
    chunks = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks)
    qa_chain = create_qa_chain(vectorstore)

    while True:
        question = input("\nAsk your reinsurance question (or type 'exit'): ")
        if question.lower() == 'exit':
            break
        answer, sources = ask_question(qa_chain, question)
        print("\nAnswer:\n", answer)
        print("\nSources:")
        for doc in sources:
            print(f"- {doc.metadata.get('source', 'unknown')} | Preview: {doc.page_content[:200]}...")
