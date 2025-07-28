from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
import os

def initialize_qa_chain(csv_path='raw_with_extractions.csv', persist_dir='my_chroma_db'):
    # 1. Load CSV
    loader = CSVLoader(file_path=csv_path)
    docs = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)

    # 3. HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Vector Store (Chroma)
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_name='sample'
    )

    # 5. Add Documents (avoid duplicates if already exists)
    if len(vector_store.get()["ids"]) == 0:
        batch_size = 5000
        for i in range(0, len(split_docs), batch_size):
            vector_store.add_documents(split_docs[i:i + batch_size])

    # 6. Retriever
    retriever = vector_store.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 3, 'lambda_mult': 0.5}
    )

    # 7. LLM (HuggingFace)
    hf_pipeline = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        max_length=512,
        do_sample=True,
        temperature=0.7,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # 8. Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        return_source_documents=True  # helpful for UI
    )

    return qa_chain
