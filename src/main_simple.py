from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.document_loaders.csv_loader import CSVLoader

import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

loader = CSVLoader(
    file_path="./src/dataset_small.csv", source_column="title")

data = loader.load()

embeddings = OpenAIEmbeddings()

quadrant_docsearch = Qdrant.from_documents(
    data,
    embeddings,
    location=":memory:",
    collection_name="book"
)

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=quadrant_docsearch.as_retriever(), return_source_documents=True)

while True:
    user_input = input("Hi I am AI librarian what can I help you with?\n")

    book_request = "You are a librarian. Help the user answer their question. Do not provide the ISBN." +\
                       f"\nUser:{user_input}"
    result = qa({"query": book_request})
    print(len(result["source_documents"]))
    print(result["result"])