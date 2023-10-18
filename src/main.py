# from langchain.chat_models import ChatOpenAI
# from langchain.prompts.chat import (
#     PromptTemplate
# )
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.llms import OpenAI
# from langchain.chains import RetrievalQA
# from langchain.vectorstores import Qdrant
# from langchain.output_parsers import PydanticOutputParser
# from qdrant_client.http import models as rest
# from pydantic import BaseModel, Field
# from langchain.document_loaders.csv_loader import CSVLoader

# import csv
# from typing import Dict, List, Optional
# from langchain.document_loaders.base import BaseLoader
# from langchain.docstore.document import Document

# import os
# from dotenv import load_dotenv
# from langchain.vectorstores import FAISS
# import openai

# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# class Book(BaseModel):
#     isbn13: int = Field(description="international serial book number of length 13")
#     isbn10: int = Field(description="international serial book number of length 13")
#     title: str = Field(description="book title")
#     subtitle: str | None = Field(description="book subtitle")
#     authors: list[str] = Field(description="book author's full names")
#     categories: str = Field(description="book category")
#     thumbnail: str = Field(description="url with image of thumbnail")
#     description: str = Field(description="description of the book content")
#     published_year: int = Field(description="year of publishing a book")
#     average_rating: float = Field(description="average readers rating as decimal number")
#     num_pages: int = Field(description="number of book's pages")
#     ratings_count: int = Field(description="number of readers ratings given")

# loader = CSVLoader(
#     file_path="./src/dataset_small.csv", source_column="title")

# data = loader.load()
# print(f"data: {data}")

# book_request = "recommend me a romantic novel from 1950s"

# parser = PydanticOutputParser(pydantic_object=Book)
# prompt = PromptTemplate(template="You are a librarian bot, recommend a list of 5 books based on user query.\n{format_instructions}\n{query}\n",
#                         input_variables=["query"],
#                         partial_variables={"format_instructions": parser.get_format_instructions()})
# _input = prompt.format_prompt(query=book_request)
# model = ChatOpenAI()
# output = model.predict(_input.to_string())
# parsed = parser.parse(output)
# print(parsed.title)
# print(parsed.published_year)

# embeddings = OpenAIEmbeddings()
# docsearch = FAISS.from_documents(data, embeddings)

# qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

# print(qa.run("a romantic novel from 1950s"))

# # def main() -> None:
# #     pass

# # if __name__ == "__main__":
# #     main()