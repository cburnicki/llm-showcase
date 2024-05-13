from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Email dataset from https://figshare.com/articles/dataset/Email_Dataset_by_Department/5765376
loader = CSVLoader(file_path="./example_data/email-dataset-unclassified.csv")
documents = loader.load()
vector_store = FAISS.from_documents(documents, OpenAIEmbeddings())
vector_store.save_local("./vector_store")