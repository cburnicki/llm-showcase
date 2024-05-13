from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Email dataset from https://figshare.com/articles/dataset/Email_Dataset_by_Department/5765376
loader = CSVLoader(file_path="./example_data/email-dataset-unclassified.csv")
documents = loader.load()
joined_emails = "\n".join([d.page_content for d in documents[:15]])


llm = Ollama(model="mistral")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant and your task is to read and summarize emails containing feedback by people using our facilities.\
            For every technical issue, describe the issue and provide a list of all the rooms where service is required.\
            Add every equipemnt request to a list with entries formatted '<equipment>: <location> (<time>)'.\
            Please do not include anything that wasn't in the emails. \
            Here are the most recent emails: {data}",
        )
    ]
)
output_parser = StrOutputParser()

# create a pipeline
chain = prompt | llm | output_parser

print(chain.invoke({"data": joined_emails}))
