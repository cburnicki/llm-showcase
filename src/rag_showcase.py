from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings

user_query = "Which rooms are missing equipment?"

embeddings = OpenAIEmbeddings()
vector_store: FAISS = FAISS.load_local("./vector_store", embeddings, allow_dangerous_deserialization=True)

llm = Ollama(model="mistral")
prompt = ChatPromptTemplate.from_template("""You are an assistant helping with customer feedback.
Answer the following question based only on the emails in the provided context:

<context>
{context}
</context>

Question: {input}""")

chain = prompt | llm

# use the user query to load related documents and concatenate them to a context string
related_documents = vector_store.as_retriever(search_kwargs={"k": 12}).invoke(user_query)
context = [(d.page_content + "\n") for d in related_documents]

res = chain.invoke({"input": user_query, "context": context})
print(res)