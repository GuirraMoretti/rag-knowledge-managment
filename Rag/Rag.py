from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Carregar a base de dados com embeddings (por exemplo, FAISS)
vector_store = FAISS.load_local("faiss_index_path", OpenAIEmbeddings())

# Configuração do retriever
retriever = vector_store.as_retriever()

# Configuração do modelo de linguagem (OpenAI GPT, por exemplo)
llm = OpenAI(model="text-davinci-003")

# Criando o RAG (Retriever-augmented Generation)
rag_chain = RetrievalQA(combine_docs_chain=llm, retriever=retriever)

# Consultar o modelo
query = "Qual é a política de férias da empresa?"
response = rag_chain.run(query)
print(response)
