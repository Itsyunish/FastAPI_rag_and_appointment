from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

def get_rag_chain(llm, retriever):
    system_prompt = """
    You are a helpful assistant. Use the following context to answer the question:
    {context}

    - Answer only using the context.
    - If the context does not have the answer, say "I don't know."
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)
