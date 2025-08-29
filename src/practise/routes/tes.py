# built-in
import re
import asyncio
# custom package 
from config import settings
from src.rag.models.llm_schema import GetContext
from src.rag.utility.prompts import get_chain_prompts


# langchain imports
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable, RunnableMap
from langchain_core.output_parsers import StrOutputParser


llm = ChatOpenAI(
    api_key=settings.OPENAI_API_KEY,
    model=settings.OPENAI_MODEL,
    temperature=0,
)


prpmpt = """
ut6rsdykchggggggggg

chunk:
{context1}

{query}


"""
prompt =  get_chain_prompts()

# TODO: re-organized this tool
async def get_context(self):
    async def get_context_tool(query:str):
        print(f"[INFO] --- get context tool: {query} :--")
        
        
        all_namespaces = self.get("namespace_lis")
        all_similar_docs = []
        all_page_numbers = []

        async with settings.pc.IndexAsyncio(host=settings.PINECONE_HOST_URL) as idx:

            result = await asyncio.gather(
                *(idx.search(
                    namespace = ns,
                    query={
                        "inputs": {"text": query}, 
                        "top_k": 1
                    },
                ) for ns in all_namespaces)
            )
            
            result = await idx.search(
                    namespace = ns,
                    query={
                        "inputs": {"text": query}, 
                        "top_k": 1
                    },
                )

        for index in range(len(all_namespaces)):
            

            first_doc = next((search.get("fields",{}).get("text") for search in result[index].get("result",{}).get("hits",{})), "")
            page_number = next((search.get("fields",{}).get("page_number") for search in result[index].get("result",{}).get("hits",{})), "")

            all_page_numbers.append(page_number)
            all_similar_docs.append(first_doc)
                
            print(f"[INFO] - page number {page_number}")
            print(f"[INFO] - DOCUMENT: \n  {first_doc} \n")
        # print(f"[INFO] - all similar docs 1: \n {all_similar_docs[0]} \n")
        # print(f"[INFO] - all similar docs 2: \n {all_similar_docs[1]} \n")
        # print(f"[INFO] - all similar docs 3: \n {all_similar_docs[2]} \n")
        # print(f"[INFO] - all similar docs 4: \n {all_similar_docs[3]} \n")
        # print(f"[INFO] - all similar docs 5: \n {all_similar_docs[4]} \n")

        # print(f"[INFO] - all similar docs 5: \n {all_page_numbers} \n")

        # print(f"[INFO] - user query {self.get('query')}")
        
        chain:Runnable = RunnableMap({
            "query": lambda x: x['query'],
            "context1": lambda x: result,
        }) | prompt | llm | StrOutputParser() 

        chain_res = await chain.ainvoke({'query':query})

        return chain_res
        
        
    return get_context_tool