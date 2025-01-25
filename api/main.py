import os
from typing import TypedDict, List

import spacy
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PubMedLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.retrievers import PubMedRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAI
from langgraph.constants import START
from langgraph.graph import StateGraph
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_keywords(question):
    prompt_template = """
    You are a helpful assistant that extracts keywords related to biology, diseases, genes, proteins, and medical terms from a given question.

    Please extract all relevant biological keywords from the following question, such as diseases, genes, medical conditions, and related biological entities:

    Question: "{question}"

    Keywords:
    """

    # Initialize the OpenAI LLM with LangChain
    helper = OpenAI(api_key=os.getenv("KEY"), model="gpt-3.5-turbo-instruct", temperature=0.0)

    # Create the LangChain prompt using the template
    prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

    # Chain the prompt with the LLM
    from langchain.chains import LLMChain
    chain = LLMChain(llm=helper, prompt=prompt)

    # Function to extract biological keywords using the chain
    return chain.run(question)


llm = ChatOpenAI(
    temperature=0,
    model="gpt-4",
    api_key=os.environ.get("KEY")
)

class Query(BaseModel):
    question: str

def create_analysis_graph(vector_store, prompt, llm):
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str
        sources: List[str]

    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {
            "context": retrieved_docs,
            "sources": [doc.metadata for doc in retrieved_docs]
        }

    def generate(state: State):
        context = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": context})
        response = llm.invoke(messages)

        result = set()
        for source in state["sources"]:
            result.add(f"{source['uid']}, {source['Title']}")

        result = list(result)

        return {
            "answer": response.content,
            "sources": result
        }

    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")

    return graph_builder.compile()

@app.post("/api/query")
async def query(query: Query):
    try:
        # Extract keywords and use in PubMed search
        retriever = PubMedRetriever(top_k_results = 10)
        q = extract_keywords(query.question)
        docs = retriever.invoke(q)

        # Split documents
        text_splitter = SemanticChunker(
            embeddings=OpenAIEmbeddings(
                model="text-embedding-3-large",
                api_key=os.environ.get("KEY")
            )
        )
        docs = text_splitter.split_documents(docs)

        vector_store = InMemoryVectorStore(
            embedding=OpenAIEmbeddings(
                model="text-embedding-3-large",
                api_key=os.environ.get("KEY")
            )
        )

        vector_store.add_documents(documents=docs)

        prompt = PromptTemplate.from_template(
            ''' You are an expert LLM capable of synthexesizing knowledge from research papers supplied to you. Provide an answer.  
               {context}
               Question: {question}
               Answer: '''
        )

        graph = create_analysis_graph(vector_store, prompt, llm)
        response = graph.invoke({"question": f"{query.question}"})
        return {"result": response['answer'], "sources": response['sources']}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))