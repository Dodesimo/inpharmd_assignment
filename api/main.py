import os
from typing import TypedDict, List

import spacy
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PubMedLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langgraph.constants import START
from langgraph.graph import StateGraph
from pydantic import BaseModel

app = FastAPI()
load_dotenv()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_keywords(question):
    doc = nlp(question)
    keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    return " OR ".join(keywords)

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
        return {
            "answer": response.content,
            "sources": state["sources"]
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
        keywords = extract_keywords(query.question)
        docs = PubMedLoader(query=f"({keywords}) AND (journal article[Publication Type])").load()

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
            ''' You are an expert LLM capable of synthesizing knowledge from research papers supplied to you. Utilize the papers you have to provide an answer, and cite all sources that you used. 
               {context}
               Question: {question}
               Answer: '''
        )

        graph = create_analysis_graph(vector_store, prompt, llm)
        response = graph.invoke({"question": f"{query.question}"})
        return {"result": response['answer'], "sources": response['sources']}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))