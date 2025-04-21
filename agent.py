from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
import os


#Setting up the LLM model
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    groq_api_key = groq_api_key,
    model_name = "Llama3-8b-8192"
)


##Defining Arxv Tool
import arxiv 
from langchain.tools import Tool
from typing import List
def fetch_arxiv_paper(query:str) -> List:
    search = arxiv.Search(
        query = query,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    for result in search.results():
        papers.append({
            "title": result.title, 
            "authors": result.authors,
            "summary": result.summary,
            "url": result.entry_id
        })
        return papers
    
arxiv_tool = Tool(
    name = "ArxicPaperSearch",
    func = fetch_arxiv_paper,
    description="Fetches relevant research papers related to stock market prediction from arXiv"
)

#Defining wikipedia tool
wiki = WikipediaAPIWrapper()
def fetch_wikipedia(query: str):
    return wiki.run(query)

wiki_tool = Tool(
    name="WikipediaSearch",
    func=fetch_wikipedia,
    description="Fetches relevant articles from Wikipedia to explain terms or concepts of the stock market."
)
#Defining DuckDuckGO search

from langchain.tools import Tool
from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun

# Initialize DuckDuckGo search tool
search = DuckDuckGoSearchRun()

def fetch_duckduckgo(query: str):
    return search.run(query)

duckduckgo_tool = Tool(
    name="DuckDuckGoSearch",
    func=fetch_duckduckgo,
    description="Searches DuckDuckGo for relevant information."
)
tools = [
    arxiv_tool,
    wiki_tool, 
    duckduckgo_tool
]
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

from langchain.agents import initialize_agent, AgentType
agent = initialize_agent(
    tools = tools,
    llm = llm,
    memory = memory,
    agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
    verbode = True
)