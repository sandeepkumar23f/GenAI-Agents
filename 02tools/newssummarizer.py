from dotenv import load_dotenv
load_dotenv()   # first, so TAVILY_API_KEY and GROQ_API_KEY are loaded

from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

model = init_chat_model("groq:openai/gpt-oss-120b", max_tokens=600)

search_tool = TavilySearch(max_results=5, topic="general")

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant.
Summarize the following news into clear bullet points.
Use only the information given.

{news}"""
)

chain = prompt | model | StrOutputParser()

# 1. Search
response = search_tool.invoke({"query": "Latest AI news"})

# 2. Turn the results into short text (keeps token use low)
news_text = "\n".join(
    f"- {r['title']}: {r['content'][:300]} ({r['url']})"
    for r in response["results"]
)

# 3. Summarize
print(chain.invoke({"news": news_text}))