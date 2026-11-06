from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableParallel

load_dotenv()
model = init_chat_model(
    "groq:openai/gpt-oss-120b",
     max_tokens=200,   
)

parser = StrOutputParser()

short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)

detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)

topic = "Embeddings in GenAi"

chain = RunnableParallel({
    "short": short_prompt | model | parser,
    "detailed" : detailed_prompt | model | parser
})

result = chain.invoke({"topic": "Embeddings in GenAi"})
print(result)

# Or we can print the short and detailed separately
print(result['short'])
print(result['detailed'])