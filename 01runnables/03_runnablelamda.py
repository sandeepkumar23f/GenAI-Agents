from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableParallel, RunnableLambda

load_dotenv()

model = init_chat_model("groq:openai/gpt-oss-120b", max_tokens=600)
parser = StrOutputParser()

short_prompt = ChatPromptTemplate.from_template("Explain {topic} in 1-2 lines")
detailed_prompt = ChatPromptTemplate.from_template("Explain {topic} in detail")

chain = RunnableParallel({
    "short": RunnableLambda(lambda x: x["short"]) | short_prompt | model | parser,
    "detailed": RunnableLambda(lambda x: x["detailed"]) | detailed_prompt | model | parser,
})

result = chain.invoke({
    "short": {"topic": "Embeddings in GenAI"},
    "detailed": {"topic": "Vector databases"},
})

print("SHORT:\n", result["short"])
print("\nDETAILED:\n", result["detailed"])