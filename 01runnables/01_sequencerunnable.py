from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
load_dotenv()


# 1. Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

# 2. Model
model = init_chat_model(
    "groq:openai/gpt-oss-120b",
    max_tokens=200,   # hard limit on the reply length
    
)

# 3. Output Parser
parser = StrOutputParser()

# 4. Runnable 
chain = prompt | model | parser

result = chain.invoke("What is API")

# print(result)
print("\n")
print(type(chain))