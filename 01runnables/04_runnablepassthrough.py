from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = init_chat_model("groq:openai/gpt-oss-120b", max_tokens=800)
parser = StrOutputParser()

# Prompt 1: generate code from a topic
code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code generator"),
    ("human", "{topic}"),
])

# Prompt 2: explain code (expects a dictionary with a "code" key)
explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who explains code in simple terms"),
    ("human", "Explain the following code in simple words: \n{code}"),
])

# STEP 1 (Sequence): {"topic": ...} -> prompt -> model -> plain string (the code)
sequence = code_prompt | model | parser

# STEP 2 (Parallel): receives the code string and builds two outputs
sequence2 = RunnableParallel({
    # branch 1: pass the code through unchanged, so it appears in the final result
    "code": RunnablePassthrough(),
    # branch 2: wrap the string as {"code": ...}, then explain it
    "explanation": RunnableLambda(lambda code: {"code": code}) | explain_prompt | model | parser,
})

# STEP 3: connect both steps into one chain
chain = sequence | sequence2

result = chain.invoke({"topic": "please write a code of palindrome in python"})

print("CODE:\n", result["code"])
print("\nEXPLANATION:\n", result["explanation"])