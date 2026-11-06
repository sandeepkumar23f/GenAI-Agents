
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from rich import print

# 1. Creating a tool
@tool
def get_text_length(text: str) -> str:
    """Return the number of characters present in the text."""
    return str(len(text))

# Initializing LLM
llm = init_chat_model(
    "groq:openai/gpt-oss-120b",
    max_tokens=600
)

# 2. Binding the tool
llm_with_tool = llm.bind_tools([get_text_length])

# 3. Invoking the LLM
result = llm.invoke("Return the number of characters present in the text: 'hello how are you")
result2 = llm_with_tool.invoke("Return the number of characters present in the text: 'hello how are you")

print(result)
print()
print(result2)
