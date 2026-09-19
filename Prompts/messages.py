from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about eventloop in nodejs?"),
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))
print(messages)