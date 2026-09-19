from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()  # Load environment variables from .env file

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
chatHistory = [SystemMessage(content="You are a helpful assistant.")]
while True:
    user_input = input('You: ')
    chatHistory.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chatHistory)
    chatHistory.append(AIMessage(content=result.content))
    print('Bot:', result.content)
print(chatHistory)