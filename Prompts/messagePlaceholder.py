from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chatTemplate = ChatPromptTemplate([
    ('system', 'You are a helpful customer support assistant.'),
    MessagesPlaceholder(variable_name='chatHistory'),
    ('human' , '{query}')
])

chatHistory = []
chat_history_path = Path(__file__).with_name('chatHistory.txt')

if chat_history_path.exists():
    with open(chat_history_path, encoding='utf-8') as f:
        chatHistory.extend(f.readlines())

print(chatHistory)

prompt = chatTemplate.invoke({'chatHistory': chatHistory, 'query': 'Where is my refund?'})

print(prompt)