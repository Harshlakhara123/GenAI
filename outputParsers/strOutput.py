from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()



model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

template1 = PromptTemplate(
    template='Write a detail report on {topic}',
    input_variables=["topic"]
)
template2 = PromptTemplate(
    template='Write 5 line summary on the following text. /n {text}',
    input_variables=["text"]
)

# prompt1 = template1.invoke({"topic": "BlackHole"})
# result = model.invoke(prompt1)

# prompt2 = template2.invoke({"text": result.content})
# result2 = model.invoke(prompt2)

# print(result2.content)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser 

result = chain.invoke(({"topic": "BlackHole"}))

print(result)