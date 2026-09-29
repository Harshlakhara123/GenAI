from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt = PromptTemplate(
    template='Write a short poem about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser)
#this can also be written as  chain = prompt | model | parser 
# and that is also called LCEL

print(chain.invoke({'topic': 'the ocean'}))