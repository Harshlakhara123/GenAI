from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough , RunnableSequence , RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='write a short joke about {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='explain the following text of the joke \n {text}',
    input_variables=['text']
)

jokeChain = RunnableSequence(prompt1 , model , parser)

parallelChain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explanation' : RunnableSequence(prompt2 , model , parser)
})

finalChain = RunnableSequence(jokeChain,parallelChain)

result = finalChain.invoke({'topic':'cricket'})

print(result)