from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnablePassthrough , RunnableParallel , RunnableLambda

load_dotenv()

def word_count(text):
    return len(text.split())


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
    'wordCount': RunnableLambda(word_count)
    #'wordCount' : RunnableLambda(lambda x: len(x.split()))
})

finalChain = RunnableSequence(jokeChain,parallelChain)

result = finalChain.invoke({'topic' : 'AI'})

print(result)

