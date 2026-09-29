from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableParallel , RunnableBranch , RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='write a report on {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='summarise the following text \n {text}',
    input_variables=['text']
)

reportChain = RunnableSequence(prompt1 , model , parser)

branchChain = RunnableBranch(
    (lambda x: len(x.split()) > 500 , RunnableSequence(prompt2 , model , parser)),
    RunnablePassthrough()
)

finalChain = RunnableSequence(reportChain,branchChain)

result = finalChain.invoke({'topic' : 'Delhi university current protest'})

print(result)