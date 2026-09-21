from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda,RunnablePassthrough
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description='give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(
    template='classify the sentiment of the following text into positive or negative \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifierChain = RunnablePassthrough.assign(
    parsedResult = prompt1 | model | parser2
)

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)


# if else chains
branchedChain = RunnableBranch(
    (lambda x:x['parsedResult'].sentiment == 'positive' , prompt2 | model | parser),
    (lambda x:x['parsedResult'].sentiment == 'negative' , prompt3 | model | parser),
    RunnableLambda(lambda x: "couldnt find sentiment")
)

finalChain = classifierChain | branchedChain

result = finalChain.invoke({'feedback' : 'this is a very bad smartphone'})

print(result)