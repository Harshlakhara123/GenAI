from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(gt=18, description="The person's age")
    city: str = Field(description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Give a person\'s name, age, and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt = template.invoke({'place':'India'})

# result = model.invoke(prompt)

# finalResult = parser.parse(result.content)
# print(finalResult)

chain = template | model | parser

finalResult = chain.invoke({'place':'New York'})
print(finalResult)
