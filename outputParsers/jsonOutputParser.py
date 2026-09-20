from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me the name, age and address of a fictional person. \n {format instructions}',
    input_variables=[],
    partial_variables={"format instructions": parser.get_format_instructions()}
)

# prompt = template.format()
# print(prompt)

# result = model.invoke(prompt)
# finalResult = parser.parse(result.content)
# print(finalResult)

chain = template | model | parser

result = chain.invoke({})
print(result)

# the structure of the json output will be decided by the LLM , we dont have any way to enforce a particular structure, so we need to handle the output accordingly. The output parser will try to parse the output into a dictionary, but if the output is not in the expected format, it may raise an error.
# thats why we use Structured output parser
