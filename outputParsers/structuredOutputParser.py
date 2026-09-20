from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

schema = [
    ResponseSchema(name="fact1", description="fact1 of the topic"),
    ResponseSchema(name="fact2", description="fact2 of the topic"),
    ResponseSchema(name="fact3", description="fact3 of the topic")
]
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.invoke({'topic':'blackhole'})

result = model.invoke(prompt)

finalResult = parser.parse(result.content)

print(finalResult)