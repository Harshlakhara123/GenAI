from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# annotated bcos we want to add description to the fields in the TypedDict
# Optional bcos the field may or may not be present in the output
class Review(TypedDict):
    summary: Annotated[str,"a brief summary of the review"]
    sentiment: Annotated[str,"the sentiment of the review, either positive,negative or neutral"]
    pros:Annotated[Optional[list[str]],"the pros of the product, if any"]
    cons:Annotated[Optional[list[str]],"the cons of the product, if any"]

structuredModel = model.with_structured_output(Review)

result = structuredModel.invoke("""The hardware is great , but the software feels bloated.There are too many pre-installed apps that i cant remove.Also the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")

print(result)