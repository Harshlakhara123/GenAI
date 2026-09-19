from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Review(BaseModel):
    summary: str = Field(description="a brief summary of the review")
    sentiment: Literal["pos", "neg", "neutral"] = Field(description="the sentiment of the review, either positive, negative or neutral")
    pros: Optional[list[str]] = Field(default=None, description="the pros of the product, if any")
    cons: Optional[list[str]] = Field(default=None, description="the cons of the product, if any")
    name: Optional[str] = Field(default=None, description="the name of the reviewer, if available")

structuredModel = model.with_structured_output(Review)

result = structuredModel.invoke("""The hardware is great , but the software feels bloated.There are too many pre-installed apps that i cant remove.Also the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")

print(result)