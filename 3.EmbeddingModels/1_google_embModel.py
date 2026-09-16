from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv() 

embModel = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    dimensions=40
)

documents = [
    "Delhi is the capital of India",
    "The capital of France is Paris",
    "The capital of Germany is Berlin",
    "The capital of Italy is Rome",
    "The capital of Spain is Madrid",
    ]

result = embModel.embed_documents(documents)

print(str(result))