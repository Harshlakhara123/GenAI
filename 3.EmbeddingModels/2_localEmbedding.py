from langchain_huggingface import HuggingFaceEmbeddings

embModel = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "Delhi is the capital of India"

documents = [
    "Delhi is the capital of India",
    "The capital of France is Paris",
    "The capital of Germany is Berlin",
    "The capital of Italy is Rome",
    "The capital of Spain is Madrid",
    ]

vector = embModel.embed_documents(documents)
print(str(vector))
