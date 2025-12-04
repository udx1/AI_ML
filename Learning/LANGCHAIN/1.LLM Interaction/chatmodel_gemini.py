from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Import wantings to filter them out.
import warnings
warnings.filterwarnings("ignore")

# Load keys from env file.
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt ="What is the capital of Pennsylvania?"
result = model.invoke(prompt)
print(result.content)


