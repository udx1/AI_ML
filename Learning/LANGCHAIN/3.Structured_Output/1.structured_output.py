# How to get LLM output in a structured format?
# The structured output will help the AI application
# to process the data, display in dashboards, store in database etc.

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

# Schema
class Review(TypedDict):
    sentiment:str
    summary:str

# model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# structured model
structured_model = model.with_structured_output(Review)

#prompt
prompt = """
            I like the new laptop.
            But it has too many customer applications that are bit annoying.
            I just dont like it.
        """

result = structured_model.invoke(prompt)
print(result)

# How does unstructured response look?
new_prompt = f" Provide a sentiment and summary from the review '{prompt}'."
result = model.invoke(new_prompt)
print(result.content)

