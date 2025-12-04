from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from dotenv import load_dotenv

# Load keys from env file
load_dotenv()

#Model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful {domain} expoert."),
    ("human", "Explain in simple terms, how {domain} shapes {topic} in future.")
])

prompt = chat_template.invoke({
    "domain" : "quantum computing",
    "topic" : "AI"
})

result = model.invoke(prompt)
print(result.content)