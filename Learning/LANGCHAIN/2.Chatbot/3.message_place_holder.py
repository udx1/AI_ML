# Message Place Holder helps to dynamically insert the 
# converation history into prompt
# while preserving the conversation roles (Human, System, AI)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

# Load keys
load_dotenv()

# model 
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# ChatPromptTemplate
chat_template = ChatPromptTemplate([
    ("system", "You are a helpful customer support agent. For now provide a funny answer."),
    MessagesPlaceholder(variable_name="chat_history"),
    "human", "{query}"
])

# load chat history
chat_history = []
with open("chatbot_history.txt") as file:
    chat_history.extend(file.readlines())

prompt = chat_template.invoke(
    {
        "chat_history": chat_history,
        "query": "Where is my refund?" 
    }
)

result = model.invoke(prompt)
print(result.content)