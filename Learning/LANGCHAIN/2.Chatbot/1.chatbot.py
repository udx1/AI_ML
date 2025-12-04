# Import libraries
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Env 
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Create model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Create chat_history list
chat_history = [
    SystemMessage(content="You are a helpful AI Assistant.")
]

while True:
    user_input = input("You:")
    chat_history.append(HumanMessage(content=user_input))
    #Exit if user input is exit.
    if user_input.lower() == "exit":
        break

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("Bot:", result.content)

print(chat_history)


