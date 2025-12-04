from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load env. variabales.
load_dotenv()

# Model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# prompt template
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variable=['topic']
)
prompt1 = template1.invoke({"topic" : "India vs South Africa Cricket Series - 2025"})
text1 = model.invoke(prompt1)

print("\n"+ str(text1))

# 2nd prompt
template2 = PromptTemplate(
    template="Write a four point concise summary on the following {text}",
    input_variables=['text']
)

prompt2 = template2.invoke({"text": str(text1)})
result = model.invoke(prompt2)

print("******************\n")
print(result.content)

