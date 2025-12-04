from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load env. vairables
load_dotenv()

# model definition.
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Prompt 1 - template
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

# Prompt 2 - prompt template
template2 = PromptTemplate(
    template="Write a four point consise summary from the following {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic': "India vs South Africa cricket series - 2025"})
print(result)