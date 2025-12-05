# Sequential chains are powerful way to implement
# complex multi-step process. 
# Chains make the implementation simple, modular and easily manageable chunks.

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# load env. variables
load_dotenv()

# model
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Prompts using prompt templates
prompt1 = PromptTemplate(
    template='Generate a 2025-Q3 earnings report of a stock {stock}',
    input_variables=['stock']
)

prompt2 = PromptTemplate(
    template="Highlight 3 important guidances from the following earnings report {report}",
    input_variables=['report']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'stock': 'MSFT'})
print(result)