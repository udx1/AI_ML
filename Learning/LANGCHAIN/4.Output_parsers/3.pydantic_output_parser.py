# How to get a structured LLM Ouput and validate the data types.

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from datetime import date

# load env. variables.
load_dotenv()

# Model definition.
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Data model.
class Company(BaseModel):
    stock:str=Field(description="Company name of the top stock.")
    sticker:str=Field(description="Stock's sticker symbol")
    price:float=Field(description="price of the stock at close.")
    cap:float=Field(gt=1_000_000_0000, description="Market cap of the company should be atleast 1 billion USD.")
    trading_date: date=Field(description="Latest date in year 2025 on which the stock is traded.")

parser = PydanticOutputParser(pydantic_object=Company)
template = PromptTemplate(
    template="""Get the details of the top stock from {category} list. 
             The market cap should be at least {cap} dollars.
             IMPORTANT INSTRUCTION: The trading_date field must be the actual latest trading date. 
             Assume the current date is {current_date}. 
             For the latest stock data, set trading_date to {current_date} in the output JSON.
             
             The output **MUST** be a valid JSON object matching the following format instructions:
             {stock_format}
             """,
    input_variables=['category', 'cap', 'current_date'],
    # Inject the JSON schema instructions here
    partial_variables={'stock_format': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'category': 'S&P', 'cap': 'One Billion USD', 'current_date': '03-Dec-2025'})
print(result)
