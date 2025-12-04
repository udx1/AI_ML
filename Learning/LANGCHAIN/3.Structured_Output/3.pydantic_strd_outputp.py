# How to create a detailed structure data with strict data types, default values, 
# optional values with strict details. 
# This will help AI to produce consistent structured, type safe and fully compatible python applications.
# Very useful in dealing with long complex outputs.

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing  import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field
import pprint

# load keys
load_dotenv()

# Data model
class Review(BaseModel):
    key_themes: list[str]=Field(description="Write down the 3-4 themes discussed in the review.")
    summary:str=Field(description="A breif summary from the review.")
    sentiment:Literal["Positive", "Negative"]=Field(description="Write down the sentiment in either positive or negative.")
    name:Optional[str]=Field(description="Name of the reviewer.")
    price:Optional[float]=Field(description="Write down the product price from the review.")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
structured_model = model.with_structured_output(Review)

prompt = """
I have had this for a while and finally used it today.
The electric shovel part of this works great. 
It does a nice job of throwing the snow far enough to clear sidewalks 
and even larger areas in lower snow amounts. 
I cleared a double drive today with a 3" snow fall. 
It's a little heavier than I thought but still very usable. 
Consider the weight if you have used other power shovels that run off other power sources.
I initially had problems with the batteries as they didn't seem to take a charge. 
I cleaned the contacts but still had the problem. 
Finally I pushed the button on each battery to see how charged 
they were both showed a full despite what the lights on the charger said. 
After pushing this button both batteries suddenly allowed the green light on 
the charger to also show a full charge. 
If you have a similar problem try this button. 
Maybe it corrects some internal problem. 
Following that the batteries work great and still had most of their charge 
after all I did with this tool today. 
I have other Worx tools. This was a good purchase at $299.
Reviewed by Uday.
"""

result = structured_model.invoke(prompt)
print("\n")

# Print in a readable format.
pprint.pprint(result)



