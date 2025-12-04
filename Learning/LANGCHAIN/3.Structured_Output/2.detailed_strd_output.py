# How to get more detailed output from LLM in a structured way?

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

import pprint
#import json

# Load keys from env file.
load_dotenv()

# Define the output structure
class Review(TypedDict):
    key_themes: Annotated[list[str], "Must provide important concepts discussed in the review."]
    sentiment: Annotated[str, "Must provide the review sentiment strictly either positive or negative."]
    summary: Annotated[str, "Must summarize the review in not more than  3-4 bulleted points."]
    competitor: Annotated[Optional[str], "Optionally provide one primary competitor name."]
    pros: Annotated[Optional[list[str]], "Optionally highlight the pros from the review."]
    cons: Annotated[Optional[list[str]], "Optionally highlight the cons from the review."]

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
I have other Worx tools. This was a good purchase.
"""

result = structured_model.invoke(prompt)
#print(result)

# Print in a readable format.
pprint.pprint(result)

#result_dict = json.loads(str(result))
#clean_result = json.dumps(
#    result_dict,
#    indent=4
#)
#print(clean_result)