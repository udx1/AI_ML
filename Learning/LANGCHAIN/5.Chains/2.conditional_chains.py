# Conditional chain allows to invoke different chains based on the conditionn.

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_classic.schema.runnable import RunnableBranch, RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment:Literal['positive', 'negative']=Field(description="The sentiment of the feedback / review")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Determine the sentiment from the following feedback / review {feedback} into positive or negative, the output should be in the following format {response_format}",
    input_variables=['feedback'],
    partial_variables={'response_format':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 =  PromptTemplate(
    template="Write an appropriate concise response for this positive feedback {feedback}",
    input_variables=['feedback']
    )

prompt3 =  PromptTemplate(
    template="Write an appropriate concise response for this negative feedback {feedback}",
    input_variables=['feedback']
    )

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser1),
    (lambda x: x.sentiment == 'negative', prompt3| model | parser1),
    RunnableLambda(lambda x: "No valid sentiment found.")
)

chain = classifier_chain | branch_chain 
result = chain.invoke(
    {'feedback' : 'The Bahubali movie is really not good.'}
)
print(result)