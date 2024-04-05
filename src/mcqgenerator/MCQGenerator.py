import os
import json
import pandas as pd
import traceback
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback
from src.mcqgenerator.logger import logging
from src.mcqgenerator.utils import read_file, get_table_data

load_dotenv()
openai_key=os.getenv("OPENAI_API_KEY3")

logging.info("Open AI API Key is loaded")

local_llm=ChatOpenAI(openai_api_key=openai_key, model_name="gpt-3.5-turbo", temperature=0.5)

logging.info("LLM is given")

RESPONSE_JSON = {
    "1": {
        "MCQ": "Multiple Choice Question",
        "Options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "2": {
        "MCQ": "Multiple Choice Question",
        "Options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "3": {
        "MCQ": "Multiple Choice Question",
        "Options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
}

TEMPLATE="""
Text: {text}
You're an expert Bible Quiz Maker. Given the above text, it is your job to create a quiz of {number} multiple choice questions for {subject} \
student in {tone} tone. Make sure the questions are not repeated and check all the questions to be conforming the text as well. \
Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
{response_json}

"""

prompt_template= PromptTemplate(
    input_variables=["text","number","subject","tone","response_json"],
    template=TEMPLATE
)

logging.info("Prompt Template 1 is created")

quiz_chain=LLMChain(llm=local_llm, prompt=prompt_template, output_key="quiz", verbose=True)

logging.info("LLM and Prompt Template 1 chain is created")

TEMPLATE2="""
You're an expert English Grammarian and writer. Given a Multiple Choice Quiz for {subject} students. \
You need to evaluate the complexity of the question and given complete analysis of the quix. Only use maximum of 50 words. \
If the quiz is not as per the cognitive and analytical abilities of the students, \
update the quiz questions which needs to be changed and change the tone such that it fits perfectly the abilities of the students.
Quiz_MCQs:
{quiz}

Check the above quiz as an expert English Writer
"""

quiz_evaluation_prompt=PromptTemplate(input_variables=["subject","quiz"], template=TEMPLATE2)

logging.info("Prompt Template 2 is created")

review_chain=LLMChain(llm=local_llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)

logging.info("LLM and Prompt Template 2 chain is created")


generate_evaluate_chain=SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text","number","subject","tone","response_json"],
                                        output_variables=["quiz","review"], verbose=True)

logging.info("Chain of quiz and review chains is created")
