import os
import json
import pandas as pd
import traceback
from langchain_community.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain_community.callbacks import get_openai_callback
#from langchain.callbacks import get_openai_callback
from src.mcqgenerator.logger import logging
from src.mcqgenerator.utils import read_file, get_table_data
import streamlit as st
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain

logging.info("Opening Response.json")

#loading json file
with open(r'Respone.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

logging.info("Give Title")

#creating a title for the app
st.title("MCQs Creator Application with Langchain")

logging.info("Taking User Inputs")

#Create a form using st.form
with st.form("user_inputs"):
    #File upload
    uploaded_file=st.file_uploader("Upload a PDF or txt file")

    #Input fields - Number
    mcq_count=st.number_input("No of MCQs", min_value=3, max_value=50)

    #Subject
    subject=st.text_input("Insert Subject", max_chars=20)

    #Quiz Tone
    tone=st.text_input("Complexity level of Questions", max_chars=20, placeholder="Easy")

    #Add button
    button=st.form_submit_button("Create MCQs")

    #Check if the button is clicked and all fields have input

    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("loading..."):
            try:
                text=read_file(uploaded_file)
                with get_openai_callback() as cb:
                    response=generate_evaluate_chain(
                        {
                        "text": text,
                        "number": mcq_count,
                        "subject": subject,
                        "tone": tone,
                        "response_json": json.dumps(RESPONSE_JSON)
                        }
                    )
                    #Count tokens and the cost of API call
                    logging.info("Tokens Summary:")
                    print(f"Total Tokens:{cb.total_tokens}")
                    print(f"Prompt Tokens:{cb.prompt_tokens}")
                    print(f"Completion Tokens:{cb.completion_tokens}")
                    print(f"Total Cost:{cb.total_cost}")
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error with response")

            else:
                #Extract the quiz data from the response
                quiz=response.get("quiz", None)
                quiz=quiz.strip("RESPONSE_JSON = ")
                print(quiz)
                if quiz is not None:
                    table_data=get_table_data(quiz)
                    if table_data is not None:
                        df=pd.DataFrame(table_data)
                        df.index=df.index+1
                        st.table(df)
                        #Display the review in the text box as well
                        st.text_area(label="Review", value=response["review"])
                    else:
                        st.error("Error in the table data")
                else:
                    st.write(response)
