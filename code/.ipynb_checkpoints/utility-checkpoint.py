from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import tiktoken
import fitz
import json
import numpy as np
import math
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import os
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from joblib import Memory



REPO_ROOT_PATH = Path("../..").resolve()
load_dotenv(REPO_ROOT_PATH / '.gpt_config.env')

memory = Memory(os.path.join(REPO_ROOT_PATH, 'cachegpt'), verbose=0)

class ParseException(Exception):
    pass


if os.environ.get('RESOURCE_ENDPOINT'):
    from openai import AzureOpenAI    
    client = AzureOpenAI(
      api_key = os.environ.get('API_KEY'),  
      api_version = os.environ.get('API_VERSION'),
      azure_endpoint = os.environ.get('RESOURCE_ENDPOINT')
    )
else:
    from openai import OpenAI
    client = OpenAI(api_key = os.environ.get('API_KEY'))


def get_question_formulation_system_prompt(number_of_questions):
    question_formulation_system_prompt = f"""
        You are an excellent Q&A dataset creator, your job is to read the given input text and create {number_of_questions} questions from it based on a passage from the text.
        Make sure that, the passage is big enough.
        You provide your output as a list of JSON objects where the object has the following format:
        {{
        instruction : <created question>
        input : <reference passage from the text, based on which the question is generated>
        output : <answer the question in a descriptive and detailed fashion, based on the reference text from the passage. Do not copy the same passage as the answer>
        }}
    """    
    return question_formulation_system_prompt

@memory.cache
def get_GPT_response(instruction, system_prompt, chat_model_id, temperature=0.3):
    return fetch_GPT_response(instruction, system_prompt, chat_model_id, temperature)

@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(5))
def fetch_GPT_response(instruction, system_prompt, chat_model_id, temperature=0.3):
    response = client.chat.completions.create(        
        temperature=temperature,
        model=chat_model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ]
    )
    if response.choices:
        return response.choices[0].message.content
    else:
        return 'Unexpected response'



def get_gpt():
    # Load the content of .gpt_config.env into the process environmnet(os.environ)
    # If you don't have .gpt_config.env, check the setup steps in the README 
    api_key = os.environ.get('API_KEY')
    if os.environ["PROVIDER"] == "azure":
        os.environ["AZURE_OPENAI_API_KEY"] = os.environ.get('API_KEY')
        os.environ["AZURE_OPENAI_ENDPOINT"] = os.environ.get('RESOURCE_ENDPOINT')
        os.environ["AZURE_OPENAI_API_VERSION"] = os.environ.get('API_VERSION')
        os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = os.environ.get('MODEL_NAME') 
        llm = AzureChatOpenAI(
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            temperature=0.3
        )
        return llm
    elif os.environ["PROVIDER"] == "openai":
        llm = ChatOpenAI(model_name=os.environ.get('MODEL_NAME'), openai_api_key=os.environ.get('API_KEY'))
        return llm
    else:
        raise Exception("The provider set in .gpt_config.env didn't match 'azure' or 'openai', reach out to Andrew or retry the env section of the readme")


def split_text_to_chunks(input_text, max_tokens=4096, encoding_name='cl100k_base'):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(input_text)
    token_splits = np.array_split(tokens, math.ceil(len(tokens)/max_tokens))
    text_chunks = []
    for token_split in token_splits: 
        text_chunks.append(encoding.decode(token_split))
    return text_chunks


def count_token(input_text, encoding_name='cl100k_base'):
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(input_text))


def create_question_formulation_template():
    question_formulation_template = """
        You are an excellent Q&A dataset creator, your job is to read the given input text and create {number_of_questions} questions from it based on a passage from the text.
        Make sure that, the passage is big enough.
        You provide your output as a list of JSON objects where the object has the following format:
        instruction : <created question>
        input : <reference passage from the text, based on which the question is generated>
        output : <answer the question in a descriptive and detailed fashion, based on the reference text from the passage. Do not copy the same passage as the answer>
        
        Here is the input text you have been asked to generate Q&A for:
        {text_input}
    """    
    question_formulation_prompt_template = PromptTemplate(
        input_variables=["text_input", "number_of_questions"], template=question_formulation_template)
    return question_formulation_prompt_template

def create_question_from_file(file, llm, page_index=0, number_of_questions_per_page=2):    
    question_formulation_prompt_template = create_question_formulation_template()
    question_formulation_chain = LLMChain(prompt=question_formulation_prompt_template, llm=llm)
    doc = fitz.open(file)
    qa_list = []
    page_text = ''
    while page_text == '':
        page = doc[page_index]
        page_text = page.get_text()            
        page_index += 1
        if page_index == len(doc)-1:
            break
    if ('Reference' in page_text) | ('Acknowledgment' in page_text):
        raise ParseException(f"Skipped {file} because: ('Reference' in page_text) | ('Acknowledgment' in page_text) was true")
    else:
        out = question_formulation_chain.run(text_input=page_text, number_of_questions=number_of_questions_per_page)
        try:
            return json.loads(out)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            cleaned_out = out[:e.pos]
            return json.loads(cleaned_out)

def create_question_from_file_v2(file, page_index=0, number_of_questions_per_page=2):    
    question_formulation_system_prompt = get_question_formulation_system_prompt(number_of_questions_per_page)
    doc = fitz.open(file)
    qa_list = []
    page_text = ''
    while page_text == '':
        page = doc[page_index]
        page_text = page.get_text()            
        page_index += 1
        if page_index == len(doc)-1:
            break
    if ('Reference' in page_text) | ('Acknowledgment' in page_text):
        raise ParseException(f"Skipped {file} because: ('Reference' in page_text) | ('Acknowledgment' in page_text) was true")
    else:
        page_text = 'Here is the input text you have been asked to generate Q&A for:\n\n' + page_text
        out = get_GPT_response(page_text, question_formulation_system_prompt, os.environ.get('MODEL_NAME'), temperature=0.3)
        try:
            return json.loads(out)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            cleaned_out = out[:e.pos]
            return json.loads(cleaned_out)

def create_question_from_text(text, llm, number_of_questions=2):
    question_formulation_prompt_template = create_question_formulation_template()
    question_formulation_chain = LLMChain(prompt=question_formulation_prompt_template, llm=llm)
    out = question_formulation_chain.run(text_input=text, number_of_questions=number_of_questions)
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        cleaned_out = out[:e.pos]
        return json.loads(cleaned_out)


def create_question_from_text_v2(text, number_of_questions=2):
    question_formulation_system_prompt = get_question_formulation_system_prompt(number_of_questions)
    out = get_GPT_response(text, question_formulation_system_prompt, os.environ.get('MODEL_NAME'), temperature=0.3)
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        cleaned_out = out[:e.pos]
        return json.loads(cleaned_out)




