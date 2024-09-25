import requests
from dotenv import load_dotenv
import os
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone
from huggingface_hub import InferenceClient
import json
import logging
from groq import Groq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

API_URL = os.environ.get('HF_ZEBRA_LLAMA_API_URL')
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
def get_groq_response(instruction, system_prompt, chat_model, temperature=0.3):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in the environment variables")
    groq_client = Groq(
        api_key=GROQ_API_KEY,
    )
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ],
        model=chat_model,
        temperature=temperature
    )
    if chat_completion.choices:
        return chat_completion.choices[0].message.content
    else:
        return 'Unexpected response'


def get_model_response(input_text, temperature: float = 0.7):
    try:
        client = InferenceClient(API_URL)
        response = client.text_generation(input_text, 
                                      max_new_tokens=1024,
                                      stop_sequences=['End of response'],
                                      stream=False,
                                      temperature=temperature
                                     )
        return response
    except requests.RequestException as e:
        logging.error(f"Failed to get model response: {str(e)}")
        return None

def get_rag_context(query, top_k=5):
    try:
        embed_model = OpenAIEmbedding(
            model='text-embedding-ada-002',
            api_key=os.environ.get('HACKATHON_API_KEY'),
        )
        Settings.embed_model = embed_model
        pc = Pinecone(api_key=os.environ.get('ANDREW_API_KEY'))
        pinecone_index = pc.Index(os.environ.get('RAG_PINECONE_INDEX'))
        query_embedding = embed_model.get_text_embedding(query)
        retrieved_doc = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        # extracted_context_summary = list(map(lambda x: json.loads(x.metadata['_node_content'])['metadata']['section_summary'], retrieved_doc.matches))
        extracted_context_summary = list(map(lambda x: json.loads(x.metadata['_node_content'])['metadata']['text'], retrieved_doc.matches))
        provenance = list(map(lambda x: x.metadata['c_document_id'], retrieved_doc.matches))
        context = ''
        for i in range(top_k):
            context += extracted_context_summary[i] + '(Ref: ' + provenance[i] + '). '
        return context
    except Exception as e:
        logging.error(f"Failed to retrieve or process context: {str(e)}")
        return "Error retrieving context."

def format_prompt(user_message, rag_context, system_prompt):
    formatted_prompt = f'''
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    
    Context: {rag_context}
    User message: {user_message}
    Always make sure to provide references in your answer. You can find the references in the Context marked as '(Ref: '. After providing references, print 'End of response'<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    '''
    return formatted_prompt

def inference(input_text, temperature: float = 0.7):
    # system_prompt = '''
    # You are an expert in the rare disease Ehlers-Danlos syndrome (EDS).
    # You are supposed to answer the question asked by the user.
    # Your response should be grounded on the given Context in the user message.
    # Context is a section summary, but your response should NOT mention this is from a summary or section or excerpt, instead mention this is from a reference.
    # Always make sure to provide references in your answer.
    # You can find the references in the Context marked as '(Ref: '.
    # If no context is given, try to answer as accurately as possible. 
    # If you don't know the answer, admit that you don't instead of making one up.  
    # '''

    # system_prompt = '''
    # You are an expert in the rare disease Ehlers-Danlos syndrome (EDS).
    # Start with a broad overview that directly addresses the question asked by the user.
    # Then, you are supposed to answer the question asked by the user in a detailed and comprehensive fashion based on your internal knowledge and the Context provided in the user message.
    # This is important: Always make sure to provide references/citations in your answer.
    # You can find the references in the Context marked as '(Ref: '. 
    # If you don't know the answer, admit that you don't instead of making one up.  
    # '''

    system_prompt = '''
    You are an expert AI assistant specializing in Ehlers-Danlos syndrome (EDS). Your role is to provide comprehensive, accurate, and well-structured answers about EDS. Try to give your response in an elegant Markdown format and in multiple paragraphs. Follow these guidelines:

    1. In the first paragraph, begin with a broad overview that directly addresses the main question.
    2. In the second paragraph, provide detailed information mainly by using the given Context. Also use your trained knowledge about EDS to supplement the assertions. Aim for a balance between these sources.
    3. Answer in multiple paragraphs and be comprehensive in your answer
    4. Structure your response logically:
       a) Start with a general answer to the question.
       b) Provide specific examples or details, always with proper citations (use the provided references marked as '(Ref: ').
       c) If relevant, mention any contradictions or areas of ongoing research.
    5. If mentioning specific studies or cases, clearly state their relevance to the main question and provide proper context.
    6. In the last paragraph, conclude with a brief summary of the key points, if the answer is lengthy.    
    IMPORTANT: If you encounter a question that is not related to EDS disease, then state that in your response and do not answer that question.
    
    '''

    if not input_text:
        logging.error("No input text provided")
        return json.dumps({"error": "No input text provided"}), 400
    try:
        rag_context = get_rag_context(input_text)
        formatted_prompt = format_prompt(input_text, rag_context, system_prompt)
        # response = get_model_response(formatted_prompt, temperature)

        # Groq layer (this is a temporary fix, until the huggingface issue is resolved)
        instruction_prompt = f'''
        User message: {input_text}
        Context : {rag_context}
        Always make sure to provide references in your answer. You can find the references in the Context marked as '(Ref: '.
        '''
        response = get_groq_response(instruction_prompt, system_prompt, "llama3-8b-8192", temperature=temperature)
        
        return json.dumps(response)
    except ValueError as ve:
        logging.error(f"Configuration error: {str(ve)}")
        return json.dumps({"error": str(ve)}), 500
    except Exception as e:
        logging.error(f"Error during inference: {str(e)}")
        return json.dumps({"error": str(e)}), 500

def lambda_handler(event, context):
    logging.info("Lambda function has started execution.")
    logging.info(f"Event received: {event}")
    try:
        body = json.loads(event['body'])
        input_text = body.get('text', None)
        temperature = body.get('temperature', 0.7)
        response = inference(input_text, temperature)
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": response
        }
    except Exception as e:
        logging.error(f"Error handling lambda event: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({"error": str(e)})
        }