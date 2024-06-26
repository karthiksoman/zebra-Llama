import requests
from dotenv import load_dotenv
import os
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone
from huggingface_hub import InferenceClient
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

API_URL = os.environ.get('HF_ZEBRA_LLAMA_API_URL')
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

def get_model_response(input_text, temperature: float = 0.7):
    try:
        client = InferenceClient(API_URL)
        response = client.text_generation(input_text, 
                                      max_new_tokens=512,
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
        extracted_context_summary = list(map(lambda x: json.loads(x.metadata['_node_content'])['metadata']['section_summary'], retrieved_doc.matches))
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
    system_prompt = '''
    You are an expert in the rare disease Ehlers-Danlos syndrome (EDS).
    You are supposed to answer the question asked by the user.
    Your response should be grounded on the given Context in the user message.
    Context is a section summary, but your response should NOT mention this is from a summary or section or excerpt, instead mention this is from a reference.
    Always make sure to provide references in your answer.
    You can find the references in the Context marked as '(Ref: '.
    If no context is given, try to answer as accurately as possible. 
    If you don't know the answer, admit that you don't instead of making one up.  
    '''
    if not input_text:
        logging.error("No input text provided")
        return json.dumps({"error": "No input text provided"}), 400
    try:
        rag_context = get_rag_context(input_text)
        formatted_prompt = format_prompt(input_text, rag_context, system_prompt)
        response = get_model_response(formatted_prompt, temperature)
        return json.dumps(response)
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