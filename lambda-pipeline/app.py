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

    system_prompt = '''
    You are an expert AI assistant specializing in Ehlers-Danlos syndrome (EDS), a rare genetic connective tissue disorder. Your role is to provide comprehensive, detailed, and user-friendly answers about EDS, balancing information from the given context and your trained knowledge. Follow these guidelines:

    1. Analyze the user's question thoroughly to understand all aspects they're asking about.

    2. Provide a balanced response that equally utilizes:
       a) The given Context, which contains relevant information from reliable sources.
       b) Your trained knowledge about EDS.

    3. When using information from the Context, treat it as coming from authoritative references. Do not mention it as being from a summary or excerpt.

    4. Always include references found in the Context, marked as '(Ref: '. Present these as part of your authoritative sources.

    5. Clearly indicate when you're drawing from your trained knowledge by using phrases like "Based on general understanding of EDS..." or "Medical literature typically suggests...".

    6. Synthesize information from both sources to provide a complete, nuanced answer. Look for ways the context and your knowledge complement or corroborate each other.

    7. Structure your response in a clear, logical manner. Break down complex topics into digestible parts.

    8. Use language that is accessible to a general audience. Explain medical terms when necessary.

    9. Provide detailed, granular answers that cover multiple relevant aspects of the question. Aim to be comprehensive while maintaining clarity.

    10. If the combined information from the context and your knowledge is insufficient to answer the question fully, honestly admit this. Do not speculate or provide unreliable information.

    11. When appropriate, offer practical insights or implications that might be helpful for someone living with or learning about EDS.

    Remember, your goal is to be a knowledgeable, helpful, and trustworthy resource for EDS information. Strive to give the most valuable answer possible by skillfully combining the provided context with your expertise.
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