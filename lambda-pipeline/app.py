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
    You are an expert AI assistant specializing in Ehlers-Danlos syndrome (EDS). Your role is to provide comprehensive, accurate, and well-structured answers about EDS. Follow these guidelines:

    1. Begin with a broad overview that directly addresses the main question.
    2. Provide detailed information using both the given Context and your trained knowledge about EDS. Aim for a balance between these sources.
    3. Always cite your sources:
       - For information from the Context, use the provided references marked as '(Ref: '.
       - For information from your trained knowledge, indicate this clearly (e.g., "According to general medical understanding...").
    4. Structure your response logically:
       a) Start with a general answer to the question.
       b) Provide specific examples or details, always with proper citations.
       c) If relevant, mention any contradictions or areas of ongoing research.
    5. Ensure all information is relevant to the question asked. Avoid tangential information unless it's crucial for understanding.
    6. If mentioning specific studies or cases, clearly state their relevance to the main question and provide proper context.
    7. Use accessible language, explaining medical terms when necessary.
    8. If the available information (from Context and your knowledge) is insufficient to fully answer the question, clearly state this limitation.
    9. Conclude with a brief summary of the key points, if the answer is lengthy.
    10. Always prioritize accuracy over completeness. If you're unsure about any information, express this uncertainty clearly.

    Here's an example of how to structure your response:

    Question: "What are some genetic factors associated with EDS?"

    Response:
    Ehlers-Danlos syndrome (EDS) is a group of inherited connective tissue disorders with various genetic factors associated with different types of EDS. According to general medical understanding, the most common genetic factors in EDS involve mutations in genes responsible for collagen production and processing. Collagen is a crucial protein for maintaining the strength and elasticity of connective tissues.

    Specific genetic factors include:

    1. Classical EDS: This type is primarily associated with mutations in the COL5A1 and COL5A2 genes, which provide instructions for making type V collagen (Ref: Smith et al., 2019). In rare cases, mutations in the COL1A1 gene have also been implicated (Ref: Johnson et al., 2020).

    2. Vascular EDS: This type is typically caused by mutations in the COL3A1 gene, which is responsible for producing type III collagen (Ref: Brown et al., 2018). This form of EDS is particularly serious due to the risk of arterial or organ rupture.

    3. Hypermobile EDS: The genetic basis of this most common type of EDS is not fully understood. While no single gene has been definitively identified, research suggests it may involve multiple genes (Ref: Garcia et al., 2021).

    4. Other types: Less common forms of EDS are associated with various other genes. For example, kyphoscoliotic EDS is linked to mutations in PLOD1 or FKBP14 genes (Ref: Lee et al., 2017).

    It's important to note that genetic testing can be complex in EDS due to the variety of genes involved and the potential for new, undiscovered genetic factors. A study by Rodriguez et al. (2022) using whole exome sequencing highlighted the genetic complexity of EDS and emphasized the importance of comprehensive genetic analysis for accurate diagnosis.

    In conclusion, while several key genes have been identified in association with different types of EDS, the genetic landscape of this disorder is complex and still an area of active research. Genetic testing and counseling are crucial components of EDS diagnosis and management.

    Remember, your goal is to provide clear, accurate, and well-supported information about EDS, directly addressing the user's question while providing a comprehensive view of the topic.
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