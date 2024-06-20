from pinecone import Pinecone, ServerlessSpec
from utility import client as openai_client
import os


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "eds"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)


def upsert(vectors):
    index.upsert(vectors=vectors)


def query(text):
    response = openai_client.embeddings.create(
        input=text, model="text-embedding-3-small", dimensions=1024
    )
    embeddings = [e.embedding for e in response.data]
    embedding = embeddings[0]
    query_results = index.query(vector=embedding, top_k=5, include_metadata=True)

    return query_results


query("What are MMP-2 and MMP-9?")

# Pinecone response:
# {'matches': [{'id': 'ba64e3cb42025ac8bed3bcb9ce56e691',
#               'metadata': {'document_id': '10.1007/s43032-020-00251-1',
#                            'document_id_type': 'doi',
#                            'document_title': 'Serum Decorin, Biglycan, and '
#                                              'Extracellular Matrix Component '
#                                              'Expression in Preterm Birth',
#                            'embedded_text': 'labor, in which there was no '
#                                             'significant difference. MMP-2 and '
#                                             'MMP-9 are endopeptidases which '
#                                             'play a role in the degradation of '
#                                             'the extracellular matrix and are '
#                                             'important factors in the '
#                                             'remodeling process of human fetal '
#                                             'membranes that occurs throughout '
#                                             'the duration of pregnancy [ 29 ,  '
#                                             '52 ]. Their activity is partially '
#                                             'regulated by tissue inhibitors of '
#                                             'metalloproteinases (TIMPs), and '
#                                             'dysregulation of these enzymes '
#                                             'has been previously linked to '
#                                             'PPROM [ 31 ]. In the amniotic '
#                                             'fluid of women with PPROM, levels '


# if __name__ == __main__:
