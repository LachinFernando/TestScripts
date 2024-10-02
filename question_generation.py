import json
import os
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pinecone import Pinecone


NAMESPACE_KEY = "Shaili"
TEXT_MODEL = "text-embedding-ada-002"
QA_MODEL = "gpt-3.5-turbo"
COMMON_TEMPLATE = """
"Assume you are an expert assisstant in health domain."
"Use the following pieces of context and patient symptoms to provide more questions to the patient to verify the disease he/she has."
"Please do not use data outside the context to generate any questions."
"If you don not know the context, just say that you don't have enough context."
"don't try to make up an answer."
"Please provide questions as a list."
"\n\n"
{context}
"\n\n"
{symptoms}
"\n"
"Helpful questions:   "
"""


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(host=os.environ["INDEX_HOST"])
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_openai_embeddings(text: str):
    response = client.embeddings.create(input=f"{text}", model=TEXT_MODEL)

    return response.data[0].embedding


# function query similar chunks
def query_response(query_embedding, k = 2, namespace_ = NAMESPACE_KEY):
    query_response = index.query(
        namespace=namespace_,
        vector=query_embedding,
        top_k=k,
        include_values=False,
        include_metadata=True,
    )

    return query_response


def content_extractor(similar_data):
    top_values = similar_data["matches"]
    # get the text out
    text_content = [sub_content["metadata"]["text"] for sub_content in top_values]
    return " ".join(text_content)
    

def get_model():
    model = ChatOpenAI(model=QA_MODEL, api_key=os.environ["OPENAI_API_KEY"])
    return model


def question_answering(query_question: str, context_text: str, template: str = COMMON_TEMPLATE):
    prompt = ChatPromptTemplate.from_template(template)
    model = get_model()
    output_parser = StrOutputParser()

    # create the chain
    chain = prompt | model | output_parser

    # get the answer
    answer = chain.invoke({"context": context_text, "symptoms": query_question})

    return answer
    
    
def process_post_request(event):
    params = event["body"]
    query = params["symptoms"]
    print(query)
    
    try:
        # get the query embeddings
        quer_embed_data = get_openai_embeddings(query)
        
        # query the similar chunks
        similar_chunks = query_response(quer_embed_data)
        
        # extract the similar text data
        similar_content = content_extractor(similar_chunks)
        
        # get the answer
        answer = question_answering(query, similar_content)
        
        return answer.split("\n")
        
    except Exception as error:
        print(str(error))
        return None


def lambda_handler(event, context):
    print(event)
    
    if event["httpMethod"] == "POST":
        try:
            response = process_post_request(event)
            if not response:
                message = "Questions generation failed"
                return {
                    'statusCode': 400,
                    'data': "",
                    'error': [message]
                }
            final_response =  {
                    'statusCode': 200,
                    'data': {"questions": response},
                    'error': []
                }
            return final_response
        except Exception as error:
            message = "Unable to extract the transcript: err-{}".format(str(error))
            return {
                'statusCode': 400,
                'data': "",
                'error': [message]
            }
    else:
        return {
                'statusCode': 400,
                'data': "",
                'error': ["Unbounded Error"]
            }
