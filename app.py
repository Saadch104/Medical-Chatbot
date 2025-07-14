from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import ChatCohere
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

import os
from src.prompt import system_prompt

from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

Pinecone_API_key = os.environ.get("PINECONE_API_KEY")
Cohere_API_key = os.environ.get("COHERE_API_KEY")

os.environ["PINECONE_API_KEY"] = Pinecone_API_key
os.environ["COHERE_API_KEY"] = Cohere_API_key

embeddings= download_hugging_face_embeddings()

index_name = "medical-chatbot"

#Load existing index
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name)

#Retrieval Chain Setup
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 6})

llm = ChatCohere(
    temperature=0.7,
    max_tokens=1000,
    cohere_api_key="B8jq4wh9z4JAl0AElKDggMh2bZDCI8YVxRJ9UFXU"  # Keep secure in production!
)

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# --- Chain Construction ---
# 
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
#rag_chain = create_retrieval_chain(retriever=retriever, combine_documents_chain=question_answer_chain)



@app.route('/')
def home(): 
    return render_template('chatbot.html')


@app.route('/ask', methods=['POST'])
def chat():
    msg = request.form['msg']
    print(f"User Input: {msg}")
    response = rag_chain.invoke({"input": msg})
    print(f"Response: {response['answer']}")
    return str(response['answer'])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)###
    


    
    
        
        
    