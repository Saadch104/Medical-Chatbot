from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from  pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()
Pinecone_API_key = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = Pinecone_API_key



extracted_data = load_pdf(data ="Data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


pc  = Pinecone(api_key=Pinecone_API_key)
index_name = "medical-chatbot"
# Create a new index
pc.create_index(
    name=index_name,
    dimension=384,  # Dimension of the embeddings
    metric="cosine",  # Similarity metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1",  # Change to your preferred region
    )
    
)




docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
   # namespace="medical-chatbot"
)



