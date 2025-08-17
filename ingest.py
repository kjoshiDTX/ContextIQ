import pypdf
import os
import json
import google.generativeai as genai
from implementation import MilvusVectorStore, Neo4jGraphDatabase
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings 
# from langchain_google_genai import ChatGoogleGenerativeAI

def extract_text(pdf_path: str) -> str:
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    print(f"extracted {len(text)} characters")
    return text

def chunk_text_semantic(text: str, embedding_model_name: str) -> list[str]:
    # split text so you have overlapping chunks
    print(f"chunking w semantic chunker: {embedding_model_name}")
    embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)
    text_splitter = SemanticChunker(embedding_function)
    chunks = text_splitter.split_text(text)
    # flat_chunks =  [item for sublist in chunks for item in sublist]
    # print(f"split text into {len(flat_chunks)} semantic chunks")
    # return flat_chunks
    print(f"split into {len(chunks)} semantic chunks")
    return chunks

def get_embeddings(chunks: list[str], model: SentenceTransformer) -> list[list[float]]:
    print("generating embeddings")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print(f"generated {len(embeddings)} embeddings")
    return embeddings.tolist() 

def extract_graph_triplets(text_chunk: str, llm):
    # uses llm to extract triplets from chunks
    prompt = f"""
        From the text below, extract entities and relationships.
        Format the output as a list of JSON objects with "head", "tail", "head_type", "tail_type", and "relation".
        Entity types should be one of the following: [Company, Product, Technology, Person, Organization, Topic].

        Example:
        Text: "The NVIDIA GeForce RTX 4090 GPU uses the Ada Lovelace architecture."
        Output: [
            {{"head": "GeForce RTX 4090", "head_type": "Product", "relation": "USES_ARCHITECTURE", "tail": "Ada Lovelace", "tail_type": "Technology"}},
            {{"head": "GeForce RTX 4090", "head_type": "Product", "relation": "IS_A", "tail": "GPU", "tail_type": "Topic"}}
        ]

        Provide only the JSON list.

        Text to analyze:
        ---
        {text_chunk}
        ---
    """
    try:
        response = llm.generate_content(prompt)
        # only json
        json_str = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        print(f"can't parse bc exception: {e}")
        return [] 

def main():
    PDF_FILE_PATH = "gpu.pdf"
    # CHUNK_SIZE = 1000
    # CHUNK_OVERLAP = 150
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    COLLECTION_NAME = "my_document_store_v4"

    URI_NEO4J = os.environ.get("URI_NEO4J")
    USER_NEO4J = os.environ.get("USER_NEO4J")
    PASSWORD_NEO4J = os.environ.get("PASSWORD_NEO4J")

    # initialize model
    genai.configure(api_key = os.environ["GEMINI_API_KEY"])
    llm = genai.GenerativeModel('gemini-1.5-flash-latest')
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # upsert to vector store and graph db

    vector_store = MilvusVectorStore(embedding_dim=384, collection_name=COLLECTION_NAME)
    graph_store = Neo4jGraphDatabase(URI_NEO4J, USER_NEO4J, PASSWORD_NEO4J)


    # extract text
    document_text = extract_text(PDF_FILE_PATH)

    # chunk text semantically 
    text_chunks = chunk_text_semantic(document_text, EMBEDDING_MODEL)

    # embed
    print(f"loading embedding model '{EMBEDDING_MODEL}'")
    chunk_embeddings = get_embeddings(text_chunks, embedding_model)
    # text_chunk = chunk_text_semantic(document_text, embedding_model)
    # chunk_embedding = get_embeddings(text_chunk, embedding_model)

    # meta data for each chunk
    metadatas = [{"source": PDF_FILE_PATH, "text": chunk} for chunk in text_chunks]
    vector_store.connect()
    # upsert vector store 
    vector_store.upsert(vectors=chunk_embeddings, texts=text_chunks, metadatas=metadatas)
    vector_store.close()
    print("completed vector store insertion\n")

    # ingest into neo4j
    print("ingesting to graphdb")
    total_triplets = 0
    for i, chunk in enumerate(text_chunks):
        print(f"processing chunk {i+1}/{len(text_chunks)} for extraction")
        triplets = extract_graph_triplets(chunk, llm)

        if triplets:
            graph_store.add_triplets(triplets)
            total_triplets += len(triplets)
            print(f"added {len(triplets)} triplets to the graph")

        print(f"completed ingestion with a total of {total_triplets} triplets")


    graph_store.close()
    print("\n Document has been ingested")

if __name__ =="__main__":
    main()


