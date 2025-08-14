import pypdf
from implementation import MilvusVectorStore
from sentence_transformers import SentenceTransformer
# from neo4j import GraphDatabase
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

def main():
    PDF_FILE_PATH = "gpu.pdf"
    # CHUNK_SIZE = 1000
    # CHUNK_OVERLAP = 150
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    COLLECTION_NAME = "my_document_store_v4"

    # extract text
    document_text = extract_text(PDF_FILE_PATH)

    # chunk text semantically 
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    text_chunks = chunk_text_semantic(document_text, EMBEDDING_MODEL)

    # embed
    print(f"loading embedding model '{EMBEDDING_MODEL}'")
    chunk_embeddings = get_embeddings(text_chunks, embedding_model)
    # text_chunk = chunk_text_semantic(document_text, embedding_model)
    # chunk_embedding = get_embeddings(text_chunk, embedding_model)

    # upsert to vector store
    store = MilvusVectorStore(embedding_dim=384, collection_name=COLLECTION_NAME)
    store.connect()

    # meta data for each chunk
    metadatas = [{"source": PDF_FILE_PATH, "text": chunk} for chunk in text_chunks]

    # perform upsert
    store.upsert(
        vectors = chunk_embeddings, 
        texts = text_chunks,
        metadatas = metadatas, 
    )

    store.close()
    print("\n Document has been ingested")

if __name__ =="__main__":
    main()


def extract_graph(text:str, llm):
    """
    Uses an LLM to extract knowledge graph triplets (head, relation, tail)
    from a piece of text.
    """
    print("getting graph w llm")

    prompt = ChatPromptTemplate("""
        From the text below, extract entities and the relationships between them.
        Format the output as a list of JSON objects, where each object has a 'head', 'relation', and 'tail'.
        The 'head' and 'tail' are the entities, and 'relation' is the connection between them.

        Example:
        Text: "The NVIDIA GeForce RTX 4090 GPU uses the Ada Lovelace architecture."
        Output: [
            {{"head": "GeForce RTX 4090", "relation": "USES_ARCHITECTURE", "tail": "Ada Lovelace"}},
            {{"head": "GeForce RTX 4090", "relation": "IS_A", "tail": "GPU"}}
        ]

        Do not add any preamble or explanation, just the JSON list.

        Text to analyze:
        {text_chunk}
        """
    )
    
    chain = prompt | llm

    return chain.invoke({"text_chunk" : text})