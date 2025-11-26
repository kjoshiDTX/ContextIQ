# import pypdf
# import os
import json
# import google.generativeai as genai
# from implementation import MilvusVectorStore, Neo4jGraphDatabase
# from sentence_transformers import SentenceTransformer
# from neo4j import GraphDatabase
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_experimental.text_splitter import SemanticChunker
print("starting import")
import os
print("import pypdf"); import pypdf
print("import genai"); import google.generativeai as genai
print("import implementation"); from implementation import MilvusVectorStore, Neo4jGraphDatabase
print("import sentence_transformers"); from sentence_transformers import SentenceTransformer
print("import semanticchunker"); from langchain_experimental.text_splitter import SemanticChunker
#from langchain_community.embeddings import HuggingFaceEmbeddings 
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
    ##################################################################
    # embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)
    # text_splitter = SemanticChunker(embedding_function)
    embedding_model = SentenceTransformer(embedding_model_name)

    class Wrapper:
        def embed_documents(self, docs):
            return embedding_model.encode(docs).tolist()
        def embed_query(self, doc):
            return embedding_model.encode([doc])[0].tolist()

    text_splitter = SemanticChunker(Wrapper())
    ##################################################################
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

def extract_graph_triplets(text_chunk: list[str], llm):
    # uses llm to extract triplets from chunks
    formatted_chunks = ""
    for i, chunk in enumerate(text_chunk):
        formatted_chunks += f"\n-- chunk {i + 1} -- \n{chunk}\n"
    prompt = f"""
        From the text chunks below, extract entities and relationships.
        Format the output as a single JSON list, where each object has "head", "tail", "head_type", "tail_type", "relation",
        and a "source_chunk" number.
        Entity types should be one of: [Company, Product, Technology, Person, Organization, Topic].
        IMPORTANT: Entities must be specific proper nouns or distinct technical terms (e.g., "GeForce RTX 4090", "Ada Lovelace").
        DO NOT extract generic terms, verbs, or adjectives as entities (e.g., "processing", "thousands", "powerful", "architecture", "generation").
        Relationship types MUST be one of: [USES, PRODUCED_BY, HAS_PART, IS_A, RELATED_TO, COMPETES_WITH].
        If the relationship does not fit any of these, use RELATED_TO.

        Example:
        Text:
        --- CHUNK 1 ---
        The NVIDIA GeForce RTX 4090 GPU uses the Ada Lovelace architecture.
        --- CHUNK 2 ---
        The RTX 4090 was released by NVIDIA.
        
        Output: [
            {{"head": "GeForce RTX 4090", "head_type": "Product", "relation": "USES", "tail": "Ada Lovelace", "tail_type": "Technology", "source_chunk": 1}},
            {{"head": "GeForce RTX 4090", "head_type": "Product", "relation": "PRODUCED_BY", "tail": "NVIDIA", "tail_type": "Company", "source_chunk": 2}}
        ]

        Provide only the single JSON list.

        Text chunks to analyze:
        ---
        {formatted_chunks}
        ---
    """
    try:
        print("send batch request to llm")
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
    llm = genai.GenerativeModel('gemini-2.5-flash')
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # upsert to vector store and graph db

    vector_store = MilvusVectorStore(embedding_dim=384, collection_name=COLLECTION_NAME)
    graph_store = Neo4jGraphDatabase(URI_NEO4J, USER_NEO4J, PASSWORD_NEO4J)


    # extract text
    document_text = extract_text(PDF_FILE_PATH)

    # chunk text semantically 
    text_chunks = chunk_text_semantic(document_text, EMBEDDING_MODEL)[:5]

    # embed
    print(f"loading embedding model '{EMBEDDING_MODEL}'")
    ################################################################################################### CHANGE FOR PROD #################################
    chunk_embeddings = get_embeddings(text_chunks, embedding_model)
    # text_chunk = chunk_text_semantic(document_text, embedding_model)
    # chunk_embedding = get_embeddings(text_chunk, embedding_model)

    # meta data for each chunk
    metadatas = [{"source": PDF_FILE_PATH, "text_chunk_index": i} for i, chunk in enumerate(text_chunks)]
    vector_store.connect()
    # upsert vector store 
    vector_store.upsert(vectors=chunk_embeddings, texts=text_chunks, metadatas=metadatas)
    vector_store.close()
    print("completed vector store insertion\n")

    # ingest into neo4j
    print("ingesting to graphdb")
    
    # 1. Call the new batch function ONCE, outside any loops.
    #    This sends all chunks to the LLM in a single API call.
    print("Sending one large batch request to LLM for graph extraction...")
    all_triplets = extract_graph_triplets(text_chunks, llm) 
    
    # 2. Add all triplets to the graph in ONE database transaction
    if all_triplets:
        graph_store.add_triplets(all_triplets)
        total_triplets = len(all_triplets)
        print(f"Successfully added {total_triplets} triplets to the graph in a single batch.")
    else:
        total_triplets = 0
        print("No triplets were extracted from the batch.")

    print(f"completed ingestion with a total of {total_triplets} triplets")


    graph_store.close()
    print("\n Document has been ingested")

if __name__ =="__main__":
    main()