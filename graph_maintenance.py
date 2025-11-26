import os
import google.generativeai as genai
from implementation import Neo4jGraphDatabase, MilvusVectorStore
from sentence_transformers import SentenceTransformer

def summarize_and_embed_communities():
    """
    offline maintanence for 
    1. running community detection on graph
    2. create a summary for each community using llm
    3. embed these summaries and store them in milvus collection
    """

    URI_NEO4J = os.environ.get("URI_NEO4J")
    USER_NEO4J = os.environ.get("USER_NEO4J")
    PASSWORD_NEO4J = os.environ.get("PASSWORD_NEO4J")
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    llm = genai.GenerativeModel('gemini-2.5-flash')
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    graph_store = Neo4jGraphDatabase(URI_NEO4J, USER_NEO4J, PASSWORD_NEO4J)

    # connect to new milvus for summaries
    community_vector_store = MilvusVectorStore(embedding_dim=384, collection_name="community_summaries")
    community_vector_store.connect()

    # run community detection
    print("running community detection")
    graph_store.run_community_detection()

    # summarize and embed each community
    community_ids_result = graph_store.execute_query(
        "MATCH (n:Entity) WHERE n.communityId IS NOT NULL RETURN DISTINCT n.communityId as c_id ORDER BY c_id"
    )

    community_ids = [res["c_id"] for res in community_ids_result]

    print(f"found {len(community_ids)} communities. now embedding.")
    summaries, metadatas = [], []

    for cid in community_ids:
        # get core nodes from graph
        content = graph_store.get_community_content(cid)
        if not content:
            continue
        # use llm for summary  
        prompt = f"summarize the following interconnected topics into a short, descriptive paragraph: {content}"
        summary = llm.generate_content(prompt).text

        summaries.append(summary.strip())
        # store communityid in metadata for later quick retrieval

        metadatas.append({"communityId": cid})
        print(f"    summarized content {cid}")
    
    summary_embeddings = embedding_model.encode(summaries, show_progress_bar = True)

    # upsert embeddings, summaries, and metadata to new milvus
    community_vector_store.upsert(vectors=summary_embeddings, texts=summaries, metadatas=metadatas)

    print("community summarizaiton and embedding complete")
    graph_store.close()
    community_vector_store.close()

if __name__ == "__main__":
    summarize_and_embed_communities()
