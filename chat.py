from implementation import MilvusVectorStore, Neo4jGraphDatabase
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
import google.generativeai as genai
import json


def extract_from_graph(question: str, llm):
    prompt = f"""
        From the user's question below, identify the key entities or concepts.
        Return them as a JSON list of strings.
        
        Example:
        Question: "Tell me about the architecture of the NVIDIA RTX 4090."
        Output: ["NVIDIA RTX 4090", "architecture"]

        Question: "How much memory does it have?"
        Output: ["memory"]

        Provide only the JSON list.

        Question to analyze:
        ---
        {question}
        ---
    """
    try:
        response = llm.generate_content(prompt)
        json_str = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        print(f"Could not parse entities from query: {e}")
        return []
    
def format_graph_results(records):
    if not records:
        print("no specific facts found in graph")
    
    formatted = []

    for record in records:
        try:
            head = record["h"]["name"]
            relation = record["r"]["type"]
            tail = record["t"]["name"]
            formatted.append(f"Fact: ('{head}' -[{relation}]-> '{tail}')")
        except(TypeError, KeyError):
            continue
    return "".join(formatted)



def getGeminiAnswer(question: str, rag_context: list[str], graph_context: str, llm) -> str:
    """
    Generates an answer using both RAG context and KG context.
    """
    rag_context_str = "\n---\n".join([f"Source: {chunk.get('source', 'N/A')}\nContent: {chunk.get('text', '')}" for chunk in rag_context])
    
    prompt = f"""
              You are an expert Q&A assistant. Your goal is to provide accurate, cited answers by synthesizing information from two sources: unstructured text passages and structured knowledge graph facts.
              
              1. First, review the unstructured TEXT PASSAGES.
              2. Second, review the structured KNOWLEDGE GRAPH FACTS.
              3. Answer the user's QUESTION based *only* on the provided information.
              4. For information from text passages, you MUST cite the source (e.g.,).
              5. If the context does not contain the answer, state that you could not find the information. Do not make things up.

              ### TEXT PASSAGES ###
              {rag_context_str}

              ### KNOWLEDGE GRAPH FACTS ###
              {graph_context}

              ### QUESTION ###
              '{question}'

              ### ANSWER ###
              """
    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the response: {e}"

def run_chat_interface(embedding_model, cross_encoder, vector_store, graph_store, llm):
    """Encapsulates the main chat loop."""
    print("\nchat. type 'exit' to quit")
    while True:
        try:
            question = input("\nyour question: ")
            if question.lower() == 'exit':
                break

            # retrieval from milvus
            print("searching vector store")
            question_embedding = embedding_model.encode(question).tolist()
            search_results = vector_store.search(question_embedding, top_k=10)

            if not search_results:
                print("Could not find relevant info.")
                rag_context = []
            else:
                pairs = [[question, result[2].get("text", "")] for result in search_results]
                scores = cross_encoder.predict(pairs)
                scored_results = sorted(zip(scores, search_results),  key=lambda x: x[0], reverse=True)
                final_result = [result for score, result in scored_results[:3]]
                rag_context = [result[2] for result in final_result]
                print("searching knowledge graph")
                entities = extract_from_graph(question, llm)
                graph_context_str = "nothing specific found in knowledge graph"
                if entities:
                    print(f"found entities {entities}")
                    cypher_query = """
                        MATCH (h:Entity)-[r]-(t:Entity)
                        WHERE h.name IN $entities
                        RETURN h, r, t
                        LIMIT 10
                        """
                    graph_results = graph_store.execute_query(cypher_query, parameters={"entities": entities})
                    graph_context_str = format_graph_results(graph_results)
                print("generating answer")
                final_answer = getGeminiAnswer(question, rag_context, graph_context_str, llm)

                print("answer: ")
                print(final_answer)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"error occured: {e}")
                    


def evaluate_rag(embedding_model, cross_encoder, store, golden_set):
    """
    Runs a basic evaluation against a perfect' of questions and ideal answers.
    """
    print("\n--- running eval ---")
    for item in golden_set:
        question = item["question"]
        golden_answer = item["answer"]

        print(f"\nevaluating question: {question}")

        question_embedding = embedding_model.encode(question).tolist()
        search_results = store.search(question_embedding, top_k=10)

        if not search_results:
            generated_answer = "no info found."
        else:
            pairs = [[question, result[2].get("text", "")] for result in search_results]
            scores = cross_encoder.predict(pairs)
            scored_results = sorted(list(zip(scores, search_results)), key=lambda x: x[0], reverse=True)
            final_results = [result for score, result in scored_results[:3]]
            context_for_llm = [result[2] for result in final_results]
            generated_answer = getGeminiAnswer(question, context_for_llm)

        print(f"generated answer:\n{generated_answer}")
        print(f"golden answer (answer key) :\n{golden_answer}")
        print("--------------------")

def main():
    COLLECTION_NAME = "my_document_store_v4"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    URI_NEO4J = os.environ.get("URI_NEO4J")
    USER_NEO4J = os.environ.get("USER_NEO4J")
    PASSWORD_NEO4J = os.environ.get("PASSWORD_NEO4J")


    # Load models
    print("loading models...")
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    llm = genai.GenerativeModel('gemini-1.5-flash-latest')
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)

    # Connect to vector store
    print(f"Connecting to collection: '{COLLECTION_NAME}'")
    vector_store = MilvusVectorStore(embedding_dim=384, collection_name=COLLECTION_NAME)
    vector_store.connect()

    # graph database
    graph_store = Neo4jGraphDatabase(URI_NEO4J, USER_NEO4J, PASSWORD_NEO4J)

    # --- Evaluation Section ---
    # Create a small "golden set" to test your system's performance
    golden_set = [
        {
            "question": "What architecture does the RTX 4090 use?",
            "answer": "The NVIDIA GeForce RTX 4090 GPU is based on the Ada Lovelace architecture."
        },
        {
            "question": "How much memory does the RTX 4090 have?",
            "answer": "The GeForce RTX 4090 comes with 24GB of GDDR6X memory."
        }
    ]
    # To run evaluation, uncomment the line below
    # evaluate_rag(embedding_model, cross_encoder, store, golden_set)

    # chat interface
    run_chat_interface(embedding_model, cross_encoder, vector_store, graph_store, llm)

    graph_store.close()
    vector_store.close()
    print("\nbye!")

if __name__ == "__main__":
    main()