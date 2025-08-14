from implementation import MilvusVectorStore
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
import google.generativeai as genai


def getGeminiAnswer(question: str, context_chunks: list[str]) -> str:
    """
    Uses the Gemini API to generate an answer based on the user's question and retrieved context.
    """
    # It's a good practice to use a specific model version
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    context_str = "\n---\n".join([f"Source: {chunk.get('source', 'N/A')}\nContent: {chunk.get('text', '')}" for chunk in context_chunks])
    
    # Construct a detailed prompt for the LLM
    prompt = f"""
              You are an expert Q&A assistant for an internal organization. Your tone should be helpful and conversational.
              Answer the user's QUESTION based *only* on the provided CONTEXT.
              For every piece of information you provide, you MUST cite the source from the metadata. For example: "The RTX 4090 uses the Ada Lovelace architecture."
              If the CONTEXT does not contain the answer, you MUST respond with "I'm sorry, but I could not find an answer to that question in the provided documents." Do not make up information.

              CONTEXT:
              {context_str}

              QUESTION: '{question}'

              ANSWER:
              """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the response: {e}"

def run_chat_interface(embedding_model, cross_encoder, store):
    """Encapsulates the main chat loop."""
    print("\nReady to chat, type 'exit' to quit")
    while True:
        try:
            question = input("\nType your question: ")
            if question.lower() == 'exit':
                break

            # 1. Initial Retrieval from Milvus (get more candidates)
            question_embedding = embedding_model.encode(question).tolist()
            # We get more results initially (e.g., 10) to give the re-ranker more to work with
            search_results = store.search(question_embedding, top_k=10)

            if not search_results:
                print("Could not find relevant info.")
                continue

            # 2. Re-ranking with Cross-Encoder
            print("Re-ranking retrieved documents for relevance...")
            # The cross-encoder takes pairs of [question, context]
            pairs = [[question, result[2].get("text", "")] for result in search_results]
            scores = cross_encoder.predict(pairs)

            # Combine results with their new scores
            scored_results = list(zip(scores, search_results))
            # Sort by the new score in descending order
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # 3. Prepare final context from top re-ranked results
            # We'll use the top 3 re-ranked results for the final context
            final_results = [result for score, result in scored_results[:3]]
            context_for_llm = [result[2] for result in final_results] # Extracting metadata dict

            print("Getting answer from Gemini...")
            final_answer = getGeminiAnswer(question, context_for_llm)
            
            print("\n✅ Answer:")
            print(final_answer)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")

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

    # Load models
    print("loading models...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)

    # Connect to vector store
    print(f"Connecting to collection: '{COLLECTION_NAME}'")
    store = MilvusVectorStore(embedding_dim=384, collection_name=COLLECTION_NAME)
    store.connect()

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

    # --- Chat Interface Section ---
    run_chat_interface(embedding_model, cross_encoder, store)

    store.close()
    print("\nbye!")

if __name__ == "__main__":
    main()