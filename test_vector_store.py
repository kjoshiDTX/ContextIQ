# # test_vector_store.py
# import traceback
# from implementation import MilvusVectorStore


# def main():
#     store = MilvusVectorStore(
#         host="localhost",    # adjust if your Milvus is elsewhere
#         port=19530,
#         collection_name="default"
#     )

#     print("→ Instantiated MilvusVectorStore")

#     try:
#         store.connect()
#         print("✅ connect() succeeded")
#     except Exception as e:
#         print("❌ connect() failed:")
#         traceback.print_exc()

#     # since we probably don't have real data yet,
#     # we'll just test close()

#     try:
#         store.close()
#         print("✅ close() succeeded")
#     except Exception as e:
#         print("❌ close() failed:")
#         traceback.print_exc()


# if __name__ == "__main__":
#     main()

# test_vector_store.py

import traceback
from implementation import MilvusVectorStore
from pymilvus import utility, connections

def main():
    """
    This script tests the full CRUD (Create, Read, Update, Delete)
    functionality of the MilvusVectorStore.
    """
    collection_name = "test_collection_123"
    store = MilvusVectorStore(embedding_dim=3, collection_name=collection_name)
    inserted_ids = []

    print("→ Running MilvusVectorStore Test Suite...")

    try:
        # 1. Connect using the store's method. This also handles setup.
        store.connect()
        print("✅ Connect: Succeeded")

        # 2. Clean up old collection *after* connecting.
        if utility.has_collection(collection_name):
            # Dropping a collection that is already loaded requires releasing it first.
            store.client.release()
            utility.drop_collection(collection_name)
            print("→ Cleaned up old test collection.")
            # Re-run connect to create the collection fresh.
            store.connect()

        # 3. Upsert Test
        print("\n→ Testing Upsert...")
        test_vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        test_texts = ["This is the first sentence.", "This is the second sentence."]
        test_metadatas = [{"source": "doc1"}, {"source": "doc2"}]
        
        inserted_ids = store.upsert(test_vectors, test_texts, test_metadatas)
        assert len(inserted_ids) == 2
        print("✅ Upsert: Succeeded")

        # 4. Search Test
        print("\n→ Testing Search...")
        query_vector = [1.0, 2.0, 3.0]
        search_results = store.search(query_vector, top_k=1)
        
        assert len(search_results) == 1
        result_id, distance, metadata = search_results[0]
        assert result_id == inserted_ids[0]
        assert distance == 0.0
        assert metadata["text"] == test_texts[0]
        print("✅ Search: Succeeded")

        # 5. Delete Test
        print("\n→ Testing Delete...")
        store.delete(inserted_ids)
        print("✅ Delete: Succeeded")

        # 6. Verify Deletion Test
        print("\n→ Verifying Deletion...")
        post_delete_results = store.search(query_vector, top_k=2)
        assert len(post_delete_results) == 0
        print("✅ Verify Deletion: Succeeded")

    except Exception:
        print("\n❌ A test failed:")
        traceback.print_exc()
    
    finally:
        # Always close the connection using the store's method.
        if store.client:
            store.close()
            print("\n✅ Connection Closed")
        
        print("\n🎉 Test Suite Finished.")


if __name__ == "__main__":
    main()