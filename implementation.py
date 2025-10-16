import json
from interface import VectorStoreInterface
from pymilvus import utility, connections, Collection, MilvusClient, FieldSchema, CollectionSchema, DataType
from neo4j import GraphDatabase

class MilvusVectorStore(VectorStoreInterface):
    # use in prod
    # def __init__(self, host: str = "localhost", port: int = 19530, collection_name: str = "default"):
    #     self.host = host
    #     self.port = port
    #     self.collection_name = collection_name
    #     self.client = None

    def __init__(self, embedding_dim: int, collection_name: str = "default"):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.client = None
        print("→ Using Milvus Lite for local, serverless vector store.")

    # def connect(self, **kwargs) -> None:
    #     from pymilvus import connections, Collection # type: ignore[import]
    #     connections.connect(alias="default", host=self.host, port=self.port)
    #     self.client = Collection(self.collection_name)
        
    def connect(self, **kwargs) -> None:
        # Milvus Lite will automatically start a server in the background
        connections.connect(alias="default", uri="./milvus_local.db") 
        # self.client = Collection(self.collection_name)
        # 2. Check if the collection exists; if not, create it
        if not utility.has_collection(self.collection_name):
            print(f"Collection '{self.collection_name}' not found. Creating now...")
            
            # Define the schema based on the documentation
            id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
            text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4000)
            embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
            
            schema = CollectionSchema(
                fields=[id_field, text_field, embedding_field],
                description="On-premises document collection",
                enable_dynamic_field=True # Allows flexible metadata
            )
            
            # Create the collection
            self.client = Collection(name=self.collection_name, schema=schema, using='default')
            print("✅ Collection created.")

            # Create an index for the embedding field for efficient searching
            print("→ Creating index...")
            index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
            self.client.create_index(field_name="embedding", index_params=index_params)
            print("✅ Index created.")
        else:
            print(f"→ Found existing collection '{self.collection_name}'.")
            self.client = Collection(self.collection_name)

        # Load the collection into memory for searching
        self.client.load()
        print(f"✅ Collection '{self.collection_name}' loaded into memory.")


    
    def upsert(self, vectors, texts, metadatas):
        # This is the crucial part: zip correctly pairs vector[0] with text[0], etc.
        entities = [
            {"embedding": vec, "text": txt, **meta}
            for vec, txt, meta in zip(vectors, texts, metadatas)
        ]
        result = self.client.insert(entities)
        self.client.flush()
        return result.primary_keys
    
    def search(self, query_embedding, top_k, filter=None):
        results = self.client.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            expr=None,  # adapt for filter expressions
            output_fields=["text", "metadata"]
        )
        return [(r.id, r.distance, {"text": r.entity.get("text"), **r.entity.get("metadata", {})}) for hit in results for r in hit]
    
    def delete(self, ids):
        self.client.delete(expr=f"id in {ids}")

    def close(self):
        from pymilvus import connections # type: ignore[import]
        connections.disconnect(alias="default")


class Neo4jGraphDatabase:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth = (user, password))
        
    def close(self):
        self._driver.close()

    def execute_query(self, query, parameters=None):
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]
        
    def add_triplets(self, triplets):
        """
        Adds a list of (head, relation, tail) triplets to the graph.
        """
        query = """
            UNWIND $triplets AS triplet
            MERGE (h:Entity {name: triplet.head})
            SET h += {type: triplet.head_type}
            
            WITH h, triplet 
            CALL apoc.create.addLabels(h, [triplet.head_type]) YIELD node AS head_node

            WITH head_node, triplet 
            MERGE (t:Entity {name: triplet.tail})
            SET t += {type: triplet.tail_type}
            
            WITH t, head_node, triplet 
            CALL apoc.create.addLabels(t, [triplet.tail_type]) YIELD node AS tail_node

            WITH head_node, tail_node, triplet 
            CALL apoc.create.relationship(head_node, triplet.relation, {}, tail_node) YIELD rel
            
            RETURN count(rel)
            """
        
        with self._driver.session() as session:
            session.run(query, triplets=triplets)

    def run_community_detection(self, graph_name: str = "context-iq", write_property: str = "communityId") -> dict:
        """
        Leiden community detection algo using neo4j GDS and writes community id back to nodes
        """
        with self._driver.session() as session:
            # use entity and relationships to project the graph to gds memory catalog
            send_query = f"""
                CALL gds.graph.project(
                    '{graph_name}',
                    'Entity',
                    {{
                        RELATIONSHIP: {{
                            orientation: 'UNDIRECTED'
                        }}
                    }}
                )
            """
            session.run(send_query)

            # use leiden to write the results back
            leiden_query = f"""
                CALL gds.leiden.write(
                    '{graph_name}',
                    {{
                        writeProperty: '{write_property}'
                    }}
                )
                YIELD communityCount, modularity
            """
            result = session.run(leiden_query)
            summary = result.single().data()

            # clean the in memory projections
            drop_query = f"CALL gds.graph.drop('{graph_name}')"
            session.run(drop_query)

            print(f"leiden complete. found {summary['communityCount']} communities.")
            return {"community_count": summary["communityCount"], "modularity": summary["modularity"]}
    
    def get_community_content(self, communityId: int, limit: int = 25) -> str:
        """
        gets the names of nodes in a specific community for summarization
        """
        query = """
            MATCH(n: Entity)
            WHERE n.communityId = $community_id
            RETURN n.name AS name
            LIMIT $limit
        """

        records = self.execute_query(query, parameters={"community_id": communityId, "limit": limit})
        node_names = [record['name'] for record in records]
        return ", ".join(node_names) 


   
