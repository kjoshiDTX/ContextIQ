"""Neo4j knowledge graph operations."""

import logging
from typing import Optional

from app.core.database import get_neo4j

logger = logging.getLogger(__name__)

# Valid entity and relation types
ENTITY_TYPES = ["Company", "Product", "Technology", "Person", "Organization", "Topic", "Concept"]
RELATION_TYPES = ["USES", "PRODUCED_BY", "HAS_PART", "IS_A", "RELATED_TO", "COMPETES_WITH",
                  "WORKS_WITH", "BELONGS_TO", "DEPENDS_ON"]


def store_triplets(triplets: list[dict], document_id: str) -> int:
    """Store extracted triplets as entities and relationships in Neo4j.

    Uses a single UNWIND query instead of N individual writes for efficiency.
    """
    if not triplets:
        return 0

    driver = get_neo4j()

    # Normalize and validate all triplets before sending to Neo4j
    prepared = []
    for t in triplets:
        head = t.get("head", "").strip()
        tail = t.get("tail", "").strip()
        if not head or not tail:
            continue
        prepared.append({
            "head": head,
            "tail": tail,
            "head_type": t.get("head_type", "Concept"),
            "tail_type": t.get("tail_type", "Concept"),
            "relation": t.get("relation", "RELATED_TO").upper().replace(" ", "_"),
            "source_chunk": t.get("source_chunk", ""),
            "doc_id": document_id,
        })

    if not prepared:
        return 0

    with driver.session() as session:
        result = session.run(
            """
            UNWIND $triplets AS t
            MERGE (h:Entity {name: t.head})
              ON CREATE SET h.type = t.head_type, h.created_at = datetime()
              ON MATCH SET h.type = CASE WHEN h.type IS NULL THEN t.head_type ELSE h.type END
            MERGE (tail:Entity {name: t.tail})
              ON CREATE SET tail.type = t.tail_type, tail.created_at = datetime()
              ON MATCH SET tail.type = CASE WHEN tail.type IS NULL THEN t.tail_type ELSE tail.type END
            MERGE (h)-[r:RELATES_TO {type: t.relation}]->(tail)
              ON CREATE SET r.source_document = t.doc_id,
                            r.source_chunk = t.source_chunk,
                            r.created_at = datetime()
            RETURN count(*) AS stored
            """,
            triplets=prepared,
        )
        stored = result.single()["stored"]

    logger.info(f"Stored {stored}/{len(triplets)} triplets for document {document_id}")
    return stored


def store_journal_triplets(triplets: list[dict], entry_id: str, user_id: str) -> int:
    """Store journal-extracted triplets as JournalConcept nodes in Neo4j.

    Unlike document triplets (global Entity nodes), journal nodes are scoped
    per-user via the user_id property. After storing, cross-connects to matching
    global Entity nodes are created automatically.
    """
    if not triplets:
        return 0

    driver = get_neo4j()

    prepared = []
    for t in triplets:
        head = t.get("head", "").strip()
        tail = t.get("tail", "").strip()
        if not head or not tail:
            continue
        prepared.append({
            "head": head,
            "tail": tail,
            "head_type": t.get("head_type", "Concept"),
            "tail_type": t.get("tail_type", "Concept"),
            "relation": t.get("relation", "RELATED_TO").upper().replace(" ", "_"),
        })

    if not prepared:
        return 0

    with driver.session() as session:
        result = session.run(
            """
            UNWIND $triplets AS t
            MERGE (h:JournalConcept {name: t.head, user_id: $user_id})
              ON CREATE SET h.type = t.head_type, h.created_at = datetime()
            MERGE (tail:JournalConcept {name: t.tail, user_id: $user_id})
              ON CREATE SET tail.type = t.tail_type, tail.created_at = datetime()
            MERGE (h)-[r:RELATES_TO {type: t.relation}]->(tail)
              ON CREATE SET r.source_journal = $entry_id,
                            r.user_id = $user_id,
                            r.created_at = datetime()
            RETURN count(*) AS stored
            """,
            triplets=prepared,
            user_id=user_id,
            entry_id=entry_id,
        )
        stored = result.single()["stored"]

        # Cross-connect journal concepts to matching document entity nodes
        session.run(
            """
            MATCH (j:JournalConcept {user_id: $user_id})
            MATCH (e:Entity)
            WHERE toLower(j.name) = toLower(e.name)
               OR toLower(e.name) CONTAINS toLower(j.name)
            MERGE (j)-[:CONNECTS_TO]->(e)
            """,
            user_id=user_id,
        )

    logger.info(f"Stored {stored} journal triplets for entry {entry_id}")
    return stored


def get_graph_context(entities: list[str], limit: int = 10) -> list[dict]:
    """Search the knowledge graph for relationships involving given entities.

    Two-phase retrieval:
    1. Find Leiden community IDs for matched entities (fuzzy CONTAINS).
    2. Fetch all triplets within those communities — surfaces semantically
       related nodes that exact-match would miss.
    Falls back to direct fuzzy match if no community IDs are assigned yet.
    """
    if not entities:
        return []

    driver = get_neo4j()
    results = []

    with driver.session() as session:
        # Phase 1: resolve community IDs for matched entities
        community_result = session.run(
            """
            MATCH (n:Entity)
            WHERE any(e IN $entities WHERE
                toLower(n.name) CONTAINS toLower(e) OR
                toLower(e) CONTAINS toLower(n.name)
            )
            AND n.communityId IS NOT NULL
            RETURN DISTINCT n.communityId AS community_id
            """,
            entities=entities,
        )
        community_ids = [r["community_id"] for r in community_result]

        if community_ids:
            # Phase 2: expand to all triplets within matched communities
            logger.info(f"Graph context: community-scoped search across {len(community_ids)} communities")
            result = session.run(
                """
                MATCH (h:Entity)-[r:RELATES_TO]-(t:Entity)
                WHERE h.communityId IN $community_ids OR t.communityId IN $community_ids
                RETURN DISTINCT
                    h.name AS head,
                    h.type AS head_type,
                    r.type AS relation,
                    t.name AS tail,
                    t.type AS tail_type,
                    h.user_context AS head_context
                LIMIT $limit
                """,
                community_ids=community_ids,
                limit=limit,
            )
        else:
            # Fallback: no community IDs assigned yet, use direct fuzzy match
            logger.info("Graph context: fallback to direct fuzzy match (no community IDs assigned)")
            result = session.run(
                """
                MATCH (h:Entity)-[r:RELATES_TO]-(t:Entity)
                WHERE any(e IN $entities WHERE
                    toLower(h.name) CONTAINS toLower(e) OR
                    toLower(e) CONTAINS toLower(h.name) OR
                    toLower(t.name) CONTAINS toLower(e) OR
                    toLower(e) CONTAINS toLower(t.name)
                )
                RETURN DISTINCT
                    h.name AS head,
                    h.type AS head_type,
                    r.type AS relation,
                    t.name AS tail,
                    t.type AS tail_type,
                    h.user_context AS head_context
                LIMIT $limit
                """,
                entities=entities,
                limit=limit,
            )

        for record in result:
            entry = {
                "head": record["head"],
                "head_type": record["head_type"],
                "relation": record["relation"],
                "tail": record["tail"],
                "tail_type": record["tail_type"],
            }
            if record.get("head_context"):
                entry["user_context"] = record["head_context"]
            results.append(entry)

    logger.info(f"Graph context returned {len(results)} triplets")
    return results


def store_conversation_triplets(triplets: list[dict], session_id: str, user_id: str) -> int:
    """Store Claude conversation-derived triplets as ConversationConcept nodes.

    Scoped per-user like JournalConcept nodes but with conversation-specific
    entity types (Insight, Decision, Observation, etc.).
    Auto-cross-connects to matching Entity nodes in the document graph.
    """
    if not triplets:
        return 0

    driver = get_neo4j()

    prepared = []
    for t in triplets:
        head = t.get("head", "").strip()
        tail = t.get("tail", "").strip()
        if not head or not tail:
            continue
        prepared.append({
            "head": head,
            "tail": tail,
            "head_type": t.get("head_type", "Insight"),
            "tail_type": t.get("tail_type", "Insight"),
            "relation": t.get("relation", "RELATED_TO").upper().replace(" ", "_"),
        })

    if not prepared:
        return 0

    with driver.session() as session:
        result = session.run(
            """
            UNWIND $triplets AS t
            MERGE (h:ConversationConcept {name: t.head, user_id: $user_id})
              ON CREATE SET h.type = t.head_type, h.session_id = $session_id, h.created_at = datetime()
            MERGE (tail:ConversationConcept {name: t.tail, user_id: $user_id})
              ON CREATE SET tail.type = t.tail_type, tail.session_id = $session_id, tail.created_at = datetime()
            MERGE (h)-[r:RELATES_TO {type: t.relation}]->(tail)
              ON CREATE SET r.source_session = $session_id,
                            r.user_id = $user_id,
                            r.created_at = datetime()
            RETURN count(*) AS stored
            """,
            triplets=prepared,
            user_id=user_id,
            session_id=session_id,
        )
        stored = result.single()["stored"]

        # Cross-connect to matching document Entity nodes
        session.run(
            """
            MATCH (c:ConversationConcept {user_id: $user_id})
            MATCH (e:Entity)
            WHERE toLower(c.name) = toLower(e.name)
               OR toLower(e.name) CONTAINS toLower(c.name)
            MERGE (c)-[:CONNECTS_TO]->(e)
            """,
            user_id=user_id,
        )

    logger.info(f"Stored {stored} conversation triplets for session {session_id}")
    return stored


def get_graph_visualization(limit: int = 50, user_id: Optional[str] = None) -> dict:
    """Get graph data for visualization — all node types with source tagging.

    Returns:
        {
            "nodes": [{id, name, type, source, connections, importance, user_context, community_id}],
            "links": [{source, target, relation}],
            "stats": {doc_count, journal_count, conversation_count}
        }
    """
    driver = get_neo4j()

    with driver.session() as session:
        # Document Entity nodes
        node_result = session.run(
            """
            MATCH (n:Entity)
            OPTIONAL MATCH (n)-[r:RELATES_TO]-()
            WITH n, COUNT(r) AS degree
            ORDER BY degree DESC
            LIMIT $limit
            RETURN
                n.name AS name,
                n.type AS type,
                n.user_context AS user_context,
                n.communityId AS community_id,
                degree,
                toFloat(degree) / $limit AS importance,
                'document' AS source
            """,
            limit=limit,
        )

        nodes = []
        node_names = set()
        for record in node_result:
            name = record["name"]
            node_names.add(name)
            nodes.append({
                "id": name,
                "name": name,
                "type": record["type"] or "Concept",
                "source": "document",
                "connections": record["degree"],
                "importance": min(record["importance"], 1.0),
                "user_context": record.get("user_context"),
                "community_id": record.get("community_id"),
            })

        doc_count = len(nodes)

        # Journal concept nodes (user-scoped)
        journal_count = 0
        if user_id:
            j_result = session.run(
                """
                MATCH (j:JournalConcept {user_id: $user_id})
                OPTIONAL MATCH (j)-[r:RELATES_TO]-()
                WITH j, COUNT(r) AS degree
                ORDER BY degree DESC
                LIMIT 30
                RETURN j.name AS name, j.type AS type, j.user_context AS user_context, degree
                """,
                user_id=user_id,
            )
            for record in j_result:
                name = record["name"]
                if name not in node_names:
                    node_names.add(name)
                    journal_count += 1
                    nodes.append({
                        "id": name,
                        "name": name,
                        "type": record["type"] or "Concept",
                        "source": "journal",
                        "connections": record["degree"],
                        "importance": 0.3,
                        "user_context": record.get("user_context"),
                        "community_id": None,
                    })

            # Conversation concept nodes (user-scoped)
            conv_count = 0
            c_result = session.run(
                """
                MATCH (c:ConversationConcept {user_id: $user_id})
                OPTIONAL MATCH (c)-[r:RELATES_TO]-()
                WITH c, COUNT(r) AS degree
                ORDER BY degree DESC
                LIMIT 30
                RETURN c.name AS name, c.type AS type, degree
                """,
                user_id=user_id,
            )
            for record in c_result:
                name = record["name"]
                if name not in node_names:
                    node_names.add(name)
                    conv_count += 1
                    nodes.append({
                        "id": name,
                        "name": name,
                        "type": record["type"] or "Insight",
                        "source": "conversation",
                        "connections": record["degree"],
                        "importance": 0.3,
                        "user_context": None,
                        "community_id": None,
                    })
        else:
            journal_count = 0
            conv_count = 0

        # Links between all node types
        links = []
        if node_names:
            link_result = session.run(
                """
                MATCH (h)-[r:RELATES_TO]->(t)
                WHERE h.name IN $names AND t.name IN $names
                RETURN h.name AS source, t.name AS target, r.type AS relation
                """,
                names=list(node_names),
            )
            links = [
                {"source": record["source"], "target": record["target"], "relation": record["relation"]}
                for record in link_result
            ]

            # Cross-links (CONNECTS_TO)
            cross_result = session.run(
                """
                MATCH (h)-[r:CONNECTS_TO]->(t)
                WHERE h.name IN $names AND t.name IN $names
                RETURN h.name AS source, t.name AS target
                """,
                names=list(node_names),
            )
            for record in cross_result:
                links.append({"source": record["source"], "target": record["target"], "relation": "CONNECTS_TO"})

    return {
        "nodes": nodes,
        "links": links,
        "stats": {
            "doc_count": doc_count,
            "journal_count": journal_count,
            "conversation_count": conv_count if user_id else 0,
        },
    }


def get_journal_graph(user_id: str, limit: int = 50) -> dict:
    """Get the user's journal knowledge graph for visualization.

    Returns JournalConcept nodes + intra-journal edges + cross-links to Entity nodes.
    Nodes are tagged with source='journal' or source='document' for frontend color-coding.

    Returns:
        {
            "nodes": [{id, name, type, connections, source, user_context}],
            "links": [{source, target, relation}]
        }
    """
    driver = get_neo4j()

    with driver.session() as session:
        # Top journal concept nodes by connection count
        node_result = session.run(
            """
            MATCH (j:JournalConcept {user_id: $user_id})
            OPTIONAL MATCH (j)-[r:RELATES_TO]-(other:JournalConcept {user_id: $user_id})
            WITH j, COUNT(r) AS degree
            ORDER BY degree DESC
            LIMIT $limit
            RETURN j.name AS name, j.type AS type, j.user_context AS user_context, degree
            """,
            user_id=user_id,
            limit=limit,
        )

        nodes = []
        node_names = set()
        for record in node_result:
            name = record["name"]
            node_names.add(name)
            nodes.append({
                "id": name,
                "name": name,
                "type": record["type"] or "Concept",
                "connections": record["degree"],
                "source": "journal",
                "user_context": record.get("user_context"),
            })

        links = []
        if not node_names:
            return {"nodes": nodes, "links": links}

        # Intra-journal edges
        link_result = session.run(
            """
            MATCH (h:JournalConcept {user_id: $user_id})-[r:RELATES_TO]->(t:JournalConcept {user_id: $user_id})
            WHERE h.name IN $names AND t.name IN $names
            RETURN h.name AS source, t.name AS target, r.type AS relation
            """,
            user_id=user_id,
            names=list(node_names),
        )
        for record in link_result:
            links.append({
                "source": record["source"],
                "target": record["target"],
                "relation": record["relation"],
            })

        # Cross-links to document Entity nodes
        cross_result = session.run(
            """
            MATCH (j:JournalConcept {user_id: $user_id})-[:CONNECTS_TO]->(e:Entity)
            WHERE j.name IN $names
            RETURN j.name AS journal_node, e.name AS entity_node, e.type AS entity_type
            """,
            user_id=user_id,
            names=list(node_names),
        )

        entity_nodes_added = set()
        for record in cross_result:
            entity_name = record["entity_node"]
            links.append({
                "source": record["journal_node"],
                "target": entity_name,
                "relation": "CONNECTS_TO",
            })
            if entity_name not in node_names and entity_name not in entity_nodes_added:
                entity_nodes_added.add(entity_name)
                nodes.append({
                    "id": entity_name,
                    "name": entity_name,
                    "type": record.get("entity_type") or "Concept",
                    "connections": 1,
                    "source": "document",
                    "user_context": None,
                })

    return {"nodes": nodes, "links": links}


def update_node_context(
    node_name: str,
    context: str,
    user_id: Optional[str] = None,
) -> bool:
    """Add or update user-provided context on a graph node."""
    driver = get_neo4j()

    with driver.session() as session:
        result = session.run(
            """
            MATCH (n:Entity {name: $name})
            SET n.user_context = $context,
                n.context_updated_by = $user_id,
                n.context_updated_at = datetime()
            RETURN n.name AS name
            """,
            name=node_name,
            context=context,
            user_id=user_id,
        )
        return result.single() is not None


def get_all_entities_and_relationships() -> tuple[list[dict], list[dict]]:
    """Export the full graph for community detection.

    Returns:
        (nodes, edges) where:
        - nodes: [{name, type, ...}]
        - edges: [{source, target, relation}]
    """
    driver = get_neo4j()

    with driver.session() as session:
        node_result = session.run("MATCH (n:Entity) RETURN n.name AS name, n.type AS type")
        nodes = [{"name": r["name"], "type": r["type"]} for r in node_result]

        edge_result = session.run(
            """
            MATCH (h:Entity)-[r:RELATES_TO]->(t:Entity)
            RETURN h.name AS source, t.name AS target, r.type AS relation
            """
        )
        edges = [
            {"source": r["source"], "target": r["target"], "relation": r["relation"]}
            for r in edge_result
        ]

    return nodes, edges


def set_community_ids(community_assignments: dict[str, int]) -> int:
    """Write community IDs back to Neo4j Entity nodes in a single batched query.

    Uses UNWIND instead of N individual writes.
    """
    if not community_assignments:
        return 0

    driver = get_neo4j()
    assignments = [{"name": k, "community_id": v} for k, v in community_assignments.items()]

    with driver.session() as session:
        result = session.run(
            """
            UNWIND $assignments AS item
            MATCH (n:Entity {name: item.name})
            SET n.communityId = item.community_id
            RETURN count(*) AS updated
            """,
            assignments=assignments,
        )
        updated = result.single()["updated"]

    logger.info(f"Set communityId on {updated}/{len(community_assignments)} nodes")
    return updated


def get_user_journal_concept_names(user_id: str) -> set[str]:
    """Return the lowercased names of all JournalConcept nodes for a user.

    Used by the smart filter novelty check to avoid re-extracting known concepts.
    """
    driver = get_neo4j()
    with driver.session() as session:
        result = session.run(
            "MATCH (j:JournalConcept {user_id: $user_id}) RETURN j.name AS name",
            user_id=user_id,
        )
        return {r["name"].lower() for r in result}


def delete_journal_concepts(entry_id: str, user_id: str) -> int:
    """Remove relationships from a deleted journal entry, then clean up orphaned nodes."""
    driver = get_neo4j()
    with driver.session() as session:
        # Step 1: Remove all RELATES_TO relationships from this entry
        session.run(
            """
            MATCH ()-[r:RELATES_TO {source_journal: $entry_id, user_id: $user_id}]-()
            DELETE r
            """,
            entry_id=entry_id,
            user_id=user_id,
        )
        # Step 2: Remove JournalConcept nodes that are now fully disconnected
        result = session.run(
            """
            MATCH (j:JournalConcept {user_id: $user_id})
            WHERE NOT (j)-[:RELATES_TO]-() AND NOT (j)-[:CONNECTS_TO]-()
            DETACH DELETE j
            RETURN count(*) AS deleted
            """,
            user_id=user_id,
        )
        deleted = result.single()["deleted"]
    logger.info(f"Deleted {deleted} orphaned journal concept nodes for entry {entry_id}")
    return deleted
