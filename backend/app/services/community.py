"""Community detection using Python-side Leiden algorithm.

Since Neo4j AuraDB Free does not include GDS, we:
1. Export the graph from Neo4j
2. Build an igraph/networkx graph locally
3. Run Leiden community detection via cdlib
4. Write community IDs back to Neo4j nodes
"""

import asyncio
import logging
from typing import Optional

from app.services.graph_service import (
    get_all_entities_and_relationships,
    set_community_ids,
)

logger = logging.getLogger(__name__)

# Debounce flag — prevents queuing duplicate runs during rapid back-to-back ingestions
_detection_running: bool = False


async def trigger_community_detection_background(resolution: float = 1.0) -> None:
    """Fire-and-forget community detection after ingestion.

    Skips silently if a run is already in progress (debounce). This means
    concurrent document ingestions only trigger one detection run, not N.
    """
    global _detection_running
    if _detection_running:
        logger.info("Community detection already running — skipping duplicate trigger")
        return
    _detection_running = True
    try:
        await asyncio.to_thread(run_leiden_community_detection, resolution)
    finally:
        _detection_running = False


def run_leiden_community_detection(resolution: float = 1.0) -> dict:
    """Run Leiden community detection on the full knowledge graph.

    Steps:
    1. Export all entities and relationships from Neo4j
    2. Build an igraph Graph
    3. Run Leiden algorithm via cdlib
    4. Write community IDs back to Neo4j nodes (single UNWIND batch)

    Args:
        resolution: Leiden resolution parameter. Higher = more communities.

    Returns:
        Summary dict with community stats.
    """
    logger.info("Exporting graph from Neo4j for community detection...")
    nodes, edges = get_all_entities_and_relationships()

    if len(nodes) < 2:
        return {"status": "skipped", "reason": "Not enough nodes for community detection", "node_count": len(nodes)}

    import igraph as ig
    from cdlib import algorithms

    logger.info(f"Building igraph graph: {len(nodes)} nodes, {len(edges)} edges")

    name_to_idx = {node["name"]: i for i, node in enumerate(nodes)}

    g = ig.Graph(directed=False)
    g.add_vertices(len(nodes))
    g.vs["name"] = [n["name"] for n in nodes]
    g.vs["type"] = [n.get("type", "Concept") for n in nodes]

    edge_list = []
    for edge in edges:
        src_idx = name_to_idx.get(edge["source"])
        tgt_idx = name_to_idx.get(edge["target"])
        if src_idx is not None and tgt_idx is not None:
            edge_list.append((src_idx, tgt_idx))
    g.add_edges(edge_list)

    isolated = [v.index for v in g.vs if g.degree(v) == 0]
    if isolated:
        logger.info(f"Removing {len(isolated)} isolated nodes before community detection")

    logger.info(f"Running Leiden algorithm (resolution={resolution})...")
    try:
        communities = algorithms.leiden(g, initial_membership=None, weights=None)
    except Exception as e:
        logger.error(f"Leiden algorithm failed: {e}")
        return {"status": "error", "error": str(e)}

    # Build name lookup once to avoid per-community iteration overhead
    name_by_idx = {v.index: v["name"] for v in g.vs}

    community_assignments = {}
    for community_id, community_members in enumerate(communities.communities):
        for member_idx in community_members:
            community_assignments[name_by_idx[member_idx]] = community_id

    logger.info(f"Detected {len(communities.communities)} communities across {len(community_assignments)} nodes")

    # Single batched write-back instead of N individual queries
    updated = set_community_ids(community_assignments)

    community_sizes = {
        comm_id: {
            "size": len(members),
            "members": [name_by_idx[idx] for idx in members[:10]],
        }
        for comm_id, members in enumerate(communities.communities)
    }

    return {
        "status": "success",
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "communities_detected": len(communities.communities),
        "nodes_assigned": updated,
        "resolution": resolution,
        "community_summary": community_sizes,
    }


def get_community_for_entity(entity_name: str) -> Optional[int]:
    """Get the community ID for a specific entity from Neo4j."""
    from app.core.database import get_neo4j

    driver = get_neo4j()
    with driver.session() as session:
        result = session.run(
            "MATCH (n:Entity {name: $name}) RETURN n.communityId AS community_id",
            name=entity_name,
        )
        record = result.single()
        if record and record["community_id"] is not None:
            return int(record["community_id"])
    return None


def get_community_members(community_id: int) -> list[dict]:
    """Get all entities in a specific community."""
    from app.core.database import get_neo4j

    driver = get_neo4j()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n:Entity {communityId: $cid})
            OPTIONAL MATCH (n)-[r:RELATES_TO]-()
            WITH n, COUNT(r) AS connections
            RETURN n.name AS name, n.type AS type, connections
            ORDER BY connections DESC
            """,
            cid=community_id,
        )
        return [dict(r) for r in result]
