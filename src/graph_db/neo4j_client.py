"""Neo4j client for code relationship graph.

This module provides a graph database interface for tracking structural relationships
between code entities across both the code_chunks and dependency_knowledge collections.

Relationships tracked:
- CALLS: Function/method calls within code
- IMPORTS: File/module imports
- EXTENDS: Class inheritance
- IMPLEMENTS: Interface implementation
- USES: Usage of external dependencies (from dependency_knowledge)
- BELONGS_TO: Entity belongs to a class/module
- DEFINES: File defines an entity
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the code graph."""

    id: str
    label: str  # Function, Class, Method, File, Module, DependencyFunction, etc.
    properties: Dict[str, Any]
    source_collection: str  # "code_chunks" or "dependency_knowledge"


@dataclass
class GraphRelationship:
    """Represents a relationship between code entities."""

    source_id: str
    target_id: str
    relationship_type: str  # CALLS, IMPORTS, EXTENDS, IMPLEMENTS, USES, etc.
    properties: Dict[str, Any]


class CodeGraphDB:
    """Neo4j client for code relationship graph.

    Manages a graph database that tracks relationships between code entities
    from both the primary codebase (code_chunks) and external dependencies
    (dependency_knowledge).
    """

    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection.

        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
        """
        self.uri = uri
        self.user = user
        self.driver = None

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info(f"Connected to Neo4j at {uri}")
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def verify_connectivity(self) -> bool:
        """Verify connection to Neo4j.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS result")
                return result.single()["result"] == 1
        except Exception as e:
            logger.error(f"Neo4j connectivity check failed: {e}")
            return False

    def create_indexes(self):
        """Create indexes for frequently queried properties."""
        indexes = [
            # Node indexes for fast lookups
            "CREATE INDEX function_id IF NOT EXISTS FOR (f:Function) ON (f.id)",
            "CREATE INDEX function_name IF NOT EXISTS FOR (f:Function) ON (f.name)",
            "CREATE INDEX class_id IF NOT EXISTS FOR (c:Class) ON (c.id)",
            "CREATE INDEX class_name IF NOT EXISTS FOR (c:Class) ON (c.name)",
            "CREATE INDEX method_id IF NOT EXISTS FOR (m:Method) ON (m.id)",
            "CREATE INDEX file_path IF NOT EXISTS FOR (f:File) ON (f.path)",
            "CREATE INDEX file_repo IF NOT EXISTS FOR (f:File) ON (f.repo_name)",
            "CREATE INDEX module_name IF NOT EXISTS FOR (m:Module) ON (m.name)",
            # Dependency node indexes
            "CREATE INDEX dep_func_id IF NOT EXISTS FOR (df:DependencyFunction) ON (df.id)",
            "CREATE INDEX dep_func_name IF NOT EXISTS FOR (df:DependencyFunction) ON (df.name)",
            "CREATE INDEX dep_class_id IF NOT EXISTS FOR (dc:DependencyClass) ON (dc.id)",
        ]

        with self.driver.session() as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                    logger.debug(f"Created index: {index_query}")
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")

    def create_node(self, node: GraphNode) -> str:
        """Create a node in the graph.

        Args:
            node: GraphNode to create

        Returns:
            Node ID
        """
        with self.driver.session() as session:
            query = f"""
            MERGE (n:{node.label} {{id: $id}})
            SET n += $properties
            SET n.source_collection = $source_collection
            RETURN n.id AS id
            """
            result = session.run(
                query,
                id=node.id,
                properties=node.properties,
                source_collection=node.source_collection
            )
            return result.single()["id"]

    def batch_create_nodes(self, nodes: List[GraphNode], batch_size: int = 1000):
        """Create multiple nodes in batches.

        Args:
            nodes: List of GraphNode objects
            batch_size: Number of nodes to create per transaction
        """
        with self.driver.session() as session:
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]

                # Group by label for efficient batching
                nodes_by_label = {}
                for node in batch:
                    if node.label not in nodes_by_label:
                        nodes_by_label[node.label] = []
                    nodes_by_label[node.label].append({
                        "id": node.id,
                        "properties": node.properties,
                        "source_collection": node.source_collection
                    })

                # Batch insert by label
                for label, label_nodes in nodes_by_label.items():
                    query = f"""
                    UNWIND $nodes AS node
                    MERGE (n:{label} {{id: node.id}})
                    SET n += node.properties
                    SET n.source_collection = node.source_collection
                    """
                    session.run(query, nodes=label_nodes)

                logger.info(f"Created batch of {len(batch)} nodes (batch {i//batch_size + 1})")

    def create_relationship(self, relationship: GraphRelationship):
        """Create a relationship between two nodes.

        Args:
            relationship: GraphRelationship to create
        """
        with self.driver.session() as session:
            query = f"""
            MATCH (source {{id: $source_id}})
            MATCH (target {{id: $target_id}})
            MERGE (source)-[r:{relationship.relationship_type}]->(target)
            SET r += $properties
            """
            session.run(
                query,
                source_id=relationship.source_id,
                target_id=relationship.target_id,
                properties=relationship.properties
            )

    def batch_create_relationships(
        self,
        relationships: List[GraphRelationship],
        batch_size: int = 5000
    ):
        """Create multiple relationships in batches.

        Args:
            relationships: List of GraphRelationship objects
            batch_size: Number of relationships to create per transaction
        """
        with self.driver.session() as session:
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i:i + batch_size]

                # Group by relationship type
                rels_by_type = {}
                for rel in batch:
                    if rel.relationship_type not in rels_by_type:
                        rels_by_type[rel.relationship_type] = []
                    rels_by_type[rel.relationship_type].append({
                        "source_id": rel.source_id,
                        "target_id": rel.target_id,
                        "properties": rel.properties
                    })

                # Batch insert by type
                for rel_type, type_rels in rels_by_type.items():
                    query = f"""
                    UNWIND $rels AS rel
                    MATCH (source {{id: rel.source_id}})
                    MATCH (target {{id: rel.target_id}})
                    MERGE (source)-[r:{rel_type}]->(target)
                    SET r += rel.properties
                    """
                    session.run(query, rels=type_rels)

                logger.info(f"Created batch of {len(batch)} relationships (batch {i//batch_size + 1})")

    def get_call_graph(
        self,
        function_id: str,
        depth: int = 3,
        direction: str = "both"
    ) -> Dict[str, Any]:
        """Get call graph for a function.

        Args:
            function_id: ID of the function
            depth: Maximum depth to traverse
            direction: "both" (callers+callees), "callers", or "callees"

        Returns:
            Graph structure with nodes and edges
        """
        with self.driver.session() as session:
            if direction == "callers":
                query = """
                MATCH path = (caller)-[:CALLS*1..%d]->(f {id: $function_id})
                RETURN path
                """ % depth
            elif direction == "callees":
                query = """
                MATCH path = (f {id: $function_id})-[:CALLS*1..%d]->(callee)
                RETURN path
                """ % depth
            else:  # both
                query = """
                MATCH path = (n)-[:CALLS*1..%d]-(f {id: $function_id})
                RETURN path
                """ % depth

            result = session.run(query, function_id=function_id)

            # Convert paths to graph structure
            nodes = {}
            edges = []

            for record in result:
                path = record["path"]
                for node in path.nodes:
                    nodes[node["id"]] = dict(node)
                for rel in path.relationships:
                    edges.append({
                        "source": rel.start_node["id"],
                        "target": rel.end_node["id"],
                        "type": rel.type,
                        "properties": dict(rel)
                    })

            return {
                "nodes": list(nodes.values()),
                "edges": edges,
                "center_node": function_id,
                "depth": depth,
                "direction": direction
            }

    def find_callers(self, function_id: str, max_depth: int = 10) -> List[Dict[str, Any]]:
        """Find all functions that call this function.

        Args:
            function_id: ID of the function
            max_depth: Maximum depth to search

        Returns:
            List of caller nodes
        """
        with self.driver.session() as session:
            query = """
            MATCH (caller)-[:CALLS*1..%d]->(f {id: $function_id})
            RETURN DISTINCT caller, length(shortestPath((caller)-[:CALLS*]->(f))) AS distance
            ORDER BY distance
            """ % max_depth

            result = session.run(query, function_id=function_id)
            return [{"node": dict(record["caller"]), "distance": record["distance"]}
                    for record in result]

    def find_callees(self, function_id: str, max_depth: int = 10) -> List[Dict[str, Any]]:
        """Find all functions that this function calls.

        Args:
            function_id: ID of the function
            max_depth: Maximum depth to search

        Returns:
            List of callee nodes
        """
        with self.driver.session() as session:
            query = """
            MATCH (f {id: $function_id})-[:CALLS*1..%d]->(callee)
            RETURN DISTINCT callee, length(shortestPath((f)-[:CALLS*]->(callee))) AS distance
            ORDER BY distance
            """ % max_depth

            result = session.run(query, function_id=function_id)
            return [{"node": dict(record["callee"]), "distance": record["distance"]}
                    for record in result]

    def find_usages(
        self,
        entity_id: str,
        repo_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find all usages of a code entity (function, class, etc.).

        Args:
            entity_id: ID of the entity
            repo_name: Optional filter by repository

        Returns:
            List of nodes that reference this entity
        """
        with self.driver.session() as session:
            if repo_name:
                query = """
                MATCH (user)-[r:CALLS|USES|REFERENCES]->(entity {id: $entity_id})
                WHERE user.repo_name = $repo_name
                RETURN user, type(r) AS relationship_type
                """
                result = session.run(query, entity_id=entity_id, repo_name=repo_name)
            else:
                query = """
                MATCH (user)-[r:CALLS|USES|REFERENCES]->(entity {id: $entity_id})
                RETURN user, type(r) AS relationship_type
                """
                result = session.run(query, entity_id=entity_id)

            return [{"node": dict(record["user"]), "relationship": record["relationship_type"]}
                    for record in result]

    def get_class_hierarchy(
        self,
        class_id: str,
        include_methods: bool = True
    ) -> Dict[str, Any]:
        """Get inheritance hierarchy for a class.

        Args:
            class_id: ID of the class
            include_methods: Whether to include class methods

        Returns:
            Tree structure showing superclasses, subclasses, and interfaces
        """
        with self.driver.session() as session:
            # Get superclasses
            superclass_query = """
            MATCH path = (c {id: $class_id})-[:EXTENDS*]->(super)
            RETURN path
            """
            superclasses = session.run(superclass_query, class_id=class_id)

            # Get subclasses
            subclass_query = """
            MATCH path = (sub)-[:EXTENDS*]->(c {id: $class_id})
            RETURN path
            """
            subclasses = session.run(subclass_query, class_id=class_id)

            # Get interfaces
            interface_query = """
            MATCH (c {id: $class_id})-[:IMPLEMENTS]->(interface)
            RETURN interface
            """
            interfaces = session.run(interface_query, class_id=class_id)

            hierarchy = {
                "class_id": class_id,
                "superclasses": [dict(record["path"].nodes[-1]) for record in superclasses],
                "subclasses": [dict(record["path"].nodes[0]) for record in subclasses],
                "interfaces": [dict(record["interface"]) for record in interfaces],
            }

            if include_methods:
                method_query = """
                MATCH (c {id: $class_id})<-[:BELONGS_TO]-(m:Method)
                RETURN m
                """
                methods = session.run(method_query, class_id=class_id)
                hierarchy["methods"] = [dict(record["m"]) for record in methods]

            return hierarchy

    def detect_circular_dependencies(self, repo_name: str) -> List[List[str]]:
        """Detect circular dependencies in a repository.

        Args:
            repo_name: Repository to analyze

        Returns:
            List of circular dependency chains
        """
        with self.driver.session() as session:
            query = """
            MATCH path = (f:File {repo_name: $repo_name})-[:IMPORTS*2..]->(f)
            RETURN [node in nodes(path) | node.path] AS cycle
            """
            result = session.run(query, repo_name=repo_name)
            return [record["cycle"] for record in result]

    def delete_by_file_path(self, file_path: str, repo_name: Optional[str] = None):
        """Delete all nodes and relationships for a specific file.

        Args:
            file_path: File path to delete nodes for
            repo_name: Optional repository filter for safety
        """
        with self.driver.session() as session:
            if repo_name:
                query = """
                MATCH (n {file_path: $file_path, repo_name: $repo_name})
                DETACH DELETE n
                """
                result = session.run(query, file_path=file_path, repo_name=repo_name)
            else:
                query = """
                MATCH (n {file_path: $file_path})
                DETACH DELETE n
                """
                result = session.run(query, file_path=file_path)

            # Get count of deleted nodes
            summary = result.consume()
            deleted_count = summary.counters.nodes_deleted
            logger.info(f"Deleted {deleted_count} nodes from {file_path}")
            return deleted_count

    def clear_graph(self, repo_name: Optional[str] = None):
        """Clear all nodes and relationships, optionally filtered by repo.

        Args:
            repo_name: If provided, only clear nodes from this repository
        """
        with self.driver.session() as session:
            if repo_name:
                query = """
                MATCH (n {repo_name: $repo_name})
                DETACH DELETE n
                """
                session.run(query, repo_name=repo_name)
                logger.info(f"Cleared graph data for repository: {repo_name}")
            else:
                query = "MATCH (n) DETACH DELETE n"
                session.run(query)
                logger.info("Cleared entire graph database")

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph database statistics.

        Returns:
            Dictionary with node counts, relationship counts, etc.
        """
        with self.driver.session() as session:
            stats_query = """
            MATCH (n)
            RETURN labels(n)[0] AS label, count(*) AS count
            """
            node_counts = session.run(stats_query)

            rel_query = """
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(*) AS count
            """
            rel_counts = session.run(rel_query)

            return {
                "nodes_by_label": {record["label"]: record["count"] for record in node_counts},
                "relationships_by_type": {record["type"]: record["count"] for record in rel_counts}
            }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
