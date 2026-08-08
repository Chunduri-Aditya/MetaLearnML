"""
Neo4j graph database client for AutoML experiment tracking.

Stores experiments as a knowledge graph for querying and analysis.
"""

from neo4j import GraphDatabase
from dataclasses import dataclass
from typing import Dict, Optional, List, Any
import os


@dataclass
class Neo4jConfig:
    """Configuration for Neo4j connection."""
    uri: str = "neo4j://127.0.0.1:7687"
    user: str = "neo4j"
    password: Optional[str] = None
    
    @classmethod
    def from_env(cls):
        """Load config from environment variables."""
        return cls(
            uri=os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD"),
        )


class Neo4jGraph:
    """
    Neo4j graph database client for AutoML experiments.
    
    Graph schema:
    - (:Dataset) - Dataset nodes with metadata
    - (:Model) - Model nodes with family/backend
    - (:Preprocessor) - Preprocessing strategy nodes
    - (:Run) - Experiment run nodes with metrics
    - Relationships:
      - (:Dataset)-[:USED_IN]->(:Run)
      - (:Model)-[:APPLIED_IN]->(:Run)
      - (:Preprocessor)-[:PART_OF]->(:Run)
    """
    
    def __init__(self, config: Optional[Neo4jConfig] = None):
        if config is None:
            config = Neo4jConfig.from_env()
        self.config = config
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Connect to Neo4j database."""
        if not self.config.password:
            print("[WARN] NEO4J_PASSWORD is not set. Graph logging will be disabled.")
            self.driver = None
            return

        try:
            # Neo4j driver typically uses bolt:// for direct connections
            # Convert neo4j:// to bolt:// if needed
            uri = self.config.uri
            if uri.startswith("neo4j://"):
                # Try bolt:// first (most common for direct connections)
                bolt_uri = uri.replace("neo4j://", "bolt://")
                try:
                    self.driver = GraphDatabase.driver(
                        bolt_uri,
                        auth=(self.config.user, self.config.password)
                    )
                    # Test connection
                    with self.driver.session() as session:
                        session.run("RETURN 1")
                    print(f"[INFO] Connected to Neo4j at {bolt_uri}")
                    return
                except Exception as e1:
                    # Fallback to neo4j:// format
                    try:
                        self.driver = GraphDatabase.driver(
                            uri,
                            auth=(self.config.user, self.config.password)
                        )
                        # Test connection
                        with self.driver.session() as session:
                            session.run("RETURN 1")
                        print(f"[INFO] Connected to Neo4j at {uri}")
                        return
                    except Exception as e2:
                        raise Exception(f"Failed with bolt://: {e1}, failed with neo4j://: {e2}")
            else:
                # Already using bolt:// or other format
                self.driver = GraphDatabase.driver(
                    uri,
                    auth=(self.config.user, self.config.password)
                )
                # Test connection
                with self.driver.session() as session:
                    session.run("RETURN 1")
                print(f"[INFO] Connected to Neo4j at {uri}")
        except Exception as e:
            print(f"[WARN] Could not connect to Neo4j: {e}")
            print(f"[WARN] URI: {self.config.uri}, User: {self.config.user}")
            print("[WARN] Graph logging will be disabled. Check Neo4j connection settings.")
            print("[WARN] Make sure Neo4j is running and accessible.")
            self.driver = None
    
    def close(self):
        """Close database connection."""
        if self.driver:
            self.driver.close()
    
    def is_connected(self) -> bool:
        """Check if connected to Neo4j."""
        return self.driver is not None
    
    def run(self, query: str, **params) -> List[Dict]:
        """
        Execute a Cypher query.
        
        Args:
            query: Cypher query string
            **params: Query parameters
        
        Returns:
            List of result records as dictionaries
        """
        if not self.driver:
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run(query, **params)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"[WARN] Neo4j query failed: {e}")
            return []
    
    # --- Upsert helpers ---
    
    def upsert_dataset(self, name: str, meta: Dict):
        """Create or update a Dataset node."""
        self.run(
            """
            MERGE (d:Dataset {name: $name})
            SET d.n_samples = $n_samples,
                d.n_features = $n_features,
                d.task_type = $task_type,
                d.n_numeric = $n_numeric,
                d.n_categorical = $n_categorical,
                d.n_classes = $n_classes
            """,
            name=name,
            n_samples=meta.get("n_samples", 0),
            n_features=meta.get("n_features", 0),
            task_type=meta.get("task_type", "unknown"),
            n_numeric=meta.get("n_numeric", 0),
            n_categorical=meta.get("n_categorical", 0),
            n_classes=meta.get("n_classes"),
        )
    
    def upsert_model(self, name: str, family: str, backend: str):
        """Create or update a Model node."""
        self.run(
            """
            MERGE (m:Model {name: $name})
            SET m.family = $family,
                m.backend = $backend
            """,
            name=name,
            family=family,
            backend=backend,
        )
    
    def upsert_preprocessor(self, name: str, tier: Optional[str] = None):
        """Create or update a Preprocessor node."""
        self.run(
            """
            MERGE (p:Preprocessor {name: $name})
            SET p.tier = $tier
            """,
            name=name,
            tier=tier,
        )
    
    def create_run(self, run_id: str, metric_name: str, metric_value: float, timestamp: Optional[str] = None):
        """Create a Run node."""
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
        
        self.run(
            """
            MERGE (r:Run {id: $run_id})
            SET r.metric_name = $metric_name,
                r.metric_value = $metric_value,
                r.timestamp = $timestamp
            """,
            run_id=run_id,
            metric_name=metric_name,
            metric_value=metric_value,
            timestamp=timestamp,
        )
    
    def link_run(
        self,
        dataset_name: str,
        model_name: str,
        preproc_name: str,
        run_id: str,
    ):
        """Link a run to its dataset, model, and preprocessor."""
        self.run(
            """
            MATCH (d:Dataset {name: $dataset_name})
            MATCH (m:Model {name: $model_name})
            MATCH (p:Preprocessor {name: $preproc_name})
            MATCH (r:Run {id: $run_id})
            MERGE (d)-[:USED_IN]->(r)
            MERGE (m)-[:APPLIED_IN]->(r)
            MERGE (p)-[:PART_OF]->(r)
            """,
            dataset_name=dataset_name,
            model_name=model_name,
            preproc_name=preproc_name,
            run_id=run_id,
        )
    
    # --- Query helpers ---
    
    def get_dataset_runs(self, dataset_name: str) -> List[Dict]:
        """Get all runs for a dataset."""
        return self.run(
            """
            MATCH (d:Dataset {name: $name})-[:USED_IN]->(r:Run)<-[:APPLIED_IN]-(m:Model)
            OPTIONAL MATCH (r)<-[:PART_OF]-(p:Preprocessor)
            RETURN r.id AS run_id, r.metric_name AS metric_name, r.metric_value AS metric_value,
                   r.timestamp AS timestamp, m.name AS model_name, m.family AS model_family,
                   p.name AS preprocessing_name
            ORDER BY r.metric_value DESC
            """,
            name=dataset_name,
        )
    
    def get_model_performance(self, model_name: str) -> List[Dict]:
        """Get performance of a model across all datasets."""
        return self.run(
            """
            MATCH (m:Model {name: $name})-[:APPLIED_IN]->(r:Run)<-[:USED_IN]-(d:Dataset)
            RETURN d.name AS dataset_name, r.metric_name AS metric_name, 
                   r.metric_value AS metric_value, r.timestamp AS timestamp
            ORDER BY r.metric_value DESC
            """,
            name=model_name,
        )
    
    def get_best_models_for_task(self, task_type: str, limit: int = 10) -> List[Dict]:
        """Get best models for a specific task type."""
        return self.run(
            """
            MATCH (d:Dataset {task_type: $task_type})-[:USED_IN]->(r:Run)<-[:APPLIED_IN]-(m:Model)
            RETURN m.name AS model_name, m.family AS model_family,
                   AVG(r.metric_value) AS avg_metric, COUNT(r) AS num_runs
            ORDER BY avg_metric DESC
            LIMIT $limit
            """,
            task_type=task_type,
            limit=limit,
        )
