"""
FastAPI web application for AutoML Graph Explorer.

Provides REST API to query the Neo4j knowledge graph of AutoML experiments.
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from graph.neo4j_client import Neo4jGraph

app = FastAPI(
    title="AutoML Graph Explorer",
    description="REST API for querying AutoML experiment knowledge graph",
    version="1.0.0",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_graph() -> Neo4jGraph:
    """Get Neo4j graph client."""
    graph = Neo4jGraph()
    if not graph.is_connected():
        raise HTTPException(status_code=503, detail="Neo4j database not available")
    return graph


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "AutoML Graph Explorer API",
        "version": "1.0.0",
        "endpoints": {
            "datasets": "/datasets",
            "dataset_runs": "/datasets/{dataset_name}/runs",
            "models": "/models",
            "model_performance": "/models/{model_name}/performance",
            "best_models": "/best-models/{task_type}",
        },
    }


@app.get("/datasets")
def list_datasets():
    """List all datasets in the graph."""
    graph = get_graph()
    try:
        rows = graph.run(
            """
            MATCH (d:Dataset)
            RETURN d.name AS name, d.n_samples AS n_samples, 
                   d.n_features AS n_features, d.task_type AS task_type
            ORDER BY name
            """
        )
        return {"datasets": rows}
    finally:
        graph.close()


@app.get("/datasets/{dataset_name}/runs")
def dataset_runs(dataset_name: str, limit: int = Query(100, ge=1, le=1000)):
    """Get all runs for a specific dataset."""
    graph = get_graph()
    try:
        rows = graph.get_dataset_runs(dataset_name)
        return {"dataset": dataset_name, "runs": rows[:limit]}
    finally:
        graph.close()


@app.get("/models")
def list_models():
    """List all models in the graph."""
    graph = get_graph()
    try:
        rows = graph.run(
            """
            MATCH (m:Model)
            RETURN m.name AS name, m.family AS family, m.backend AS backend
            ORDER BY name
            """
        )
        return {"models": rows}
    finally:
        graph.close()


@app.get("/models/{model_name}/performance")
def model_performance(model_name: str):
    """Get performance of a model across all datasets."""
    graph = get_graph()
    try:
        rows = graph.get_model_performance(model_name)
        return {"model": model_name, "performance": rows}
    finally:
        graph.close()


@app.get("/best-models/{task_type}")
def best_models_for_task(task_type: str, limit: int = Query(10, ge=1, le=100)):
    """Get best models for a specific task type."""
    graph = get_graph()
    try:
        rows = graph.get_best_models_for_task(task_type, limit=limit)
        return {"task_type": task_type, "best_models": rows}
    finally:
        graph.close()


@app.get("/stats")
def get_stats():
    """Get overall statistics."""
    graph = get_graph()
    try:
        stats = graph.run(
            """
            MATCH (r:Run)
            RETURN COUNT(r) AS total_runs,
                   COUNT(DISTINCT (r)<-[:USED_IN]-(:Dataset)) AS num_datasets,
                   COUNT(DISTINCT (r)<-[:APPLIED_IN]-(:Model)) AS num_models,
                   AVG(r.metric_value) AS avg_metric
            """
        )
        return {"stats": stats[0] if stats else {}}
    finally:
        graph.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
