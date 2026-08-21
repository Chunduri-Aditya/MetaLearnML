"""Mocked graph and API integration checks that need no external service."""

from __future__ import annotations

import pytest

from graph.neo4j_client import Neo4jConfig, Neo4jGraph
from web import app as web_app

pytestmark = pytest.mark.integration


class FakeGraph:
    def __init__(self):
        self.closed = False

    def run(self, query, **params):
        assert "Dataset" in query
        return [{"name": "fixture", "n_samples": 12, "n_features": 3, "task_type": "regression"}]

    def close(self):
        self.closed = True


def test_api_root_and_mocked_dataset_query(monkeypatch):
    fake = FakeGraph()
    monkeypatch.setattr(web_app, "get_graph", lambda: fake)
    assert web_app.root()["version"] == "1.0.0"
    response = web_app.list_datasets()
    assert response["datasets"][0]["name"] == "fixture"
    assert fake.closed


def test_graph_logging_records_objective_and_structured_preprocessor(monkeypatch):
    graph = Neo4jGraph(Neo4jConfig(password=None))
    calls = []
    monkeypatch.setattr(graph, "run", lambda query, **params: calls.append((query, params)) or [])
    graph.upsert_preprocessor(
        "structured",
        components={
            "numeric_scaler": "standard",
            "numeric_imputer": "median",
            "categorical_encoder": "onehot",
            "categorical_imputer": "most_frequent",
        },
    )
    graph.create_run("run", "rmse", -2.0, display_value=2.0, candidate_id="abc")
    assert calls[0][1]["numeric_scaler"] == "standard"
    assert calls[1][1]["metric_value"] == -2.0
    assert calls[1][1]["display_value"] == 2.0
    graph.get_best_models_for_task("regression")
    assert "ORDER BY avg_objective DESC" in calls[-1][0]
