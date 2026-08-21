"""Opt-in live Neo4j integration test."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.neo4j]


def test_connection():
    """Connect and run a side-effect-free query only when explicitly enabled."""
    if os.getenv("RUN_NEO4J_TESTS", "0") != "1":
        pytest.skip("Set RUN_NEO4J_TESTS=1 to run Neo4j integration tests.")

    from graph.neo4j_client import Neo4jConfig, Neo4jGraph

    graph = Neo4jGraph(Neo4jConfig())
    try:
        assert graph.is_connected()
        assert graph.run("RETURN 1 AS value")[0]["value"] == 1
    finally:
        graph.close()
