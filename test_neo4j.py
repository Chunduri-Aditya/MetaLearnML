#!/usr/bin/env python3
"""
Quick test script for Neo4j connection.
Run this after starting Neo4j to verify the connection works.
"""

from graph.neo4j_client import Neo4jGraph, Neo4jConfig

def test_connection():
    """Test Neo4j connection and basic operations."""
    print("=" * 60)
    print("Neo4j Connection Test")
    print("=" * 60)
    print()
    
    # Create graph client
    config = Neo4jConfig()
    print(f"Connecting to: {config.uri}")
    print(f"User: {config.user}")
    print()
    
    graph = Neo4jGraph(config)
    
    if not graph.is_connected():
        print("❌ Connection failed!")
        print()
        print("Troubleshooting:")
        print("1. Make sure Neo4j is running:")
        print("   - Check Neo4j Desktop")
        print("   - Or: docker ps | grep neo4j")
        print()
        print("2. Verify connection details:")
        print(f"   - URI: {config.uri}")
        print(f"   - User: {config.user}")
        print()
        print("3. Test connection manually:")
        print("   - Open Neo4j Browser: http://localhost:7474")
        print("   - Try connecting with the same credentials")
        return False
    
    print("✅ Connected successfully!")
    print()
    
    # Test basic query
    print("Testing basic query...")
    result = graph.run("RETURN 1 as test, 'Hello Neo4j!' as message")
    if result:
        print(f"✅ Query successful: {result[0]}")
    else:
        print("❌ Query returned no results")
        return False
    
    print()
    print("Testing graph operations...")
    
    # Test upsert operations
    try:
        graph.upsert_dataset("test_dataset", {
            "n_samples": 1000,
            "n_features": 10,
            "task_type": "binary_classification",
            "n_numeric": 5,
            "n_categorical": 5,
            "n_classes": 2,
        })
        print("✅ Dataset upsert successful")
        
        graph.upsert_model("test_model", "test_family", "test_backend")
        print("✅ Model upsert successful")
        
        graph.upsert_preprocessor("test_preprocessor", "tier1")
        print("✅ Preprocessor upsert successful")
        
        # Clean up test data
        graph.run("MATCH (n) WHERE n.name IN ['test_dataset', 'test_model', 'test_preprocessor'] DETACH DELETE n")
        print("✅ Test data cleaned up")
        
    except Exception as e:
        print(f"❌ Graph operations failed: {e}")
        return False
    
    print()
    print("=" * 60)
    print("✅ All tests passed! Neo4j is ready to use.")
    print("=" * 60)
    
    graph.close()
    return True

if __name__ == "__main__":
    success = test_connection()
    exit(0 if success else 1)

