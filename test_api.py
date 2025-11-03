"""
Test script for Adaptive Search API

Tests the API endpoints and verifies functionality
"""

import requests
import time
import json
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health endpoint"""
    print("="*80)
    print("🏥 Testing Health Check Endpoint")
    print("="*80)

    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()

        data = response.json()
        print(f"✅ Status: {data['status']}")
        print(f"   Version: {data['version']}")
        print(f"   Uptime: {data['uptime_seconds']:.2f}s")
        print(f"   Engine Loaded: {data['engine_loaded']}")

        return data['status'] == 'healthy'

    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Test a search query"""
    print("\n" + "="*80)
    print(f"🔍 Testing Search: '{query}'")
    print("="*80)

    try:
        payload = {
            "query": query,
            "top_k": top_k
        }

        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/search",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        elapsed = (time.time() - start_time) * 1000

        response.raise_for_status()
        data = response.json()

        print(f"\n✅ Search completed in {elapsed:.0f}ms")
        print(f"   Query Type: {data['metadata']['query_type']}")
        print(f"   Answer Type: {data['metadata']['answer_type']}")
        print(f"   Results: {len(data['results'])}")

        print(f"\n📊 Top Results:")
        for i, result in enumerate(data['results'][:3], 1):
            print(f"\n   [{i}] {result['title'][:60]}...")
            print(f"       Score: {result['score']:.3f}")
            print(f"       Source: {result['source']}")
            print(f"       URL: {result['url'][:70]}...")
            print(f"       Snippet: {result['snippet'][:100]}...")

        return data

    except Exception as e:
        print(f"❌ Search failed: {e}")
        return None


def run_benchmark_tests():
    """Run a set of benchmark queries"""
    print("\n" + "="*80)
    print("🧪 Running Benchmark Tests")
    print("="*80)

    test_queries = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "When was Python programming language created?",
        "How to bake a chocolate cake?",
        "Best practices for REST API design",
    ]

    results = []
    for query in test_queries:
        result = test_search(query, top_k=5)
        if result:
            results.append({
                'query': query,
                'search_time_ms': result['metadata']['search_time_ms'],
                'num_results': len(result['results']),
                'avg_score': sum(r['score'] for r in result['results']) / len(result['results'])
            })
        time.sleep(1)  # Rate limiting

    # Summary
    if results:
        print("\n" + "="*80)
        print("📊 BENCHMARK SUMMARY")
        print("="*80)

        avg_time = sum(r['search_time_ms'] for r in results) / len(results)
        avg_score = sum(r['avg_score'] for r in results) / len(results)

        print(f"\nQueries tested: {len(results)}")
        print(f"Average search time: {avg_time:.0f}ms")
        print(f"Average result score: {avg_score:.3f}")

        print("\n✅ All tests passed!" if len(results) == len(test_queries) else "\n⚠️  Some tests failed")


def main():
    """Main test runner"""
    print("="*80)
    print("🚀 Adaptive Search API Test Suite")
    print("="*80)
    print(f"\nTarget: {API_BASE_URL}")

    # Test 1: Health Check
    if not test_health_check():
        print("\n❌ Health check failed. Is the API running?")
        print("   Start the API with: python api.py")
        return

    print("\n✅ API is healthy!")

    # Test 2: Basic search tests
    test_search("What is machine learning?")
    time.sleep(1)

    test_search("capital of Japan")
    time.sleep(1)

    # Test 3: Run benchmark suite
    run_benchmark_tests()

    print("\n" + "="*80)
    print("🎉 Test Suite Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
