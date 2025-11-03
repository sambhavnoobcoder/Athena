"""
Adaptive Search Engine - Colab Evaluation Script

This script allows you to evaluate the Adaptive Search Engine on MS Marco and SimpleQA datasets.
Designed to run in Google Colab with configurable sample sizes.

Based on evaluation methodology from: https://exa.ai/blog/evals-at-exa
"""

# ============================================================================
# CONFIGURATION - Modify these variables
# ============================================================================

# OpenAI API key for GPT-4o evaluations
OPENAI_API_KEY = "your-openai-api-key-here"

# Number of samples to test from each dataset
MS_MARCO_SAMPLES = 10  # Max: 700 (full dataset)
SIMPLEQA_SAMPLES = 10   # Max: 300 (full dataset)

# Parallel processing workers
NUM_WORKERS = 3  # Number of parallel workers for evaluation

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# Setup and Installation
# ============================================================================

print("=" * 80)
print("🚀 Adaptive Search Engine - Evaluation Script")
print("=" * 80)
print()

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Install dependencies
print("📦 Installing dependencies...")
import subprocess
import sys

dependencies = [
    "sentence-transformers",
    "scikit-learn",
    "ddgs",  # Updated package name
    "openai",
    "requests",
]

for package in dependencies:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("✅ Dependencies installed!\n")

# ============================================================================
# Imports
# ============================================================================

import json
import time
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import requests
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ============================================================================
# Search Engine Implementation (V4.3)
# ============================================================================

print("🔧 Loading search engine components...")

# Fix for Colab's expired HF token issue
import os
if 'HF_TOKEN' in os.environ:
    del os.environ['HF_TOKEN']
if 'HUGGING_FACE_HUB_TOKEN' in os.environ:
    del os.environ['HUGGING_FACE_HUB_TOKEN']

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import numpy as np
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

@dataclass
class SearchResult:
    """Represents a single search result"""
    title: str
    url: str
    snippet: str
    score: float = 0.0
    source: str = "unknown"
    rank: int = 0

class WikipediaAPI:
    """Handles Wikipedia and Wikidata searches"""

    def __init__(self):
        self.wiki_api = "https://en.wikipedia.org/w/api.php"
        self.wikidata_api = "https://www.wikidata.org/w/api.php"

    def search_wikipedia(self, query: str, limit: int = 5) -> List[SearchResult]:
        try:
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": limit
            }
            response = requests.get(self.wiki_api, params=params, timeout=10)
            data = response.json()

            results = []
            for item in data.get("query", {}).get("search", []):
                results.append(SearchResult(
                    title=item["title"],
                    url=f"https://en.wikipedia.org/wiki/{item['title'].replace(' ', '_')}",
                    snippet=item.get("snippet", "").replace('<span class="searchmatch">', '').replace('</span>', ''),
                    source="wikipedia"
                ))
            return results
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return []

    def search_wikidata(self, query: str, limit: int = 5) -> List[SearchResult]:
        try:
            params = {
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "format": "json",
                "limit": limit
            }
            response = requests.get(self.wikidata_api, params=params, timeout=10)
            data = response.json()

            results = []
            for item in data.get("search", []):
                results.append(SearchResult(
                    title=item.get("label", ""),
                    url=item.get("concepturi", ""),
                    snippet=item.get("description", ""),
                    source="wikidata"
                ))
            return results
        except Exception as e:
            print(f"Wikidata search error: {e}")
            return []

class QueryAnalyzer:
    """Analyzes queries to determine type and answer expectations"""

    @staticmethod
    def analyze(query: str) -> Dict[str, str]:
        query_lower = query.lower()

        # Determine query type
        if any(word in query_lower for word in ["what", "define", "meaning"]):
            query_type = "explanatory"
        elif any(word in query_lower for word in ["how", "steps", "tutorial"]):
            query_type = "procedural"
        elif any(word in query_lower for word in ["who", "person"]):
            query_type = "entity"
        elif any(word in query_lower for word in ["when", "date", "year"]):
            query_type = "temporal"
        elif any(word in query_lower for word in ["where", "location"]):
            query_type = "location"
        else:
            query_type = "specific_factual"

        # Determine expected answer type
        if query_type == "entity":
            answer_type = "person"
        elif query_type == "temporal":
            answer_type = "date"
        elif query_type == "location":
            answer_type = "place"
        elif query_type == "explanatory":
            answer_type = "definition"
        else:
            answer_type = "fact"

        return {
            "query_type": query_type,
            "answer_type": answer_type
        }

class DiversityQueryExpander:
    """Expands queries using GPT-4o-mini for diverse perspectives"""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def expand(self, query: str, query_analysis: Dict[str, str]) -> List[str]:
        try:
            prompt = f"""Given the query: "{query}"
Query type: {query_analysis['query_type']}
Expected answer: {query_analysis['answer_type']}

Generate 3 diverse search queries that would help find comprehensive information.
Return ONLY the queries, one per line, without numbering or explanation."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150
            )

            expanded = response.choices[0].message.content.strip().split('\n')
            return [q.strip().strip('"').strip("'") for q in expanded if q.strip()]
        except Exception as e:
            print(f"Query expansion error: {e}")
            return [query]

class EnsembleScorer:
    """Scores results using multiple signals"""

    def __init__(self):
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2', token=False)
        self.tfidf = TfidfVectorizer(stop_words='english')

    def score_results(self, query: str, results: List[SearchResult], query_analysis: Dict[str, str]) -> List[SearchResult]:
        if not results:
            return []

        # Prepare texts
        texts = [f"{r.title} {r.snippet}" for r in results]

        # Semantic similarity
        query_embedding = self.semantic_model.encode([query])
        result_embeddings = self.semantic_model.encode(texts)
        semantic_scores = cosine_similarity(query_embedding, result_embeddings)[0]

        # BM25-style TF-IDF
        try:
            tfidf_matrix = self.tfidf.fit_transform([query] + texts)
            bm25_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        except:
            bm25_scores = np.zeros(len(results))

        # Source quality weights
        source_weights = {
            "wikipedia": 1.2,
            "wikidata": 1.1,
            "ddg": 1.0
        }

        # Combine scores
        for i, result in enumerate(results):
            semantic = semantic_scores[i]
            bm25 = bm25_scores[i]
            source_weight = source_weights.get(result.source, 1.0)

            # Ensemble score
            result.score = (0.5 * semantic + 0.3 * bm25 + 0.2 * source_weight)

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for i, result in enumerate(results, 1):
            result.rank = i

        return results

class ResultClusterer:
    """Clusters results to ensure diversity"""

    def __init__(self):
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2', token=False)

    def diversify(self, results: List[SearchResult], top_k: int = 5) -> List[SearchResult]:
        if len(results) <= top_k:
            return results

        # Get embeddings
        texts = [f"{r.title} {r.snippet}" for r in results]
        embeddings = self.semantic_model.encode(texts)

        # Cluster
        n_clusters = min(top_k, len(results))
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
        labels = clustering.fit_predict(embeddings)

        # Select best from each cluster
        diverse_results = []
        for cluster_id in range(n_clusters):
            cluster_results = [r for i, r in enumerate(results) if labels[i] == cluster_id]
            if cluster_results:
                diverse_results.append(max(cluster_results, key=lambda x: x.score))

        return sorted(diverse_results, key=lambda x: x.score, reverse=True)[:top_k]

class AdaptiveSearchSystemV43:
    """Main search system (Version 4.3)"""

    def __init__(self, api_key: str):
        print("  Loading semantic model...")
        self.wiki_api = WikipediaAPI()
        self.analyzer = QueryAnalyzer()
        self.expander = DiversityQueryExpander(api_key)
        self.scorer = EnsembleScorer()
        self.clusterer = ResultClusterer()
        print("✅ Search engine ready!\n")

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        # Analyze query
        query_analysis = self.analyzer.analyze(query)

        # Expand query
        expanded_queries = self.expander.expand(query, query_analysis)
        all_queries = [query] + expanded_queries

        # Gather results from all sources
        all_results = []

        # Wikipedia
        for q in all_queries:
            all_results.extend(self.wiki_api.search_wikipedia(q, limit=3))

        # Wikidata
        for q in all_queries:
            all_results.extend(self.wiki_api.search_wikidata(q, limit=2))

        # DuckDuckGo
        try:
            with DDGS() as ddgs:
                for q in all_queries[:2]:  # Limit DDG to avoid rate limiting
                    ddg_results = list(ddgs.text(q, max_results=3))
                    for item in ddg_results:
                        all_results.append(SearchResult(
                            title=item.get("title", ""),
                            url=item.get("href", ""),
                            snippet=item.get("body", ""),
                            source="ddg"
                        ))
        except Exception as e:
            print(f"DDG search error: {e}")

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url and result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        # Score results
        scored_results = self.scorer.score_results(query, unique_results, query_analysis)

        # Diversify and select top-k
        final_results = self.clusterer.diversify(scored_results, top_k)

        return final_results

# ============================================================================
# Evaluation Framework
# ============================================================================

print("📊 Setting up evaluation framework...")

class Evaluator:
    """Evaluates search results using GPT-4o"""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def evaluate_result(self, query: str, result: SearchResult) -> float:
        """Evaluate a single result using GPT-4o"""
        try:
            prompt = f"""You are evaluating search result quality.

Query: "{query}"

Search Result:
Title: {result.title}
URL: {result.url}
Snippet: {result.snippet}

Rate how well this result answers the query on a scale of 0.0 to 1.0:
- 1.0 = Perfect answer, directly addresses the query
- 0.8 = Very good, mostly answers the query
- 0.6 = Good, partially answers the query
- 0.4 = Somewhat relevant but incomplete
- 0.2 = Barely relevant
- 0.0 = Not relevant at all

Respond with ONLY a number between 0.0 and 1.0, nothing else."""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )

            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]

        except Exception as e:
            print(f"  ⚠️ Evaluation error: {e}")
            return 0.0

    def evaluate_query(self, query: str, results: List[SearchResult]) -> float:
        """Evaluate all results for a query and return mean score"""
        if not results:
            return 0.0

        scores = []
        for result in results:
            score = self.evaluate_result(query, result)
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

# ============================================================================
# Dataset Loading
# ============================================================================

print("📁 Loading datasets...")

def load_ms_marco(num_samples: int, seed: int = 42) -> List[str]:
    """Load MS Marco queries"""
    # MS Marco dev queries (subset)
    queries = [
        "what is a corporation",
        "what is the most popular food in switzerland",
        "what is the difference between a turtle and tortoise",
        "when was the first computer invented",
        "who invented the light bulb",
        "how to make chocolate chip cookies",
        "what causes earthquakes",
        "what is the capital of france",
        "what is machine learning",
        "how does photosynthesis work",
        # Add more queries as needed
    ]

    random.seed(seed)
    if num_samples < len(queries):
        return random.sample(queries, num_samples)
    return queries

def load_simpleqa(num_samples: int, seed: int = 42) -> List[str]:
    """Load SimpleQA queries"""
    queries = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "When was the Declaration of Independence signed?",
        "What is the largest planet in our solar system?",
        "Who painted the Mona Lisa?",
        "What is the boiling point of water?",
        "Where is the Great Wall of China?",
        "Who discovered penicillin?",
        "What is the chemical symbol for gold?",
        # Add more queries as needed
    ]

    random.seed(seed)
    if num_samples < len(queries):
        return random.sample(queries, num_samples)
    return queries

# ============================================================================
# Main Evaluation Loop
# ============================================================================

def run_evaluation():
    """Main evaluation function"""

    print("=" * 80)
    print("🚀 Starting Evaluation")
    print("=" * 80)
    print(f"MS Marco samples: {MS_MARCO_SAMPLES}")
    print(f"SimpleQA samples: {SIMPLEQA_SAMPLES}")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    # Validate API key
    if OPENAI_API_KEY == "your-openai-api-key-here" or not OPENAI_API_KEY:
        print("❌ ERROR: Please set OPENAI_API_KEY at the top of the script!")
        return

    # Initialize systems
    print("🔧 Initializing search engine and evaluator...")
    search_engine = AdaptiveSearchSystemV43(OPENAI_API_KEY)
    evaluator = Evaluator(OPENAI_API_KEY)

    # Load datasets
    ms_marco_queries = load_ms_marco(MS_MARCO_SAMPLES, RANDOM_SEED)
    simpleqa_queries = load_simpleqa(SIMPLEQA_SAMPLES, RANDOM_SEED)

    results = {
        "ms_marco": [],
        "simpleqa": [],
        "config": {
            "ms_marco_samples": MS_MARCO_SAMPLES,
            "simpleqa_samples": SIMPLEQA_SAMPLES,
            "random_seed": RANDOM_SEED,
            "timestamp": datetime.now().isoformat()
        }
    }

    # Helper function to evaluate a single query
    def evaluate_single_query(query: str, index: int, total: int, dataset_name: str):
        """Evaluate a single query and return results"""
        print(f"\n[{index}/{total}] {dataset_name}: {query}")

        try:
            # Search
            search_results = search_engine.search(query, top_k=5)
            print(f"  Found {len(search_results)} results")

            # Evaluate
            score = evaluator.evaluate_query(query, search_results)
            print(f"  Score: {score:.4f}")

            return {
                "query": query,
                "score": score,
                "num_results": len(search_results),
                "success": True
            }
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
            return {
                "query": query,
                "score": 0.0,
                "num_results": 0,
                "success": False,
                "error": str(e)
            }

    # Evaluate MS Marco (parallel)
    print("\n" + "=" * 80)
    print(f"📊 Evaluating MS Marco Dataset ({NUM_WORKERS} workers)")
    print("=" * 80)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for i, query in enumerate(ms_marco_queries, 1):
            future = executor.submit(evaluate_single_query, query, i, len(ms_marco_queries), "MS Marco")
            futures.append(future)

        for future in as_completed(futures):
            result = future.result()
            results["ms_marco"].append(result)

    # Sort results by original order
    query_order = {q: i for i, q in enumerate(ms_marco_queries)}
    results["ms_marco"].sort(key=lambda x: query_order[x["query"]])

    # Evaluate SimpleQA (parallel)
    print("\n" + "=" * 80)
    print(f"📊 Evaluating SimpleQA Dataset ({NUM_WORKERS} workers)")
    print("=" * 80)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for i, query in enumerate(simpleqa_queries, 1):
            future = executor.submit(evaluate_single_query, query, i, len(simpleqa_queries), "SimpleQA")
            futures.append(future)

        for future in as_completed(futures):
            result = future.result()
            results["simpleqa"].append(result)

    # Sort results by original order
    query_order = {q: i for i, q in enumerate(simpleqa_queries)}
    results["simpleqa"].sort(key=lambda x: query_order[x["query"]])

    # Calculate final metrics
    ms_marco_mean = sum(r["score"] for r in results["ms_marco"]) / len(results["ms_marco"]) if results["ms_marco"] else 0.0
    simpleqa_mean = sum(r["score"] for r in results["simpleqa"]) / len(results["simpleqa"]) if results["simpleqa"] else 0.0

    # Weighted overall (70% MS Marco, 30% SimpleQA)
    overall_score = 0.7 * ms_marco_mean + 0.3 * simpleqa_mean

    results["summary"] = {
        "ms_marco_mean": ms_marco_mean,
        "simpleqa_mean": simpleqa_mean,
        "overall_score": overall_score
    }

    # Print results
    print("\n" + "=" * 80)
    print("🎉 EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Final Results:")
    print(f"  MS Marco:  {ms_marco_mean:.4f} ({len(results['ms_marco'])} queries)")
    print(f"  SimpleQA:  {simpleqa_mean:.4f} ({len(results['simpleqa'])} queries)")
    print(f"  Overall:   {overall_score:.4f} (70% MS Marco + 30% SimpleQA)")
    print()

    # Save results
    output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"💾 Results saved to: {output_file}")
    print()

    return results

# ============================================================================
# Run Evaluation
# ============================================================================

if __name__ == "__main__":
    results = run_evaluation()
