# Testing Instructions for Adaptive Search API

## Quick Start Test

### Terminal 1: Start the API

```bash
cd /Users/sambhavdixit/Desktop/poker-couter/adaptive-search-api
source ../venv/bin/activate
python api.py
```

**Expected Output:**
```
🚀 Starting Adaptive Search API...
📦 Loading search engine (this may take a moment)...
  Loading semantic model...
  Initializing free data sources...
  Setting up algorithmic components...
✅ V4.3 ready! All systems operational.

INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
✅ Search engine loaded successfully!
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Terminal 2: Run Tests

```bash
cd /Users/sambhavdixit/Desktop/poker-couter/adaptive-search-api
source ../venv/bin/activate
python test_api.py
```

**Expected Output:**
```
================================================================================
🚀 Adaptive Search API Test Suite
================================================================================

Target: http://localhost:8000

================================================================================
🏥 Testing Health Check Endpoint
================================================================================
✅ Status: healthy
   Version: 4.3.0
   Uptime: 5.23s
   Engine Loaded: True

✅ API is healthy!

================================================================================
🔍 Testing Search: 'What is machine learning?'
================================================================================

✅ Search completed in 2345ms
   Query Type: explanatory
   Answer Type: definition
   Results: 5

📊 Top Results:

   [1] Machine learning - Wikipedia...
       Score: 0.950
       Source: wikipedia
       URL: https://en.wikipedia.org/wiki/Machine_learning...
       Snippet: Machine learning is a field of inquiry devoted to understanding...

   [2] What is Machine Learning? | IBM...
       Score: 0.920
       Source: ddg
       URL: https://www.ibm.com/topics/machine-learning...
       Snippet: Machine learning is a branch of artificial intelligence...

... (more tests)

================================================================================
🎉 Test Suite Complete!
================================================================================
```

## Manual API Testing

### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Expected:**
```json
{
  "status": "healthy",
  "version": "4.3.0",
  "uptime_seconds": 123.45,
  "engine_loaded": true,
  "timestamp": "2025-11-04T00:00:00Z"
}
```

### 2. Simple Search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "top_k": 5
  }'
```

**Expected:**
```json
{
  "query": "What is the capital of France?",
  "results": [
    {
      "title": "Paris - Capital of France",
      "url": "https://en.wikipedia.org/wiki/Paris",
      "snippet": "Paris is the capital and most populous city...",
      "score": 0.95,
      "source": "wikipedia",
      "rank": 1
    }
  ],
  "metadata": {
    "query_type": "specific_factual",
    "answer_type": "location",
    "best_source": "wikipedia",
    "difficulty": "easy",
    "total_results": 5,
    "search_time_ms": 1234,
    "timestamp": "2025-11-04T00:00:00Z"
  }
}
```

### 3. Interactive Documentation

Open in your browser:
- **Swagger UI**: http://localhost:8000/docs
- Try the `/search` endpoint interactively!

## Performance Testing

### Benchmark Multiple Queries

```bash
# Create a test script
cat > benchmark_test.sh << 'EOF'
#!/bin/bash

queries=(
  "What is Python?"
  "How to bake a cake?"
  "Best practices for REST APIs"
  "Capital of Japan"
  "Who wrote Hamlet?"
)

for query in "${queries[@]}"; do
  echo "Testing: $query"
  curl -s -X POST http://localhost:8000/search \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"$query\", \"top_k\": 5}" \
    | jq '.metadata.search_time_ms'
  sleep 1
done
EOF

chmod +x benchmark_test.sh
./benchmark_test.sh
```

## Troubleshooting

### API Won't Start

**Check:**
1. Is port 8000 already in use?
   ```bash
   lsof -i :8000
   ```
2. Is the virtual environment activated?
   ```bash
   which python  # Should point to venv
   ```
3. Are dependencies installed?
   ```bash
   pip list | grep fastapi
   ```

### Slow First Request

**This is normal!** The first request loads the sentence transformer model (~100MB). Subsequent requests will be fast (1-3 seconds).

### Search Returns Empty Results

**Possible causes:**
1. DuckDuckGo rate limiting (wait a moment)
2. Wikipedia API error (check logs)
3. Network connectivity issue

**Check logs:**
```bash
# Look for error messages in the API output
```

### Memory Issues

If the API crashes with memory errors:
1. Close other applications
2. Reduce MAX_WORKERS to 1 in `.env`
3. Ensure you have at least 2GB free RAM

## Success Criteria

✅ **The API is working correctly if:**

1. Health check returns `"status": "healthy"`
2. Search queries return 3-5 results
3. Results have scores > 0.5
4. Search completes in < 5 seconds (after model load)
5. Multiple searches work without errors

## What to Test

### Basic Functionality
- [ ] Health endpoint responds
- [ ] Search endpoint accepts queries
- [ ] Results are returned with scores
- [ ] Multiple queries work consecutively

### Query Types
- [ ] Factual queries ("What is X?")
- [ ] How-to queries ("How to X?")
- [ ] Entity queries ("Who is X?")
- [ ] Location queries ("Where is X?")

### Edge Cases
- [ ] Empty query (should fail gracefully)
- [ ] Very long query (should truncate)
- [ ] Special characters in query
- [ ] Rapid consecutive requests

### Performance
- [ ] First request completes (even if slow)
- [ ] Subsequent requests are faster
- [ ] API doesn't crash under load
- [ ] Memory usage stays reasonable

## Expected Performance

**After Model Load:**
- Simple queries: 1-2 seconds
- Complex queries: 2-4 seconds
- Average score: 0.6-0.8
- Results per query: 5

**Resource Usage:**
- RAM: ~500MB-1GB
- CPU: Moderate during search, idle otherwise
- Network: Minimal (free API calls)

## Ready for GitHub

Once all tests pass, the API is ready to be pushed to GitHub!

```bash
cd adaptive-search-api
git init
git add .
git commit -m "Initial commit: Adaptive Search API v4.3"
git remote add origin https://github.com/yourusername/adaptive-search-api.git
git push -u origin main
```
