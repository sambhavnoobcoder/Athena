# Setup Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 2GB RAM minimum
- OpenAI API key

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/adaptive-search-api.git
cd adaptive-search-api
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- FastAPI (web framework)
- Uvicorn (ASGI server)
- Sentence Transformers (semantic search)
- DuckDuckGo Search (free search API)
- OpenAI (query expansion)
- And other dependencies

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-actual-key-here
```

### 5. Start the API

```bash
python api.py
```

You should see:

```
🚀 Starting Adaptive Search API...
📦 Loading search engine (this may take a moment)...
  Loading semantic model...
  Initializing free data sources...
  Setting up algorithmic components...
✅ V4.3 ready! All systems operational.

INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 6. Test the API

Open a new terminal and run:

```bash
python test_api.py
```

Or test manually with curl:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?", "top_k": 5}'
```

### 7. Access Documentation

Open your browser and visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Docker Setup (Alternative)

### 1. Build the Docker Image

```bash
docker build -t adaptive-search-api .
```

### 2. Run the Container

```bash
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY="sk-your-key" \
  --name adaptive-search \
  adaptive-search-api
```

### 3. Check Logs

```bash
docker logs -f adaptive-search
```

### 4. Test

```bash
curl http://localhost:8000/health
```

## Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution:** Make sure you activated the virtual environment and installed dependencies:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "OpenAI API key not set"

**Solution:** Set the environment variable:

```bash
export OPENAI_API_KEY="sk-your-key"
```

Or add it to your `.env` file.

### Issue: "Port 8000 already in use"

**Solution:** Change the port in `.env`:

```env
API_PORT=8001
```

### Issue: "Slow first request"

**Solution:** This is normal! The first request loads the sentence transformer model (~100MB). Subsequent requests will be fast.

### Issue: "Rate limited by DuckDuckGo"

**Solution:** The API includes automatic rate limiting and retry logic. If you're making many requests, add delays between calls or reduce concurrency.

## Performance Optimization

### 1. Use Multiple Workers

```bash
# In .env
MAX_WORKERS=4
```

⚠️ **Note:** Each worker loads its own model (~500MB RAM per worker)

### 2. Enable Caching

```bash
# In .env
CACHE_SIZE=1000
CACHE_TTL_SECONDS=3600
```

### 3. Adjust Rate Limits

```bash
# In .env
RATE_LIMIT_PER_MINUTE=120
```

## Production Deployment

### Using Systemd (Linux)

Create `/etc/systemd/system/adaptive-search.service`:

```ini
[Unit]
Description=Adaptive Search API
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/adaptive-search-api
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python api.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable adaptive-search
sudo systemctl start adaptive-search
```

### Using Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - API_HOST=0.0.0.0
      - API_PORT=8000
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Run:

```bash
docker-compose up -d
```

### Behind Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Next Steps

- Read the [API Documentation](README.md#api-documentation)
- Run [Benchmarks](README.md#benchmarking)
- Check out [Contributing Guidelines](CONTRIBUTING.md)

## Support

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Search [GitHub Issues](https://github.com/yourusername/adaptive-search-api/issues)
3. Create a new issue with:
   - Your OS and Python version
   - Full error message
   - Steps to reproduce
