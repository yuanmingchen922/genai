#!/bin/bash

# Docker Build Script for GenAI API
# This script builds and runs the GenAI API Docker container

set -e

echo "=================================="
echo "GenAI API Docker Build Script"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

echo -e "${GREEN}Step 1: Stopping existing containers...${NC}"
docker-compose down 2>/dev/null || true
echo ""

echo -e "${GREEN}Step 2: Building Docker image...${NC}"
docker-compose build --no-cache
echo ""

echo -e "${GREEN}Step 3: Starting containers...${NC}"
docker-compose up -d
echo ""

echo -e "${GREEN}Step 4: Waiting for API to be ready...${NC}"
sleep 5

# Check if container is running
if docker ps | grep -q genai-api; then
    echo -e "${GREEN}Container is running!${NC}"
else
    echo -e "${RED}Container failed to start. Checking logs:${NC}"
    docker-compose logs
    exit 1
fi

echo ""
echo -e "${GREEN}Step 5: Testing API health endpoint...${NC}"
sleep 10

# Try to access health endpoint
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}API is healthy and responding!${NC}"
        break
    else
        if [ $i -eq 10 ]; then
            echo -e "${YELLOW}Warning: Health check failed after 10 attempts${NC}"
            echo -e "${YELLOW}Container logs:${NC}"
            docker-compose logs --tail=50
        else
            echo "Waiting for API to be ready... ($i/10)"
            sleep 3
        fi
    fi
done

echo ""
echo "=================================="
echo -e "${GREEN}Build Complete!${NC}"
echo "=================================="
echo ""
echo "API is available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Useful commands:"
echo "  View logs:     docker-compose logs -f"
echo "  Stop:          docker-compose down"
echo "  Restart:       docker-compose restart"
echo "  Shell access:  docker exec -it genai-api /bin/bash"
echo ""
echo "Test the GAN endpoint:"
echo "  curl -X POST http://localhost:8000/generate-digit \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"digit\": 5}'"
echo ""

