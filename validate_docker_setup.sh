#!/bin/bash

# Validation script for Docker setup
# Checks all required files and configurations before building

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

echo "=================================="
echo "Docker Setup Validation"
echo "=================================="
echo ""

# Check Docker
echo -n "Checking Docker installation... "
if command -v docker &> /dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "  Docker is not installed"
    ERRORS=$((ERRORS + 1))
fi

echo -n "Checking Docker daemon... "
if docker info &> /dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "  Docker daemon is not running"
    ERRORS=$((ERRORS + 1))
fi

# Check required files
echo ""
echo "Checking required files:"

FILES=(
    "Dockerfile"
    "docker-compose.yml"
    ".dockerignore"
    "requirements.txt"
    "project.toml"
    "app/main.py"
    "app/mnist_gan_model.py"
)

for file in "${FILES[@]}"; do
    echo -n "  $file... "
    if [ -f "$file" ]; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}MISSING${NC}"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check model files
echo ""
echo "Checking model files:"

MODEL_FILES=(
    "models/generator_gan.pth"
    "models/discriminator_gan.pth"
    "models/cnn_classifier.pth"
)

for file in "${MODEL_FILES[@]}"; do
    echo -n "  $file... "
    if [ -f "$file" ]; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}MISSING${NC}"
        echo "    Warning: Model file not found"
        WARNINGS=$((WARNINGS + 1))
    fi
done

# Check spaCy model in requirements.txt
echo ""
echo -n "Checking spaCy model in requirements.txt... "
if grep -q "en-core-web-lg @ https" requirements.txt; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "  SpaCy model URL not found in requirements.txt"
    ERRORS=$((ERRORS + 1))
fi

# Check Python version in project.toml
echo -n "Checking Python version in project.toml... "
if grep -q 'requires-python = ">=3.10"' project.toml; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${YELLOW}WARNING${NC}"
    echo "  Python version should be >=3.10 to match Dockerfile"
    WARNINGS=$((WARNINGS + 1))
fi

# Check health endpoint in main.py
echo -n "Checking health endpoint in main.py... "
if grep -q '@app.get("/health")' app/main.py; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "  Health endpoint not found in main.py"
    ERRORS=$((ERRORS + 1))
fi

# Check correct model paths in main.py
echo -n "Checking GAN model paths in main.py... "
if grep -q 'models/generator_gan.pth' app/main.py; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "  Incorrect model path in main.py (should be models/generator_gan.pth)"
    ERRORS=$((ERRORS + 1))
fi

# Summary
echo ""
echo "=================================="
echo "Validation Summary"
echo "=================================="

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All required checks passed!${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}$WARNINGS warning(s) found${NC}"
    fi
    echo ""
    echo "You can now build the Docker container:"
    echo "  ./docker-build.sh"
    exit 0
else
    echo -e "${RED}$ERRORS error(s) found${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}$WARNINGS warning(s) found${NC}"
    fi
    echo ""
    echo "Please fix the errors before building"
    exit 1
fi

