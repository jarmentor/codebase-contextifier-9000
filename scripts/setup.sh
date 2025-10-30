#!/bin/bash

# Codebase Contextifier 9000 Setup Script
set -e

echo "=========================================="
echo "Codebase Contextifier 9000 Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
echo "Checking prerequisites..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker Desktop from https://www.docker.com/products/docker-desktop"
    exit 1
fi
echo -e "${GREEN}✓ Docker installed${NC}"

# Check if Docker Compose is available
if ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not available${NC}"
    echo "Please install Docker Compose or update Docker Desktop"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose available${NC}"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}Warning: Ollama is not installed${NC}"
    echo "Please install Ollama from https://ollama.ai"
    echo "After installation, run: ollama pull nomic-embed-text"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ Ollama installed${NC}"

    # Check if embedding model is available
    if ollama list | grep -q "nomic-embed-text"; then
        echo -e "${GREEN}✓ nomic-embed-text model available${NC}"
    else
        echo -e "${YELLOW}Warning: nomic-embed-text model not found${NC}"
        read -p "Would you like to pull it now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Pulling nomic-embed-text model..."
            ollama pull nomic-embed-text
            echo -e "${GREEN}✓ Model downloaded${NC}"
        fi
    fi
fi

echo ""
echo "=========================================="
echo "Configuration"
echo "=========================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
    echo ""
    echo -e "${YELLOW}Important: Please edit .env and set CODEBASE_PATH to your project directory${NC}"
    echo ""
    read -p "Press Enter to open .env for editing..."
    ${EDITOR:-nano} .env
else
    echo -e "${GREEN}✓ .env file exists${NC}"
fi

echo ""
echo "=========================================="
echo "Building Docker Image"
echo "=========================================="

# Build the Docker image
echo "Building Docker image (this may take a few minutes)..."
docker-compose build

echo -e "${GREEN}✓ Docker image built successfully${NC}"

echo ""
echo "=========================================="
echo "Starting Services"
echo "=========================================="

# Start the services
echo "Starting Qdrant and MCP server..."
docker-compose up -d

echo -e "${GREEN}✓ Services started${NC}"

# Wait for services to be healthy
echo ""
echo "Waiting for services to be healthy..."
sleep 5

# Check if Qdrant is healthy
if docker-compose ps | grep -q "qdrant.*healthy"; then
    echo -e "${GREEN}✓ Qdrant is healthy${NC}"
else
    echo -e "${YELLOW}⚠ Qdrant may still be starting up${NC}"
fi

# Check if MCP server is running
if docker-compose ps | grep -q "mcp-server.*running"; then
    echo -e "${GREEN}✓ MCP server is running${NC}"
else
    echo -e "${YELLOW}⚠ MCP server may still be starting up${NC}"
fi

echo ""
echo "=========================================="
echo "Claude Desktop Configuration"
echo "=========================================="

# Determine Claude Desktop config path based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    CLAUDE_CONFIG_DIR="$HOME/Library/Application Support/Claude"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    CLAUDE_CONFIG_DIR="$APPDATA/Claude"
else
    CLAUDE_CONFIG_DIR="$HOME/.config/Claude"
fi

CLAUDE_CONFIG_FILE="$CLAUDE_CONFIG_DIR/claude_desktop_config.json"

echo "Claude Desktop configuration location:"
echo "$CLAUDE_CONFIG_FILE"
echo ""

if [ -f "$CLAUDE_CONFIG_FILE" ]; then
    echo -e "${YELLOW}Note: Claude Desktop config already exists${NC}"
    echo "You'll need to manually merge the configuration from:"
    echo "  claude_desktop_config.example.json"
else
    echo "Would you like to create the Claude Desktop configuration?"
    read -p "(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p "$CLAUDE_CONFIG_DIR"
        cp claude_desktop_config.example.json "$CLAUDE_CONFIG_FILE"
        echo -e "${GREEN}✓ Claude Desktop configuration created${NC}"
        echo ""
        echo -e "${YELLOW}Please restart Claude Desktop for changes to take effect${NC}"
    fi
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Make sure CODEBASE_PATH in .env points to your project"
echo "2. Restart Claude Desktop if you configured it"
echo "3. In Claude Desktop, try: 'Index the repository at /workspace'"
echo ""
echo "Useful commands:"
echo "  docker-compose logs -f    # View logs"
echo "  docker-compose restart    # Restart services"
echo "  docker-compose down       # Stop services"
echo ""
echo "For more information, see README.md"
echo ""
