# Documentation

Welcome to the Codebase Contextifier 9000 documentation!

## Getting Started

- **[Quick Start Guide](QUICKSTART.md)** - Get up and running in 5 minutes
- **[Main README](../README.md)** - Complete feature overview and reference

## Deep Dives

- **[File Watcher](FILE_WATCHER.md)** - Real-time file system monitoring and automatic re-indexing
- **[AST Code Chunking](AST_codeChunking.md)** - Research and methodology behind semantic code search

## Architecture

### System Components

```
FastMCP Server
├── Indexer
│   ├── AST Chunker (tree-sitter)
│   ├── Embeddings (Ollama)
│   ├── Merkle Tree (incremental indexing)
│   └── File Watcher (real-time updates)
├── Vector DB (Qdrant)
└── MCP Tools
    ├── index_repository
    ├── search_code
    ├── get_symbols
    ├── get_indexing_status
    ├── get_watcher_status
    ├── clear_index
    └── health_check
```

### Key Technologies

- **Tree-sitter**: Multi-language AST parsing
- **Qdrant**: Vector database for semantic search
- **Ollama**: Local embedding generation
- **watchdog**: File system monitoring
- **FastMCP**: Model Context Protocol server
- **Docker**: Containerized deployment

## Configuration

See [main README](../README.md#configuration) for all environment variables.

Key settings:
- `CODEBASE_PATH` - Your project directory
- `ENABLE_FILE_WATCHER` - Real-time updates (default: true)
- `EMBEDDING_MODEL` - Ollama model to use (default: nomic-embed-text)

## Troubleshooting

Common issues and solutions are documented in the [main README](../README.md#troubleshooting).

For file watcher specific issues, see [FILE_WATCHER.md](FILE_WATCHER.md#troubleshooting).

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

## Support

- GitHub Issues: Report bugs or request features
- Discussions: Ask questions and share ideas

## Research & References

The [AST Code Chunking](AST_codeChunking.md) document contains extensive research on:
- State-of-the-art semantic code search
- AST-aware chunking methodology
- Vector database selection
- Local LLM recommendations
- Incremental indexing strategies
