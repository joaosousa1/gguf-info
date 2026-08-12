# gguf-info

Simple tool to read metadata from GGUF model files.

## Features

- Reads GGUF header information
- Displays all metadata key-value pairs
- Shows a summary with common fields (architecture, layers, context length, etc.)
- Handles large arrays gracefully (shows size instead of full content)

## Requirements

- Python 3.6+

## Usage

```bash
python gguf-info.py model.gguf
