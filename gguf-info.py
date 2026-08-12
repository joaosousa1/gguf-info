#!/usr/bin/env python3
"""
GGUF Metadata Reader
Usage: python gguf_meta.py model.gguf
"""

import struct
import sys
from pathlib import Path

# GGUF value types
GGUF_TYPE = {
    0: "UINT8",
    1: "INT8",
    2: "UINT16",
    3: "INT16",
    4: "UINT32",
    5: "INT32",
    6: "FLOAT32",
    7: "BOOL",
    8: "STRING",
    9: "ARRAY",
    10: "UINT64",
    11: "INT64",
    12: "FLOAT64",
}


def read_string(f):
    """Read a GGUF string (uint64 length + bytes)"""
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def read_value(f, value_type):
    """Read a value according to the GGUF type"""
    if value_type == 0:  # UINT8
        return struct.unpack("<B", f.read(1))[0]
    elif value_type == 1:  # INT8
        return struct.unpack("<b", f.read(1))[0]
    elif value_type == 2:  # UINT16
        return struct.unpack("<H", f.read(2))[0]
    elif value_type == 3:  # INT16
        return struct.unpack("<h", f.read(2))[0]
    elif value_type == 4:  # UINT32
        return struct.unpack("<I", f.read(4))[0]
    elif value_type == 5:  # INT32
        return struct.unpack("<i", f.read(4))[0]
    elif value_type == 6:  # FLOAT32
        return struct.unpack("<f", f.read(4))[0]
    elif value_type == 7:  # BOOL
        return bool(struct.unpack("<B", f.read(1))[0])
    elif value_type == 8:  # STRING
        return read_string(f)
    elif value_type == 9:  # ARRAY
        array_type = struct.unpack("<I", f.read(4))[0]
        array_len = struct.unpack("<Q", f.read(8))[0]
        # For large arrays (e.g. tokens), only show the size
        if array_len > 50:
            # Skip the array content
            for _ in range(array_len):
                read_value(f, array_type)
            return f"<array of {array_len} items of type {GGUF_TYPE.get(array_type, array_type)}>"
        else:
            return [read_value(f, array_type) for _ in range(array_len)]
    elif value_type == 10:  # UINT64
        return struct.unpack("<Q", f.read(8))[0]
    elif value_type == 11:  # INT64
        return struct.unpack("<q", f.read(8))[0]
    elif value_type == 12:  # FLOAT64
        return struct.unpack("<d", f.read(8))[0]
    else:
        raise ValueError(f"Unknown type: {value_type}")


def read_gguf_metadata(filepath):
    """Read only the header and metadata of a GGUF file"""
    with open(filepath, "rb") as f:
        # === Header ===
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError("Not a valid GGUF file (incorrect magic number)")

        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]

        print("=" * 60)
        print(f"File          : {Path(filepath).name}")
        print(f"GGUF Version  : {version}")
        print(f"Tensors       : {tensor_count}")
        print(f"Metadata      : {kv_count} keys")
        print("=" * 60)
        print()

        # === Metadata (Key-Value) ===
        metadata = {}
        for i in range(kv_count):
            key = read_string(f)
            value_type = struct.unpack("<I", f.read(4))[0]
            value = read_value(f, value_type)
            metadata[key] = value

            # Nice formatting
            type_name = GGUF_TYPE.get(value_type, str(value_type))
            if isinstance(value, str) and len(value) > 80:
                display = value[:77] + "..."
            else:
                display = value
            print(f"{key:45} ({type_name:8}) = {display}")

        print()
        print("=" * 60)

        # Important highlights
        arch = metadata.get("general.architecture", "?")
        name = metadata.get("general.name", "?")
        layers = metadata.get(f"{arch}.block_count") or metadata.get("llama.block_count")
        ctx = metadata.get(f"{arch}.context_length") or metadata.get("llama.context_length")
        embd = metadata.get(f"{arch}.embedding_length") or metadata.get("llama.embedding_length")

        print("SUMMARY:")
        print(f"  Name            : {name}")
        print(f"  Architecture    : {arch}")
        if layers is not None:
            print(f"  Layers          : {layers}")
        if ctx is not None:
            print(f"  Context length  : {ctx}")
        if embd is not None:
            print(f"  Embedding       : {embd}")
        print("=" * 60)

        return metadata


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gguf_meta.py <file.gguf>")
        sys.exit(1)

    filepath = sys.argv[1]
    if not Path(filepath).exists():
        print(f"Error: file '{filepath}' not found.")
        sys.exit(1)

    try:
        read_gguf_metadata(filepath)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
