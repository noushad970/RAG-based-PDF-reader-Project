"""
Simple Local RAG - Web UI Launcher

Runs the FastAPI web server. Defaults to port 8000, with automatic fallback
or custom port configuration.
"""

import sys
import socket
import argparse
import uvicorn
from src.server import app


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Checks if a network port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def find_available_port(start_port: int = 8000, max_attempts: int = 10, host: str = "127.0.0.1") -> int:
    """Finds the first available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port, host):
            return port
    return start_port


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Local PDF RAG Web UI")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    args = parser.parse_args()

    host = args.host
    port = args.port

    if is_port_in_use(port, host):
        available = find_available_port(port, host=host)
        print(f"\n[!] Note: Port {port} was in use. Automatically switching to port {available}.\n")
        port = available

    print("=" * 55)
    print(f"  Starting Local PDF RAG Web UI on http://{host}:{port}")
    print("=" * 55)
    uvicorn.run("src.server:app", host=host, port=port, reload=False)
