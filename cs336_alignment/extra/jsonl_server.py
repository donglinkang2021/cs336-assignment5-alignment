#!/usr/bin/env python3
"""
Simple HTTP server to view HTML files with API support
Usage: python server.py [port]
Default port: auto-detect starting from 8000
"""

import http.server
import socketserver
import sys
import os
import json
import urllib.parse
import socket
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

def find_free_port(start_port=8000, max_attempts=100):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find a free port in range {start_port}-{start_port + max_attempts}")

# Set port: use command line argument if provided, otherwise find a free port
if len(sys.argv) > 1:
    PORT = int(sys.argv[1])
else:
    PORT = find_free_port()
    console.print(f"[bold cyan]🔍 Auto-detected free port: {PORT}[/bold cyan]")

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow cross-origin access
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        # Handle API requests
        if self.path.startswith('/api/list-files'):
            self.handle_list_files()
        elif self.path.startswith('/api/load-file'):
            self.handle_load_file()
        else:
            # Default behavior: serve static files
            super().do_GET()
    
    def handle_list_files(self):
        """List all .jsonl files in the current directory"""
        try:
            current_dir = Path('.')
            print(current_dir.absolute())
            jsonl_files = []
            
            # Find all .jsonl files
            for file in current_dir.rglob('*.jsonl'):
                if not any(part.startswith('.') for part in file.parts):  # Ignore hidden directories
                    jsonl_files.append(str(file))
            
            # Sort, most recently modified first
            jsonl_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(jsonl_files).encode())
            
        except Exception as e:
            self.send_error(500, f"Error listing files: {str(e)}")
    
    def handle_load_file(self):
        """Load the specified JSONL file"""
        try:
            # Parse query parameters
            parsed_path = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed_path.query)
            
            if 'file' not in query_params:
                self.send_error(400, "Missing 'file' parameter")
                return
            
            filename = query_params['file'][0]
            filepath = Path(filename)
            
            # Security check: ensure the file is within the current directory
            try:
                filepath = filepath.resolve()
                current_dir = Path('.').resolve()
                filepath.relative_to(current_dir)
            except ValueError:
                self.send_error(403, "Access denied")
                return
            
            if not filepath.exists():
                self.send_error(404, "File not found")
                return
            
            # Read and return file content
            self.send_response(200)
            self.send_header('Content-type', 'text/plain; charset=utf-8')
            self.end_headers()
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                self.wfile.write(content.encode('utf-8'))
                
        except Exception as e:
            self.send_error(500, f"Error loading file: {str(e)}")
    
    def log_message(self, format, *args):
        # Custom log format with rich
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = format % args
        
        # Color code based on status
        if "200" in message:
            style = "green"
        elif "404" in message:
            style = "yellow"
        elif "500" in message or "403" in message:
            style = "red"
        else:
            style = "blue"
        
        console.print(f"[dim]{timestamp}[/dim] [{style}]{message}[/{style}]")

def run_server():
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        hostname = os.uname().nodename if hasattr(os, 'uname') else 'localhost'
        
        # Create a beautiful info table with rich
        table = Table(show_header=False, box=box.ROUNDED, border_style="blue")
        table.add_column("Key", style="cyan bold", width=20)
        table.add_column("Value", style="white")
        
        table.add_row("📂 Directory", os.getcwd())
        table.add_row("🔌 Port", str(PORT))
        table.add_row("🖥️  Hostname", hostname)
        table.add_row("", "")
        table.add_row("🌐 Local URL", f"http://localhost:{PORT}/cs336_alignment/extra/jsonl_viewer.html")
        table.add_row("", "")
        table.add_row("📡 API 1", "GET /api/list-files")
        table.add_row("📡 API 2", "GET /api/load-file?file=<path>")
        
        panel = Panel(
            table,
            title="[bold green]🚀 HTTP Server Started[/bold green]",
            subtitle="[dim]Press Ctrl+C to stop[/dim]",
            border_style="green"
        )
        
        console.print()
        console.print(panel)
        console.print()
        console.print("[bold yellow]📊 Server logs:[/bold yellow]")
        console.print()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            console.print("\n")
            console.print("[bold red]🛑 Server stopped[/bold red]")

if __name__ == "__main__":
    run_server()
