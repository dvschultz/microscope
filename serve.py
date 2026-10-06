"""Serve the static viewer with deep links: any path without a file extension
falls back to index.html, so /inceptionv1/mixed4a/123 loads the app.

    python serve.py [--port 8000]
"""
import argparse
import http.server
from functools import partial
from pathlib import Path

SITE = Path(__file__).resolve().parent / "site"


class Handler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if not Path(path).suffix:
            self.path = "/index.html"
        return super().send_head()

    def end_headers(self):
        if self.path.endswith((".json", ".html", ".js", ".css")):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, *args):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(SITE)))
    print(f"http://127.0.0.1:{args.port}/inceptionv1")
    server.serve_forever()


if __name__ == "__main__":
    main()
