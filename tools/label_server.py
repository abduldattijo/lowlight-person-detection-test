#!/usr/bin/env python3
"""Local labeling page for the low-light detection test.

Run:  python3 tools/label_server.py     then open http://127.0.0.1:8800
Serves frames from ../data/<clip>/{nl_final,ll_final} and saves hand labels to
../labels/<clip>.json. Standard library only. Binds to localhost only.
"""
import json, os, re, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LABELS = os.path.join(ROOT, "labels")
HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_page.html")
os.makedirs(LABELS, exist_ok=True)
SAFE = re.compile(r"^[A-Za-z0-9_\-\.]+$")


def clips():
    out = []
    for c in sorted(os.listdir(DATA)):
        p = os.path.join(DATA, c, "nl_final")
        if os.path.isdir(p):
            out.append({"clip": c, "frames": sorted(f for f in os.listdir(p) if f.endswith(".jpg"))})
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if ctype != "image/jpeg" else "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = unquote(self.path.split("?")[0])
        if p == "/":
            return self.send(200, open(HTML, "rb").read(), "text/html; charset=utf-8")
        if p == "/api/clips":
            return self.send(200, json.dumps(clips()))
        m = re.match(r"^/api/labels/([^/]+)$", p)
        if m and SAFE.match(m.group(1)):
            f = os.path.join(LABELS, m.group(1) + ".json")
            return self.send(200, open(f).read() if os.path.exists(f) else "{}")
        m = re.match(r"^/frame/([^/]+)/(nl_final|ll_final)/([^/]+)$", p)
        if m and all(SAFE.match(x) for x in (m.group(1), m.group(3))):
            f = os.path.join(DATA, m.group(1), m.group(2), m.group(3))
            if os.path.isfile(f):
                return self.send(200, open(f, "rb").read(), "image/jpeg")
        self.send(404, "not found", "text/plain")

    def do_POST(self):
        m = re.match(r"^/api/labels/([^/]+)$", unquote(self.path))
        if not (m and SAFE.match(m.group(1))):
            return self.send(404, "not found", "text/plain")
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            data = json.loads(body)
            assert isinstance(data, dict)
        except Exception:
            return self.send(400, "bad json", "text/plain")
        f = os.path.join(LABELS, m.group(1) + ".json")
        if os.path.exists(f):  # keep one backup of the previous save
            os.replace(f, f + ".bak")
        tmp = f + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        os.replace(tmp, f)
        self.send(200, '{"ok":true}')


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8800
    print(f"Labeling page: http://127.0.0.1:{port}  (labels -> {LABELS})")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
