"""SlideScan web jobs: take a walkaround video, run the pipeline, serve the results.

Runs on the Azure VM behind Caddy (HTTPS). The page is dhruvmulajkar.me/slidescan/ (repo final-portfolio).

    SLIDESCAN_TOKEN=... python web/server.py [--port 8080] [--jobs ~/slidescan-jobs] [--origins A,B]

API (JSON except result files):
    GET  /api/health                     {"ok": true, "queued": n}
    GET  /api/check                      header X-Token -> 200, or 401 (the page checks the code before a long upload)
    POST /api/jobs?name=IMG_1.MOV        header X-Token; body = the raw video bytes -> 201 {"id": ...}
    GET  /api/jobs/<id>                  state (queued|running|done|failed), step, steps ([name, start time] each),
                                         times, result file names
    GET  /api/jobs/<id>/out/<file>       one result file (heightmap.png, reconstruction.mp4, ...); honours a
                                         single Range header, which Safari needs to play the video;
                                         ?download=1 makes the browser save it instead of opening it

Job ids are 128-bit random, so a results link works without the code; only uploads need it.
Each job runs `bash web/run_job.sh <job dir>`, which prints "STEP ..." lines; the latest is the progress shown.

Idle auto-off (VM only): with SLIDESCAN_IDLE_MIN=20 (or --idle-min 20) the server deallocates its own VM after
20 minutes with no job and no real use: a correct access code or a request for an existing job. Health checks
and internet scanners don't count. It uses the VM's managed identity; the page's Start button turns it back on.
"""
import argparse
import hmac
import json
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

HERE = Path(__file__).resolve().parent
VIDEO = {".mov", ".mp4", ".m4v"}
MAX_BYTES = 2 * 1024**3  # ponytail: one cap; a 60 s 4K iPhone clip is about 200 MB
ID = re.compile(r"^[A-Za-z0-9_-]{22}$")
FILE = re.compile(r"^\w[\w.-]*$")  # no slashes, no leading dot
TYPES = {".png": "image/png", ".json": "application/json", ".txt": "text/plain; charset=utf-8", ".mp4": "video/mp4"}


def read_state(job):
    return json.loads((job / "state.json").read_text())


def write_state(job, **kw):
    p = job / "state.json"
    s = read_state(job) if p.exists() else {}
    s.update(kw)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(s))
    os.replace(tmp, p)  # atomic, so a status read never sees half a file


def tail(path, n=12):
    try:
        return path.read_text(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def drop_video(job):
    """The page promises the video and frames are deleted when a job ends, failed or not; results stay."""
    for p in [*job.glob("upload.*"), job / "run" / "raw", job / "run" / "frames"]:
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)


def worker(srv, pipeline):
    """One job at a time; ponytail: in-process queue, so jobs waiting at a restart are marked failed."""
    while True:
        job = srv.q.get()
        srv.busy = True
        write_state(job, state="running", step="Starting", started=time.time())
        rc, steps = -1, []
        try:
            with open(job / "job.log", "w") as log:
                p = subprocess.Popen([*pipeline, str(job)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, errors="replace")
                for line in p.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith("STEP "):
                        steps.append([line[5:].strip(), round(time.time(), 1)])  # the page times each step
                        write_state(job, step=steps[-1][0], steps=steps)
                rc = p.wait()
        except Exception as e:  # keep the worker alive for the next job
            with open(job / "job.log", "a") as log:
                log.write(f"server: {e!r}\n")
        drop_video(job)
        write_state(job, state="done" if rc == 0 else "failed", ended=time.time(), exit_code=rc)
        srv.busy, srv.last_use = False, time.time()


def deallocate_self():
    """Deallocate this Azure VM with its managed identity (needs the deallocate action on itself)."""
    def imds(path):
        req = urllib.request.Request("http://169.254.169.254/metadata/" + path, headers={"Metadata": "true"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)
    vm = imds("instance/compute?api-version=2021-02-01")["resourceId"]
    tok = imds("identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/")["access_token"]
    req = urllib.request.Request(f"https://management.azure.com{vm}/deallocate?api-version=2024-03-01", data=b"",
                                 method="POST", headers={"Authorization": "Bearer " + tok})
    urllib.request.urlopen(req, timeout=30).close()  # 202; the VM is off a minute or two later


def idle_off(srv, idle, stop):
    """Call stop() once nothing has run and nobody has used the server for `idle` seconds."""
    while True:
        time.sleep(min(30, idle / 4))
        if not srv.busy and srv.q.empty() and time.time() - srv.last_use > idle:
            print(f"idle {idle / 60:.0f} min, switching off", flush=True)
            try:
                stop()
            except Exception as e:
                print(f"switch-off failed: {e!r}", flush=True)
            srv.last_use = time.time()  # if the VM is still up after another idle stretch, try again


class Handler(BaseHTTPRequestHandler):
    server_version = "SlideScan"

    def cors(self):
        o = self.headers.get("Origin")
        if o in self.server.origins:
            self.send_header("Access-Control-Allow-Origin", o)
            self.send_header("Vary", "Origin")

    def reply(self, code, body=b"", ctype="application/json", headers=None):
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.cors()
        self.send_header("Content-Type", ctype)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, f, save=False):
        n, ctype = f.stat().st_size, TYPES.get(f.suffix, "application/octet-stream")
        h = {"Accept-Ranges": "bytes"}
        if save:  # browsers ignore <a download> across origins; this header makes them save instead of open
            h["Content-Disposition"] = f'attachment; filename="{f.name}"'
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
        if not m or m.groups() == ("", ""):  # no range, or several: send it all
            return self.reply(200, f.read_bytes(), ctype, h)
        a, b = m.groups()
        start, end = (int(a), min(int(b), n - 1) if b else n - 1) if a else (max(0, n - int(b)), n - 1)
        if start > end:
            return self.reply(416, b"", ctype, {"Content-Range": f"bytes */{n}"})
        with open(f, "rb") as fh:
            fh.seek(start)
            body = fh.read(end - start + 1)
        self.reply(206, body, ctype, {**h, "Content-Range": f"bytes {start}-{end}/{n}"})

    def token_ok(self):
        ok = hmac.compare_digest(self.headers.get("X-Token", "").encode(), self.server.token.encode())
        if ok:
            self.server.last_use = time.time()
        return ok

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST")
        self.send_header("Access-Control-Allow-Headers", "X-Token, Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self):
        parts = urlsplit(self.path).path.strip("/").split("/")
        if parts == ["api", "health"]:
            return self.reply(200, {"ok": True, "queued": self.server.q.qsize()})
        if parts == ["api", "check"]:
            return self.reply(200, {"ok": True}) if self.token_ok() else self.reply(401, {"error": "wrong access code"})
        if len(parts) in (3, 5) and parts[:2] == ["api", "jobs"] and ID.match(parts[2]):
            job = self.server.jobs / parts[2]
            if (job / "state.json").exists():
                self.server.last_use = time.time()  # someone is following or reading a real job
                if len(parts) == 3:
                    s = read_state(job)
                    out = job / "out"
                    s["files"] = sorted(os.listdir(out)) if out.is_dir() else []
                    if s["state"] == "failed":
                        s["log"] = tail(job / "job.log")
                    return self.reply(200, s)
                f = job / "out" / parts[4]
                if parts[3] == "out" and FILE.match(parts[4]) and f.is_file():
                    return self.send_file(f, "download" in parse_qs(urlsplit(self.path).query))
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        u = urlsplit(self.path)
        if u.path.rstrip("/") != "/api/jobs":
            return self.reply(404, {"error": "not found"})
        if not self.token_ok():
            return self.reply(401, {"error": "wrong access code"})
        name = parse_qs(u.query).get("name", ["video.mov"])[0][:120]
        ext = Path(name).suffix.lower()
        if ext not in VIDEO:
            return self.reply(400, {"error": "send a video: " + ", ".join(sorted(VIDEO))})
        try:
            n = int(self.headers["Content-Length"])
        except (TypeError, ValueError):
            return self.reply(411, {"error": "Content-Length needed"})
        if not 0 < n <= MAX_BYTES:
            return self.reply(413, {"error": "empty, or over the 2 GB cap"})
        job = self.server.jobs / secrets.token_urlsafe(16)
        job.mkdir()
        left = n
        with open(job / f"upload{ext}", "wb") as f:
            while left:
                b = self.rfile.read(min(1 << 20, left))
                if not b:
                    break
                f.write(b)
                left -= len(b)
        if left:  # the browser went away mid-upload
            shutil.rmtree(job, ignore_errors=True)
            return
        write_state(job, state="queued", step="Waiting for the processor", name=name, size=n, created=time.time())
        self.server.q.put(job)
        self.reply(201, {"id": job.name})


def make_server(jobs, token, origins, pipeline, port=8080, host="127.0.0.1", idle=0, stop=deallocate_self):
    jobs = Path(jobs)
    jobs.mkdir(parents=True, exist_ok=True)
    for st in jobs.glob("*/state.json"):
        if read_state(st.parent).get("state") in ("queued", "running"):
            drop_video(st.parent)
            write_state(st.parent, state="failed", step="The server restarted; upload again", ended=time.time())
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.jobs, srv.token, srv.origins, srv.q = jobs, token, set(origins), queue.Queue()
    srv.busy, srv.last_use = False, time.time()
    threading.Thread(target=worker, args=(srv, pipeline), daemon=True).start()
    if idle:
        threading.Thread(target=idle_off, args=(srv, idle, stop), daemon=True).start()
    return srv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--jobs", default=str(Path.home() / "slidescan-jobs"))
    ap.add_argument("--origins", default="https://dhruvmulajkar.me,https://www.dhruvmulajkar.me")
    ap.add_argument("--idle-min", type=float, default=float(os.environ.get("SLIDESCAN_IDLE_MIN", 0)),
                    help="deallocate this Azure VM after this many idle minutes (0 = never)")
    a = ap.parse_args(argv)
    token = os.environ.get("SLIDESCAN_TOKEN", "")
    if len(token) < 16:
        sys.exit("set SLIDESCAN_TOKEN (16+ characters)")
    srv = make_server(a.jobs, token, a.origins.split(","), ["bash", str(HERE / "run_job.sh")], a.port,
                      idle=a.idle_min * 60)
    print(f"listening on 127.0.0.1:{a.port}; jobs in {a.jobs}; origins {a.origins}; idle-off {a.idle_min:g} min",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
