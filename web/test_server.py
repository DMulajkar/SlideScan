"""Self-test for web/server.py with a fake pipeline (no COLMAP, no video). Prints ok.

    python web/test_server.py
"""
import json
import sys
import tempfile
import textwrap
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server  # noqa: E402

TOKEN = "t" * 20
ORIGIN = "https://dhruvmulajkar.me"
FAKE = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    job = Path(sys.argv[1])
    up = next(job.glob("upload.*")).read_bytes()
    print("STEP Pretending to work", flush=True)
    time.sleep(1.5 if up == b"slow" else 0.3)
    if up == b"fail":
        print("boom", flush=True)
        sys.exit(3)
    (job / "out").mkdir()
    (job / "out" / "heightmap.png").write_bytes(up[::-1])
    print("STEP Done", flush=True)
""")


def call(base, path, method="GET", body=None, headers=None):
    req = urllib.request.Request(base + path, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def wait(base, jid):
    for _ in range(100):
        s = json.loads(call(base, f"/api/jobs/{jid}")[2])
        if s["state"] in ("done", "failed"):
            return s
        time.sleep(0.1)
    raise AssertionError("job never finished")


def main():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "fake.py").write_text(FAKE)
        jobs = d / "jobs"
        # a job left "running" by a crash must come back as failed
        (jobs / ("x" * 22) / "run" / "frames").mkdir(parents=True)
        (jobs / ("x" * 22) / "upload.mov").write_bytes(b"v")
        server.write_state(jobs / ("x" * 22), state="running")
        srv = server.make_server(jobs, TOKEN, [ORIGIN], [sys.executable, str(d / "fake.py")], port=0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            assert server.read_state(jobs / ("x" * 22))["state"] == "failed"
            assert sorted(p.name for p in (jobs / ("x" * 22)).rglob("*")) == ["run", "state.json"]  # video dropped
            assert call(base, "/api/health")[0] == 200

            # CORS only for the listed origin
            code, h, _ = call(base, "/api/jobs", "OPTIONS", headers={"Origin": ORIGIN})
            assert code == 204 and h.get("Access-Control-Allow-Origin") == ORIGIN, h
            assert "X-Token" in h.get("Access-Control-Allow-Headers", "")
            _, h, _ = call(base, "/api/health", headers={"Origin": "https://evil.example"})
            assert "Access-Control-Allow-Origin" not in h

            # the access code
            assert call(base, "/api/check", headers={"X-Token": "nope"})[0] == 401
            assert call(base, "/api/check", headers={"X-Token": TOKEN})[0] == 200
            assert call(base, "/api/jobs?name=a.mov", "POST", b"video")[0] == 401
            assert call(base, "/api/jobs?name=a.txt", "POST", b"video", {"X-Token": TOKEN})[0] == 400

            # a job that works: upload, progress, result file
            code, _, body = call(base, "/api/jobs?name=IMG_1.MOV", "POST", b"video-bytes", {"X-Token": TOKEN})
            assert code == 201, (code, body)
            jid = json.loads(body)["id"]
            s = wait(base, jid)
            assert s["state"] == "done" and s["step"] == "Done" and s["files"] == ["heightmap.png"], s
            assert [n for n, _ in s["steps"]] == ["Pretending to work", "Done"], s["steps"]
            assert s["started"] <= s["steps"][0][1] + 0.1 <= s["steps"][1][1] + 0.1 <= s["ended"] + 0.2, s
            assert s["name"] == "IMG_1.MOV" and s["size"] == 11
            assert call(base, f"/api/jobs/{jid}/out/heightmap.png")[2] == b"setyb-oediv"

            # byte ranges (Safari asks for them before it plays a video): 11 bytes "setyb-oediv"
            hm = f"/api/jobs/{jid}/out/heightmap.png"
            for rng, want in (("bytes=0-1", (206, b"se", "bytes 0-1/11")), ("bytes=9-", (206, b"iv", "bytes 9-10/11")),
                              ("bytes=-3", (206, b"div", "bytes 8-10/11")), ("bytes=4-99", (206, b"b-oediv", "bytes 4-10/11")),
                              ("bytes=11-", (416, b"", "bytes */11")), ("bytes=0-1,4-5", (200, b"setyb-oediv", None))):
                code, h, body = call(base, hm, headers={"Range": rng})
                assert (code, body, h.get("Content-Range")) == want, (rng, code, body, h.get("Content-Range"))
            assert server.TYPES[".mp4"] == "video/mp4"

            # ?download=1 asks the browser to save the file (the page's buttons); plain links still open it
            assert "Content-Disposition" not in call(base, hm)[1]
            code, h, body = call(base, hm + "?download=1")
            assert code == 200 and body == b"setyb-oediv" and h["Content-Disposition"] == 'attachment; filename="heightmap.png"', h
            code, h, _ = call(base, hm + "?download=1", headers={"Range": "bytes=0-1"})
            assert code == 206 and h["Content-Disposition"].startswith("attachment"), h

            # a job that fails shows its log
            jid2 = json.loads(call(base, "/api/jobs?name=b.mp4", "POST", b"fail", {"X-Token": TOKEN})[2])["id"]
            s = wait(base, jid2)
            assert s["state"] == "failed" and s["exit_code"] == 3 and "boom" in s["log"], s
            for j in (jid, jid2):  # the video is deleted whether the job worked or not
                assert not list((jobs / j).glob("upload.*")), j

            # nothing outside a job's out/ folder, no made-up ids
            for p in (f"/api/jobs/{jid}/out/..", f"/api/jobs/{jid}/out/.state.json", f"/api/jobs/{jid}/state.json",
                      "/api/jobs/short", "/api/jobs/" + "y" * 22, "/etc/passwd"):
                assert call(base, p)[0] == 404, p

            # what counts as use for idle auto-off: a right code or a real job, not health checks or scanners
            srv.last_use = 0
            for p in ("/api/health", "/api/jobs/" + "y" * 22, "/wp-login.php"):
                call(base, p)
            call(base, "/api/check", headers={"X-Token": "nope"})
            assert srv.last_use == 0
            call(base, f"/api/jobs/{jid}")
            assert srv.last_use > 0
        finally:
            srv.shutdown()
            srv.server_close()

        # idle auto-off: not while a job runs (1.5 s here, longer than idle), then once `idle` after the job ends
        offs = []
        srv = server.make_server(d / "jobs2", TOKEN, [ORIGIN], [sys.executable, str(d / "fake.py")], port=0,
                                 idle=0.6, stop=lambda: offs.append(time.time()))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            jid = json.loads(call(base, "/api/jobs?name=c.mov", "POST", b"slow", {"X-Token": TOKEN})[2])["id"]
            for _ in range(100):  # read the state from disk: polling over HTTP would count as use
                s = server.read_state(d / "jobs2" / jid)
                if s["state"] == "done" and offs:
                    break
                time.sleep(0.05)
            assert s["state"] == "done" and offs, (s, offs)
            assert offs[0] - s["ended"] >= 0.6, (offs[0] - s["ended"])
        finally:
            srv.shutdown()
            srv.server_close()
    print("ok")


if __name__ == "__main__":
    main()
