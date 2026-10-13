#!/usr/bin/env python
"""Local tagging tool for the RQ5 frame pool (pre-registration section 4.5).

    python scripts/tag_rq5.py                      # tag docs/rq5_frame_pool.csv
    python scripts/tag_rq5.py --make-demo-pool data/rq5/demo_pool.csv
    python scripts/tag_rq5.py --demo --pool data/rq5/demo_pool.csv --tags data/rq5/demo_tags.csv

Then open http://127.0.0.1:8765 in a browser. Standard library only; serves on
localhost only.

* Frames are shown strictly in the pool's ``tag_order``, one at a time, with no
  skipping. The page shows **only the image**: no model score, rank, prediction, phase
  label, video ID or frame index.
* Tags: smoke, blur, glare, blood, occlusion (any combination) or clean (alone).
  Keys 1-5 toggle the artefacts, 0 or c toggles clean, Enter saves, u undoes the last
  saved frame.
* Every frame is saved as it is tagged (appended to the tags CSV, flushed and fsynced),
  so a session can stop at any time and resume where it left off.
* The page shows the stopping rule: per-artefact counts against 50, the 300-frame
  minimum and the 1,000-frame cap. Once the rule is met the tool accepts no more tags.

The registered pool must hash to the SHA-256 recorded in LOG.md. ``--demo`` is for
trying the tool: it accepts only pools whose frames all come from training videos
1-32, so the demo can never show a pool or test frame.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from srb.datasets.overlap import CHOLEC80_TEST_IDS, CHOLEC80_TRAIN_IDS
from srb.rq5 import (ARTEFACTS, MAX_TAGGED, MIN_TAGGED, POOL_COLUMNS, POOL_SHA256,
                     STOP_PER_ARTEFACT, TAG_COLUMNS, TAGS, check_tags, next_frame,
                     stopping_status)

ROOT = Path(__file__).resolve().parents[1]
DEMO_SEED = 20261013


class Tagger:
    def __init__(self, pool_path: Path, tags_path: Path, frames_root: Path, demo: bool):
        raw = pool_path.read_bytes()
        self.pool = pd.read_csv(pool_path)
        if list(self.pool.columns) != POOL_COLUMNS:
            raise SystemExit(f"{pool_path}: unexpected columns")
        vids = set(int(v) for v in self.pool["video_id"])
        if demo:
            if not vids <= CHOLEC80_TRAIN_IDS:
                raise SystemExit("--demo accepts only frames from training videos 1-32")
        else:
            if hashlib.sha256(raw).hexdigest() != POOL_SHA256:
                raise SystemExit(f"{pool_path}: SHA-256 differs from the registered pool")
            if not vids <= CHOLEC80_TEST_IDS:
                raise SystemExit("the registered pool holds test videos 41-80 only")
        self.tags_path, self.frames_root = tags_path, frames_root
        if not tags_path.exists():
            tags_path.parent.mkdir(parents=True, exist_ok=True)
            tags_path.write_text(",".join(TAG_COLUMNS) + "\n")
        self.tagged()  # validates an existing file against the pool

    def tagged(self) -> pd.DataFrame:
        df = pd.read_csv(self.tags_path)
        if list(df.columns) != TAG_COLUMNS:
            raise SystemExit(f"{self.tags_path}: unexpected columns")
        ref = self.pool.set_index("tag_order").loc[df["tag_order"]]
        if not (np.array_equal(ref["video_id"], df["video_id"])
                and np.array_equal(ref["frame_idx_25fps"], df["frame_idx_25fps"])):
            raise SystemExit(f"{self.tags_path}: rows do not match the pool")
        return df

    def state(self) -> dict:
        df = self.tagged()
        status = stopping_status(df)
        return {"next": next_frame(self.pool, df), "pool_size": len(self.pool),
                "status": status, "artefacts": list(ARTEFACTS), "tags": list(TAGS),
                "per_artefact": STOP_PER_ARTEFACT, "min": MIN_TAGGED, "max": MAX_TAGGED}

    def image(self, order: int) -> bytes:
        if order != self.state()["next"]:
            raise PermissionError("only the next frame in tagging order is shown")
        rel = self.pool.loc[self.pool["tag_order"] == order, "path"].iloc[0]
        path = (self.frames_root / rel).resolve()
        if self.frames_root.resolve() not in path.parents:
            raise PermissionError("path outside the frames root")
        return path.read_bytes()

    def save(self, order: int, tags: list[str]) -> None:
        if order != self.state()["next"]:
            raise ValueError("not the next frame in tagging order")
        check_tags(set(tags))
        row = self.pool.loc[self.pool["tag_order"] == order].iloc[0]
        rec = [order, int(row["video_id"]), int(row["frame_idx_25fps"]),
               *[int(t in tags) for t in TAGS], dt.datetime.now().isoformat(timespec="seconds")]
        with open(self.tags_path, "a", newline="") as f:
            csv.writer(f, lineterminator="\n").writerow(rec)
            f.flush()
            os.fsync(f.fileno())

    def undo(self) -> None:
        lines = self.tags_path.read_text().splitlines(keepends=True)
        if len(lines) <= 1:
            return
        tmp = self.tags_path.with_suffix(".tmp")
        tmp.write_text("".join(lines[:-1]))
        os.replace(tmp, self.tags_path)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>RQ5 tagging</title>
<style>
 body{margin:0;font:15px system-ui,sans-serif;background:#111;color:#eee;display:flex;gap:20px;padding:16px}
 #img{max-width:calc(100vw - 340px);max-height:calc(100vh - 32px);object-fit:contain;background:#000}
 aside{width:290px;flex:none} label{display:block;padding:6px 8px;margin:4px 0;border:1px solid #444;border-radius:6px;cursor:pointer}
 label.on{background:#2d5b8a;border-color:#5b9bd5} kbd{color:#aaa;font-size:12px;margin-right:6px}
 button{font:inherit;padding:8px 14px;margin:8px 6px 0 0;border-radius:6px;border:1px solid #666;background:#222;color:#eee;cursor:pointer}
 table{border-collapse:collapse;margin-top:14px;width:100%} td{padding:2px 6px;border-bottom:1px solid #333} td.n{text-align:right}
 .ok{color:#7c7} .msg{color:#e88;min-height:1.2em;margin-top:8px} .done{font-size:18px;color:#7c7}
</style></head><body>
<img id="img" alt="frame">
<aside>
 <div id="pos"></div>
 <div id="boxes"></div>
 <button id="save">Save <kbd>Enter</kbd></button><button id="undo">Undo last <kbd>u</kbd></button>
 <div class="msg" id="msg"></div>
 <table id="prog"></table>
</aside>
<script>
let S=null, sel=new Set();
const keys={smoke:"1",blur:"2",glare:"3",blood:"4",occlusion:"5",clean:"0"};
async function load(){
  S=await (await fetch("/state")).json(); sel.clear(); render();
}
function render(){
  const st=S.status, b=document.getElementById("boxes");
  b.innerHTML=S.tags.map(t=>`<label data-t="${t}" class="${sel.has(t)?"on":""}"><kbd>${keys[t]}</kbd>${t}</label>`).join("");
  b.querySelectorAll("label").forEach(l=>l.onclick=()=>toggle(l.dataset.t));
  const img=document.getElementById("img");
  if(S.next===null){img.removeAttribute("src");
    document.getElementById("pos").innerHTML=`<div class="done">Done: ${st.reason||"pool exhausted"}.</div>`;}
  else{img.src="/img/"+S.next; document.getElementById("pos").textContent=`Frame ${S.next+1} of the pool (${st.n_tagged} tagged)`;}
  let rows=S.artefacts.map(a=>`<tr><td>${a}</td><td class="n ${st.counts[a]>=S.per_artefact?"ok":""}">${st.counts[a]} / ${S.per_artefact}</td></tr>`).join("");
  rows+=`<tr><td>clean (does not count)</td><td class="n">${st.counts.clean}</td></tr>`;
  rows+=`<tr><td>tagged (min ${S.min})</td><td class="n ${st.n_tagged>=S.min?"ok":""}">${st.n_tagged}</td></tr>`;
  rows+=`<tr><td>cap</td><td class="n">${S.max}</td></tr>`;
  document.getElementById("prog").innerHTML=rows;
}
function toggle(t){
  if(t==="clean"){ if(sel.has("clean")) sel.delete("clean"); else {sel.clear(); sel.add("clean");} }
  else { sel.delete("clean"); sel.has(t)?sel.delete(t):sel.add(t); }
  render();
}
async function post(url,body){
  const r=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body||{})});
  const j=await r.json(); document.getElementById("msg").textContent=j.error||""; if(!j.error) await load();
}
document.getElementById("save").onclick=()=>post("/tag",{tag_order:S.next,tags:[...sel]});
document.getElementById("undo").onclick=()=>post("/undo");
document.addEventListener("keydown",e=>{
  const t=Object.keys(keys).find(k=>keys[k]===e.key||(k==="clean"&&e.key==="c"));
  if(t) toggle(t); else if(e.key==="Enter") document.getElementById("save").click();
  else if(e.key==="u") document.getElementById("undo").click();
});
load();
</script></body></html>"""


def make_handler(tagger: Tagger):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/state":
                return self._json(tagger.state())
            if self.path.startswith("/img/"):
                try:
                    return self._send(200, tagger.image(int(self.path[5:])), "image/jpeg")
                except (ValueError, PermissionError, IndexError) as e:
                    return self._json({"error": str(e)}, 403)
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                if self.path == "/tag":
                    tagger.save(int(body["tag_order"]), list(body["tags"]))
                elif self.path == "/undo":
                    tagger.undo()
                else:
                    return self._json({"error": "not found"}, 404)
            except (ValueError, KeyError, TypeError) as e:
                return self._json({"error": str(e)}, 400)
            self._json({"ok": True})

        def log_message(self, *a):  # keep the terminal quiet
            pass

    return Handler


def make_demo_pool(out: Path, n: int) -> None:
    """n frames drawn from training videos 1-32 only, in the pool's format."""
    man = pd.read_parquet(ROOT / "data" / "cholec80" / "manifest.parquet",
                          columns=["video_id", "frame_idx_25fps", "path"])
    man = man[man["video_id"].isin(sorted(CHOLEC80_TRAIN_IDS))].reset_index(drop=True)
    pick = np.random.default_rng(DEMO_SEED).choice(len(man), n, replace=False)
    demo = man.iloc[pick].reset_index(drop=True)
    demo.insert(0, "draw_idx", range(n))
    demo.insert(0, "tag_order", range(n))
    out.parent.mkdir(parents=True, exist_ok=True)
    demo[POOL_COLUMNS].to_csv(out, index=False, lineterminator="\n")
    print(f"wrote {n} training-video frames to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", type=Path, default=ROOT / "docs" / "rq5_frame_pool.csv")
    ap.add_argument("--tags", type=Path, default=ROOT / "data" / "rq5" / "tags.csv")
    ap.add_argument("--frames-root", type=Path, default=ROOT / "data" / "cholec80")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--demo", action="store_true", help="training-video frames only")
    ap.add_argument("--make-demo-pool", type=Path, metavar="OUT")
    ap.add_argument("-n", type=int, default=5, help="frames in the demo pool")
    args = ap.parse_args()
    if args.make_demo_pool:
        return make_demo_pool(args.make_demo_pool, args.n)
    if not args.demo and args.tags.resolve() == (ROOT / "data/rq5/tags.csv").resolve() \
            and args.pool.resolve() != (ROOT / "docs/rq5_frame_pool.csv").resolve():
        sys.exit("a non-registered pool needs its own --tags file")
    tagger = Tagger(args.pool, args.tags, args.frames_root, args.demo)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(tagger))
    s = tagger.state()["status"]
    print(f"{'DEMO, training frames' if args.demo else 'RQ5 pool'}: {s['n_tagged']} tagged; "
          f"tags -> {args.tags}\nopen http://127.0.0.1:{args.port}  (Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
