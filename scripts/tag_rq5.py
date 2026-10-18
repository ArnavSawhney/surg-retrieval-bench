#!/usr/bin/env python
"""Local tagging tool for the RQ5 frame pool (pre-registration section 4.5).

    python scripts/tag_rq5.py                      # tag docs/rq5_frame_pool.csv
    python scripts/tag_rq5.py --make-demo-pool data/rq5/demo_pool.csv
    python scripts/tag_rq5.py --demo --pool data/rq5/demo_pool.csv --tags data/rq5/demo_tags.csv
    python scripts/tag_rq5.py --review-out-of-body # frames 0-435: out of body, y / n

Then open http://127.0.0.1:8765 in a browser. Standard library only; serves on
localhost only.

* Frames are shown strictly in the pool's ``tag_order``, one at a time, with no
  skipping. The page shows **only the image**: no model score, rank, prediction, phase
  label, video ID or frame index.
* Tags: smoke, blur, glare, blood, occlusion (any combination) or clean (alone).
  Keys 1-5 toggle the artefacts, 0 or c toggles clean, 6 or o toggles out of body,
  Enter saves, u undoes the last saved frame.
* Out of body (tagging guide, 2026-10-18) is not an artefact: the tags row gets clean
  (or blur alone, if blurred) and a row (tag_order, 1, tagging, timestamp) goes to a
  separate out-of-body file (default data/rq5/out_of_body.csv); the tags file keeps
  its schema. Smoke, glare, blood and occlusion cannot be combined with it.
* --review-out-of-body: frames 0-435 were tagged before the out-of-body key existed.
  The review shows each of them in tag order, image only (no tags, labels or scores),
  and asks only "out of body? y / n"; each answer is appended to the out-of-body file
  (source review), so it resumes; u undoes the last answer. Tagging of the registered
  pool resumes only once the review is complete.
* Every frame is saved as it is tagged (appended to the tags CSV, flushed and fsynced),
  so a session can stop at any time and resume where it left off.
* The page shows the stopping rule, counted on the corrected tags (raw tags with the
  out-of-body corrections, see scripts/rq5_apply_corrections.py): per-artefact counts
  against 50, the 300-frame minimum and the 1,000-frame cap. Once the rule is met the
  tool accepts no more tags.
* The registered tags file's first 436 rows must hash to the SHA-256 in LOG.md, and
  undo never reaches them.

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
from srb.rq5 import (ARTEFACTS, MAX_TAGGED, MIN_TAGGED, OUT_OF_BODY_COLUMNS, POOL_COLUMNS,
                     POOL_SHA256, RAW_TAGS_SHA256, REVIEWED_FRAMES, STOP_PER_ARTEFACT,
                     TAG_COLUMNS, TAGS, apply_corrections, next_frame, raw_prefix_sha256,
                     reviewed_prefix, stopping_status, tag_values)

ROOT = Path(__file__).resolve().parents[1]
DEMO_SEED = 20261013


def default_out_of_body(tags_path: Path) -> Path:
    name = "out_of_body.csv" if tags_path.name == "tags.csv" else f"{tags_path.stem}_out_of_body.csv"
    return tags_path.with_name(name)


def _append(path: Path, rec: list) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow(rec)
        f.flush()
        os.fsync(f.fileno())


def _drop_last_line(path: Path) -> None:
    lines = path.read_text().splitlines(keepends=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(lines[:-1]))
    os.replace(tmp, path)


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


class Tagger:
    def __init__(self, pool_path: Path, tags_path: Path, frames_root: Path, demo: bool,
                 oob_path: Path | None = None, review_n: int | None = None):
        """review_n: frames 0..review_n-1 need a review answer (registered pool: 436;
        demo: 0 unless given)."""
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
        if review_n is not None and not demo and review_n != REVIEWED_FRAMES:
            raise SystemExit(f"the registered review covers frames 0-{REVIEWED_FRAMES - 1}")
        self.demo = demo
        self.review_n = (review_n or 0) if demo else REVIEWED_FRAMES
        self.tags_path, self.frames_root = tags_path, frames_root
        self.oob_path = oob_path or default_out_of_body(tags_path)
        if not tags_path.exists():
            tags_path.parent.mkdir(parents=True, exist_ok=True)
            tags_path.write_text(",".join(TAG_COLUMNS) + "\n")
        if not self.oob_path.exists():
            self.oob_path.write_text(",".join(OUT_OF_BODY_COLUMNS) + "\n")
        if not demo:
            try:
                same = raw_prefix_sha256(tags_path.read_bytes()) == RAW_TAGS_SHA256
            except ValueError:
                same = False
            if not same:
                raise SystemExit(f"{tags_path}: the first {REVIEWED_FRAMES} rows differ "
                                 f"from the frozen raw tags (SHA-256 in LOG.md)")
        self._reconcile()
        self.corrected()  # validates both files against the pool and each other

    def _reconcile(self) -> None:
        """A save writes the out-of-body row first, then the tags row (and undo removes
        them in the reverse order), so a crash can only leave a trailing out-of-body row
        for an untagged frame: drop it."""
        n = len(self.tagged())
        while True:
            oob = pd.read_csv(self.oob_path)
            if not len(oob):
                return
            last = oob.iloc[-1]
            if last["source"] != "tagging" or int(last["tag_order"]) < n:
                return
            _drop_last_line(self.oob_path)

    def out_of_body(self) -> pd.DataFrame:
        oob = pd.read_csv(self.oob_path)
        if list(oob.columns) != OUT_OF_BODY_COLUMNS:
            raise SystemExit(f"{self.oob_path}: unexpected columns")
        return oob

    def corrected(self) -> pd.DataFrame:
        try:
            return apply_corrections(self.tagged(), self.out_of_body())
        except ValueError as e:
            raise SystemExit(f"{self.oob_path}: {e}") from None

    def tagged(self) -> pd.DataFrame:
        df = pd.read_csv(self.tags_path)
        if list(df.columns) != TAG_COLUMNS:
            raise SystemExit(f"{self.tags_path}: unexpected columns")
        ref = self.pool.set_index("tag_order").loc[df["tag_order"]]
        if not (np.array_equal(ref["video_id"], df["video_id"])
                and np.array_equal(ref["frame_idx_25fps"], df["frame_idx_25fps"])):
            raise SystemExit(f"{self.tags_path}: rows do not match the pool")
        return df

    # ---- tagging --------------------------------------------------------------

    def review_done(self) -> bool:
        return self.review_state()["next"] is None

    def state(self) -> dict:
        df = self.corrected()
        status = stopping_status(df)
        nxt = next_frame(self.pool, df) if self.review_done() else None
        return {"next": nxt, "pool_size": len(self.pool), "status": status,
                "n_out_of_body": int(df["out_of_body"].sum()),
                "review_pending": not self.review_done(), "artefacts": list(ARTEFACTS),
                "tags": list(TAGS), "per_artefact": STOP_PER_ARTEFACT, "min": MIN_TAGGED,
                "max": MAX_TAGGED}

    def image(self, order: int) -> bytes:
        if order != self.state()["next"]:
            raise PermissionError("only the next frame in tagging order is shown")
        return self._read_image(order)

    def _read_image(self, order: int) -> bytes:
        rel = self.pool.loc[self.pool["tag_order"] == order, "path"].iloc[0]
        path = (self.frames_root / rel).resolve()
        if self.frames_root.resolve() not in path.parents:
            raise PermissionError("path outside the frames root")
        return path.read_bytes()

    def save(self, order: int, tags: list[str], out_of_body: bool = False) -> None:
        if order != self.state()["next"]:
            raise ValueError("not the next frame in tagging order")
        values = tag_values(set(tags), out_of_body)
        row = self.pool.loc[self.pool["tag_order"] == order].iloc[0]
        now = _now()
        if out_of_body:
            _append(self.oob_path, [order, 1, "tagging", now])
        _append(self.tags_path, [order, int(row["video_id"]), int(row["frame_idx_25fps"]),
                                 *[values[t] for t in TAGS], now])

    def undo(self) -> None:
        n = len(self.tagged())
        if n == 0:
            raise ValueError("nothing to undo")
        if n <= self.review_n:
            raise ValueError(f"frames 0-{self.review_n - 1} are frozen; undo cannot change them")
        _drop_last_line(self.tags_path)
        self._reconcile()

    # ---- out-of-body review of frames 0..review_n-1 ---------------------------

    def review_state(self) -> dict:
        k = reviewed_prefix(self.out_of_body())
        if len(self.tagged()) < self.review_n:
            raise SystemExit(f"frames 0-{self.review_n - 1} are not all tagged")
        return {"next": k if k < self.review_n else None, "n_reviewed": k,
                "n_total": self.review_n}

    def review_image(self, order: int) -> bytes:
        if order != self.review_state()["next"]:
            raise PermissionError("only the next frame in review order is shown")
        return self._read_image(order)

    def review_answer(self, order: int, out_of_body: bool) -> None:
        if order != self.review_state()["next"]:
            raise ValueError("not the next frame in review order")
        _append(self.oob_path, [order, int(bool(out_of_body)), "review", _now()])

    def review_undo(self) -> None:
        oob = self.out_of_body()
        if not len(oob) or oob.iloc[-1]["source"] != "review":
            raise ValueError("no review answer to undo")
        _drop_last_line(self.oob_path)


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
const keys={smoke:"1",blur:"2",glare:"3",blood:"4",occlusion:"5",clean:"0",out_of_body:"6"};
const alt={clean:"c",out_of_body:"o"};
async function load(){
  S=await (await fetch("/state")).json(); sel.clear(); render();
}
function render(){
  const st=S.status, b=document.getElementById("boxes");
  b.innerHTML=[...S.tags,"out_of_body"].map(t=>`<label data-t="${t}" class="${sel.has(t)?"on":""}"><kbd>${keys[t]}</kbd>${t.replace(/_/g," ")}</label>`).join("");
  b.querySelectorAll("label").forEach(l=>l.onclick=()=>toggle(l.dataset.t));
  const img=document.getElementById("img");
  if(S.review_pending){img.removeAttribute("src");
    document.getElementById("pos").innerHTML=`<div class="msg">Finish the out-of-body review first (--review-out-of-body).</div>`;}
  else if(S.next===null){img.removeAttribute("src");
    document.getElementById("pos").innerHTML=`<div class="done">Done: ${st.reason||"pool exhausted"}.</div>`;}
  else{img.src="/img/"+S.next; document.getElementById("pos").textContent=`Frame ${S.next+1} of the pool (${st.n_tagged} tagged)`;}
  let rows=S.artefacts.map(a=>`<tr><td>${a}</td><td class="n ${st.counts[a]>=S.per_artefact?"ok":""}">${st.counts[a]} / ${S.per_artefact}</td></tr>`).join("");
  rows+=`<tr><td>clean (does not count)</td><td class="n">${st.counts.clean}</td></tr>`;
  rows+=`<tr><td>out of body (not an artefact)</td><td class="n">${S.n_out_of_body}</td></tr>`;
  rows+=`<tr><td>tagged (min ${S.min})</td><td class="n ${st.n_tagged>=S.min?"ok":""}">${st.n_tagged}</td></tr>`;
  rows+=`<tr><td>cap</td><td class="n">${S.max}</td></tr>`;
  document.getElementById("prog").innerHTML=rows;
}
function toggle(t){
  const only_blur=["blur","out_of_body"];
  if(t==="clean"){ if(sel.has("clean")) sel.delete("clean"); else {sel.clear(); sel.add("clean");} }
  else if(t==="out_of_body"){ if(sel.has(t)) sel.delete(t); else {[...sel].forEach(x=>{if(!only_blur.includes(x)) sel.delete(x);}); sel.add(t);} }
  else { sel.delete("clean"); if(!only_blur.includes(t)) sel.delete("out_of_body"); sel.has(t)?sel.delete(t):sel.add(t); }
  render();
}
async function post(url,body){
  const r=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body||{})});
  const j=await r.json(); document.getElementById("msg").textContent=j.error||""; if(!j.error) await load();
}
document.getElementById("save").onclick=()=>post("/tag",{tag_order:S.next,
  tags:[...sel].filter(t=>t!=="out_of_body"),out_of_body:sel.has("out_of_body")});
document.getElementById("undo").onclick=()=>post("/undo");
document.addEventListener("keydown",e=>{
  const t=Object.keys(keys).find(k=>keys[k]===e.key||alt[k]===e.key);
  if(t) toggle(t); else if(e.key==="Enter") document.getElementById("save").click();
  else if(e.key==="u") document.getElementById("undo").click();
});
load();
</script></body></html>"""


REVIEW_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>RQ5 out-of-body review</title>
<style>
 body{margin:0;font:15px system-ui,sans-serif;background:#111;color:#eee;display:flex;gap:20px;padding:16px}
 #img{max-width:calc(100vw - 340px);max-height:calc(100vh - 32px);object-fit:contain;background:#000}
 aside{width:290px;flex:none} h2{font-size:18px;font-weight:600} kbd{color:#aaa;font-size:12px;margin-right:6px}
 button{font:inherit;padding:8px 14px;margin:8px 6px 0 0;border-radius:6px;border:1px solid #666;background:#222;color:#eee;cursor:pointer}
 .msg{color:#e88;min-height:1.2em;margin-top:8px} .done{font-size:18px;color:#7c7}
</style></head><body>
<img id="img" alt="frame">
<aside>
 <div id="pos"></div>
 <h2>Out of body?</h2>
 <button id="yes"><kbd>y</kbd>yes</button><button id="no"><kbd>n</kbd>no</button><br>
 <button id="undo"><kbd>u</kbd>Undo last answer</button>
 <div class="msg" id="msg"></div>
</aside>
<script>
let S=null;
async function load(){
  S=await (await fetch("/state")).json();
  const img=document.getElementById("img");
  if(S.next===null){img.removeAttribute("src");
    document.getElementById("pos").innerHTML=`<div class="done">Review done: ${S.n_reviewed} of ${S.n_total}.</div>`;}
  else{img.src="/img/"+S.next; document.getElementById("pos").textContent=`Frame ${S.n_reviewed+1} of ${S.n_total}`;}
}
async function post(url,body){
  const r=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body||{})});
  const j=await r.json(); document.getElementById("msg").textContent=j.error||""; if(!j.error) await load();
}
const answer=v=>{ if(S&&S.next!==null) post("/answer",{tag_order:S.next,out_of_body:v}); };
document.getElementById("yes").onclick=()=>answer(true);
document.getElementById("no").onclick=()=>answer(false);
document.getElementById("undo").onclick=()=>post("/undo");
document.addEventListener("keydown",e=>{
  if(e.key==="y") answer(true); else if(e.key==="n") answer(false);
  else if(e.key==="u") document.getElementById("undo").click();
});
load();
</script></body></html>"""


def make_handler(tagger: Tagger, review: bool = False):
    page = REVIEW_PAGE if review else PAGE
    state = tagger.review_state if review else tagger.state
    image = tagger.review_image if review else tagger.image

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
                return self._send(200, page.encode(), "text/html; charset=utf-8")
            if self.path == "/state":
                return self._json(state())
            if self.path.startswith("/img/"):
                try:
                    return self._send(200, image(int(self.path[5:])), "image/jpeg")
                except (ValueError, PermissionError, IndexError) as e:
                    return self._json({"error": str(e)}, 403)
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                if review and self.path == "/answer":
                    tagger.review_answer(int(body["tag_order"]), bool(body["out_of_body"]))
                elif review and self.path == "/undo":
                    tagger.review_undo()
                elif not review and self.path == "/tag":
                    tagger.save(int(body["tag_order"]), list(body["tags"]),
                                bool(body.get("out_of_body", False)))
                elif not review and self.path == "/undo":
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
    ap.add_argument("--out-of-body", type=Path, default=None,
                    help="default: out_of_body.csv next to tags.csv (<stem>_out_of_body.csv)")
    ap.add_argument("--review-out-of-body", action="store_true",
                    help="image-only review of frames 0-435: out of body, y / n")
    ap.add_argument("--review-n", type=int, default=None,
                    help="--demo only: frames 0..n-1 to review")
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
    oob = args.out_of_body or default_out_of_body(args.tags)
    if not args.demo and oob.resolve() != (ROOT / "data/rq5/out_of_body.csv").resolve():
        sys.exit("the registered pool's out-of-body file is data/rq5/out_of_body.csv")
    if args.demo and oob.resolve() == (ROOT / "data/rq5/out_of_body.csv").resolve():
        sys.exit("the demo needs its own --out-of-body file")
    if args.review_n is not None and not args.demo:
        sys.exit("--review-n is for --demo only")
    tagger = Tagger(args.pool, args.tags, args.frames_root, args.demo, oob, args.review_n)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port),
                              make_handler(tagger, review=args.review_out_of_body))
    who = "DEMO, training frames" if args.demo else "RQ5 pool"
    if args.review_out_of_body:
        r = tagger.review_state()
        print(f"{who}, OUT-OF-BODY REVIEW: {r['n_reviewed']} of {r['n_total']} answered; "
              f"answers -> {oob}", flush=True)
    else:
        if not tagger.review_done():
            sys.exit("finish the out-of-body review first (--review-out-of-body)")
        s = tagger.state()["status"]
        print(f"{who}: {s['n_tagged']} tagged; tags -> {args.tags}, out of body -> {oob}",
              flush=True)
    print(f"open http://127.0.0.1:{args.port}  (Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
