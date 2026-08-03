"""Build a click-through review page for the zone-label audit.

audit_zone_labels.py finds suspects; this renders them as croppped thumbnails with the
box in context, a proposed fix, and buttons to accept or reject. The decisions save to
localStorage and export as a JSON patch, which apply_zone_fixes.py then applies. Nothing
edits the label file without the annotator's say-so — these labels are the ground truth
everything else is measured against, so a wrong "correction" is worse than the original
error.

    python scripts/zone_audit_review.py
    # review in browser, click 导出, save next to the labels, then:
    python scripts/apply_zone_fixes.py --patch ../output/zone_annotation/zone_fixes.json
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "output" / "zone_annotation"

ZONES = ["my_hand", "river", "seat_left", "seat_across", "seat_right", "__delete__"]
ZH = {"my_hand": "自家", "river": "牌池", "seat_left": "上家", "seat_across": "对家",
      "seat_right": "下家", "__delete__": "删除此框", "opponent_wall": "牌墙(废弃)", "-": "—"}


def crop_b64(img: np.ndarray, box: list[float], pad: float = 2.6, size: int = 260) -> str:
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) * pad
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
    x2, y2 = int(min(img.shape[1], cx + half)), int(min(img.shape[0], cy + half))
    out = img[y1:y2, x1:x2].copy()
    cv2.rectangle(out, (int(x - x1), int(y - y1)), (int(x + w - x1), int(y + h - y1)), (0, 0, 255), 2)
    s = size / max(out.shape[0], out.shape[1], 1)
    out = cv2.resize(out, (int(out.shape[1] * s), int(out.shape[0] * s)))
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode() if ok else ""


def propose(f: dict, item: dict) -> str:
    """A suggestion, not a decision. Position-based, deliberately simple."""
    if f["kind"] == "dead_label":
        b = f["bbox"]
        nx = (b[0] + b[2] / 2) / item["w"]
        if nx >= 0.78:
            return "seat_right"
        if nx <= 0.30:
            return "seat_left"
        return "river"
    if f["kind"] in {"isolated_label", "zone_outlier"}:
        return f.get("neighbour_zone") or ""
    return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, default=ANN / "audit.json")
    ap.add_argument("--labels", type=Path, default=ANN / "zone_labels_with_class.json")
    ap.add_argument("--out", type=Path, default=ANN / "review.html")
    args = ap.parse_args(argv)

    data = {d["image"]: d for d in json.loads(args.labels.read_text(encoding="utf-8"))}
    findings = json.loads(args.audit.read_text(encoding="utf-8"))

    images: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    merged: dict[tuple, dict] = {}
    for f in findings:
        key = (f["image"], f["box"], tuple(f["bbox"]))
        if key in merged:
            merged[key]["kinds"].append(f["kind"])
            merged[key]["details"].append(f["detail"])
        else:
            merged[key] = {**f, "kinds": [f["kind"]], "details": [f["detail"]]}

    order = {"certain": 0, "likely": 1, "suspect": 2}
    for f in sorted(merged.values(), key=lambda x: (order[x["severity"]], x["image"], x["box"])):
        img = images.setdefault(f["image"], cv2.imread(str(ANN / "images" / f["image"])))
        if img is None:
            continue
        item = data[f["image"]]
        # Missed tiles all carry box == -1, so the box number alone does not identify a
        # row — every missing box in one image would share a key and move together. The
        # bbox is what makes it unique.
        x, y, w, h = (round(v) for v in f["bbox"])
        rows.append({
            "id": f"{f['image']}#{f['box']}@{x},{y},{w},{h}",
            "label": f"{f['image']}#{f['box']}" if f["box"] >= 0 else f"{f['image']} 漏标@{x},{y}",
            "image": f["image"], "box": f["box"], "bbox": f["bbox"],
            "zone": f["zone"], "cls": f["cls"], "severity": f["severity"],
            "kinds": f["kinds"], "details": f["details"],
            "propose": propose(f, item),
            "thumb": crop_b64(img, f["bbox"]),
        })

    payload = json.dumps(rows, ensure_ascii=False)
    zh = json.dumps(ZH, ensure_ascii=False)
    zones = json.dumps(ZONES, ensure_ascii=False)

    html = """<!doctype html><meta charset="utf-8"><title>区域标注复核</title>
<style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#101012;color:#e8e8ea}
header{position:sticky;top:0;background:#18181b;border-bottom:1px solid #303034;padding:14px 20px;z-index:9}
h1{margin:0;font-size:17px} .sub{color:#9a9aa2;font-size:13px;margin-top:5px}
button{background:#2a2a30;color:#e8e8ea;border:1px solid #3a3a42;border-radius:6px;padding:5px 11px;
  font-size:13px;cursor:pointer;margin-right:6px}
button:hover{background:#35353d} button.on{background:#2f6f3f;border-color:#4a9c60;color:#fff}
button.del.on{background:#8a2a2e;border-color:#c04a50}
.card{display:flex;gap:16px;padding:14px 20px;border-bottom:1px solid #26262a;align-items:flex-start}
.card.done{opacity:.45}
img{border-radius:5px;background:#000}
.meta{flex:1;min-width:0}
.tt{font-size:14px;font-weight:600;margin-bottom:4px}
.dt{color:#a8a8b0;font-size:13px;line-height:1.55;margin:3px 0}
.sev{display:inline-block;padding:1px 7px;border-radius:4px;font-size:11px;margin-left:8px}
.certain{background:#8a2a2e} .likely{background:#8a5a15} .suspect{background:#5a5a20}
.acts{margin-top:9px}
</style>
<header><h1>区域标注复核</h1>
<div class=sub>看图选正确区域;<b>保持原样</b>表示标注没问题。决定自动保存,改完点导出。
&nbsp;<button onclick=exp()>导出 zone_fixes.json</button>
<span id=prog></span></div></header>
<div id=list></div>
<script>
const ROWS=__ROWS__, ZH=__ZH__, ZONES=__ZONES__;
// v2: row ids used to collide for missed tiles, so old saves are not reusable.
const KEY='zone_audit_decisions_v2';
let dec=JSON.parse(localStorage.getItem(KEY)||'{}');
function prog(){
  const n=Object.keys(dec).length;
  document.getElementById('prog').textContent=` — 已决定 ${n}/${ROWS.length}`;
}
function pick(id,z){
  if(dec[id]===z) delete dec[id]; else dec[id]=z;
  localStorage.setItem(KEY,JSON.stringify(dec));
  render(); prog();
}
function render(){
  document.getElementById('list').innerHTML=ROWS.map(r=>{
    const d=dec[r.id];
    const btns=['__keep__'].concat(ZONES).map(z=>{
      const keep = r.box<0 ? '不补(误检)' : '保持原样 ('+(ZH[r.zone]||r.zone)+')';
      const lab = z==='__keep__' ? keep : (r.box<0 ? '补为 '+ZH[z] : ZH[z]);
      if(z==='__delete__'&&r.box<0) return '';
      const on = d===z ? ' on' : '';
      const cls = z==='__delete__' ? ' del' : '';
      return `<button class="${cls}${on}" onclick="pick('${r.id}','${z}')">${lab}</button>`;
    }).join('');
    return `<div class="card${d?' done':''}">
      <img src="data:image/jpeg;base64,${r.thumb}">
      <div class=meta>
        <div class=tt>${r.label} &nbsp;<span style="color:#8a8a92">${r.cls}</span>
          <span class="sev ${r.severity}">${({certain:'确定错误',likely:'很可能',suspect:'存疑'})[r.severity]}</span></div>
        <div class=dt>当前标注: <b>${ZH[r.zone]||r.zone}</b></div>
        ${r.details.map(t=>'<div class=dt>· '+t+'</div>').join('')}
        <div class=acts>${btns}</div>
      </div></div>`;
  }).join('');
}
function exp(){
  const out=ROWS.filter(r=>dec[r.id]&&dec[r.id]!=='__keep__')
    .map(r=>({image:r.image,box:r.box,bbox:r.bbox,from:r.zone,to:dec[r.id]}));
  const b=new Blob([JSON.stringify(out,null,1)],{type:'application/json'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(b); a.download='zone_fixes.json'; a.click();
}
render(); prog();
</script>"""
    html = html.replace("__ROWS__", payload).replace("__ZH__", zh).replace("__ZONES__", zones)
    args.out.write_text(html, encoding="utf-8")
    print(f"{len(rows)} 条待复核 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
