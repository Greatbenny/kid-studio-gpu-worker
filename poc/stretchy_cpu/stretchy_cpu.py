from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _bezier1d(t: float, a: float, b: float) -> float:
    mt = 1.0 - t
    return 3 * mt * mt * t * a + 3 * mt * t * t * b + t * t * t


def _cubic(x: float, cx1: float, cy1: float, cx2: float, cy2: float) -> float:
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    if cx1 == cy1 and cx2 == cy2: return x
    lo, hi, t = 0.0, 1.0, x
    for _ in range(12):
        cur = _bezier1d(t, cx1, cx2)
        if abs(cur - x) < 1e-4: break
        if x > cur: lo = t
        else: hi = t
        t = (lo + hi) / 2
    return _bezier1d(t, cy1, cy2)


def ease(t: float, mode: Any) -> float:
    if mode == "linear": return t
    if mode == "stepped": return 0.0
    if mode == "ease-in": return _cubic(t, .42, 0, 1, 1)
    if mode == "ease-out": return _cubic(t, 0, 0, .58, 1)
    if isinstance(mode, list) and len(mode) == 4:
        return _cubic(t, *[float(x) for x in mode])
    return _cubic(t, .42, 0, .58, 1)


def interp_track(kfs: list[dict], t_ms: float):
    if not kfs: return None
    if t_ms <= kfs[0]["time"]: return kfs[0]["value"]
    if t_ms >= kfs[-1]["time"]: return kfs[-1]["value"]
    for a, b in zip(kfs, kfs[1:]):
        if a["time"] <= t_ms <= b["time"]:
            u = (t_ms - a["time"]) / max(1e-9, b["time"] - a["time"])
            u = ease(u, a.get("easing"))
            va, vb = a["value"], b["value"]
            if isinstance(va, bool): return va
            if isinstance(va, list):
                out = []
                for pa, pb in zip(va, vb):
                    out.append({"x": pa["x"] + (pb["x"] - pa["x"]) * u,
                                "y": pa["y"] + (pb["y"] - pa["y"]) * u})
                return out
            return va + (vb - va) * u
    return kfs[-1]["value"]


def mat_transform(tr: dict) -> np.ndarray:
    x, y = float(tr.get("x", 0)), float(tr.get("y", 0))
    r = math.radians(float(tr.get("rotation", 0)))
    sx, sy = float(tr.get("scaleX", 1)), float(tr.get("scaleY", 1))
    px, py = float(tr.get("pivotX", 0)), float(tr.get("pivotY", 0))
    c, s = math.cos(r), math.sin(r)
    T1 = np.array([[1,0,-px],[0,1,-py],[0,0,1]], np.float32)
    S = np.array([[sx,0,0],[0,sy,0],[0,0,1]], np.float32)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]], np.float32)
    T2 = np.array([[1,0,x+px],[0,1,y+py],[0,0,1]], np.float32)
    return T2 @ R @ S @ T1


@dataclass
class StretchProject:
    root: Path
    data: dict

    @classmethod
    def open(cls, path: str | Path) -> "StretchProject":
        path = Path(path)
        temp = Path(tempfile.mkdtemp(prefix="stretchy_cpu_"))
        with zipfile.ZipFile(path) as z:
            z.extractall(temp)
        pj = temp / "project.json"
        if not pj.exists():
            shutil.rmtree(temp, ignore_errors=True)
            raise ValueError("Not a valid .stretch file: project.json missing")
        data = json.loads(pj.read_text())
        return cls(temp, data)

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)

    @property
    def nodes(self): return self.data.get("nodes", [])
    @property
    def animations(self): return self.data.get("animations", [])

    def animation(self, name: str | None):
        if not self.animations: return None
        if name is None: return self.animations[0]
        lname = name.lower()
        for a in self.animations:
            if str(a.get("name", "")).lower() == lname or str(a.get("id", "")) == name:
                return a
        raise ValueError(f"Animation not found: {name}")

    def texture(self, node_id: str) -> Image.Image:
        p = self.root / "textures" / f"{node_id}.png"
        if not p.exists(): raise FileNotFoundError(f"Missing part texture: {p}")
        return Image.open(p).convert("RGBA")


def overrides(animation: dict | None, t_ms: float) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not animation: return out
    for tr in animation.get("tracks", []):
        v = interp_track(tr.get("keyframes", []), t_ms)
        if v is not None:
            out.setdefault(tr["nodeId"], {})[tr["property"]] = v
    return out


def world_matrix(node: dict, by_id: dict[str, dict], ov: dict[str, dict], cache: dict[str, np.ndarray]):
    if node["id"] in cache: return cache[node["id"]]
    tr = dict(node.get("transform") or {})
    for k in ("x","y","rotation","scaleX","scaleY"):
        if k in ov.get(node["id"], {}): tr[k] = ov[node["id"]][k]
    m = mat_transform(tr)
    parent = by_id.get(node.get("parent"))
    if parent is not None: m = world_matrix(parent, by_id, ov, cache) @ m
    cache[node["id"]] = m
    return m


def apply_m(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ones = np.ones((len(pts), 1), np.float32)
    p = np.concatenate([pts.astype(np.float32), ones], axis=1)
    return (m @ p.T).T[:, :2]


def bilinear_grid(point: np.ndarray, wd: dict, deformed: list[dict] | None) -> np.ndarray:
    if not deformed: return point
    cols, rows = int(wd.get("col", 2)), int(wd.get("row", 2))
    gx, gy = float(wd.get("gridX", 0)), float(wd.get("gridY", 0))
    gw, gh = max(1e-6, float(wd.get("gridW", 1))), max(1e-6, float(wd.get("gridH", 1)))
    u = np.clip((point[0] - gx) / gw, 0, 1)
    v = np.clip((point[1] - gy) / gh, 0, 1)
    fx, fy = u * cols, v * rows
    cx, cy = min(cols - 1, int(fx)), min(rows - 1, int(fy))
    tx, ty = fx - cx, fy - cy
    def gp(ix, iy):
        p = deformed[iy * (cols + 1) + ix]
        return np.array([p["x"], p["y"]], np.float32)
    p00, p10 = gp(cx,cy), gp(cx+1,cy)
    p01, p11 = gp(cx,cy+1), gp(cx+1,cy+1)
    top = p00 * (1-tx) + p10 * tx
    bot = p01 * (1-tx) + p11 * tx
    return top * (1-ty) + bot * ty


def warp_ancestors(node: dict, by_id: dict[str, dict]) -> list[dict]:
    out, cur = [], by_id.get(node.get("parent"))
    while cur is not None:
        if cur.get("type") == "warpDeformer": out.append(cur)
        cur = by_id.get(cur.get("parent"))
    return list(reversed(out))


def triangle_blit(canvas: np.ndarray, tex: np.ndarray, src: np.ndarray, dst: np.ndarray, opacity: float):
    x,y,w,h = cv2.boundingRect(dst.astype(np.float32))
    if w <= 0 or h <= 0: return
    src_rect = src.astype(np.float32)
    dst_local = dst.astype(np.float32) - np.array([x,y], np.float32)
    M = cv2.getAffineTransform(src_rect, dst_local)
    patch = cv2.warpAffine(tex, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = np.zeros((h,w), np.uint8)
    cv2.fillConvexPoly(mask, np.round(dst_local).astype(np.int32), 255, cv2.LINE_AA)
    x0,y0,x1,y1 = max(0,x),max(0,y),min(canvas.shape[1],x+w),min(canvas.shape[0],y+h)
    if x1 <= x0 or y1 <= y0: return
    px0, py0 = x0-x, y0-y
    crop = patch[py0:py0+(y1-y0), px0:px0+(x1-x0)].astype(np.float32)
    m = (mask[py0:py0+(y1-y0), px0:px0+(x1-x0)].astype(np.float32)/255.0)[...,None]
    a = (crop[...,3:4]/255.0) * m * opacity
    dstc = canvas[y0:y1,x0:x1].astype(np.float32)
    dstc[...,:3] = crop[...,:3] * a + dstc[...,:3] * (1-a)
    dstc[...,3:4] = 255 * (a + (dstc[...,3:4]/255.0)*(1-a))
    canvas[y0:y1,x0:x1] = np.clip(dstc,0,255).astype(np.uint8)


def render_frame(project: StretchProject, animation: dict | None, t_ms: float) -> np.ndarray:
    c = project.data.get("canvas", {})
    W,H = int(c.get("width",800)), int(c.get("height",600))
    bg = c.get("bgColor", "#ffffff")
    rgb = tuple(int(bg.lstrip('#')[i:i+2],16) for i in (0,2,4)) if isinstance(bg,str) and len(bg.lstrip('#'))==6 else (255,255,255)
    canvas = np.zeros((H,W,4), np.uint8); canvas[:] = (*rgb,255)
    ov = overrides(animation, t_ms)
    by_id = {n["id"]: n for n in project.nodes}
    cache: dict[str,np.ndarray] = {}
    parts = sorted([n for n in project.nodes if n.get("type") == "part" and n.get("visible",True)], key=lambda n:n.get("draw_order",0))
    for n in parts:
        tex = np.asarray(project.texture(n["id"]))
        mesh = n.get("mesh")
        if mesh:
            verts = np.array([[p["x"],p["y"]] for p in mesh.get("vertices",[])], np.float32)
            if "mesh_verts" in ov.get(n["id"],{}):
                verts = np.array([[p["x"],p["y"]] for p in ov[n["id"]]["mesh_verts"]], np.float32)
            uvs = np.array(mesh.get("uvs",[]),np.float32).reshape(-1,2)
            tris = mesh.get("triangles",[])
        else:
            h,w = tex.shape[:2]
            verts = np.array([[0,0],[w,0],[w,h],[0,h]],np.float32)
            uvs = np.array([[0,0],[1,0],[1,1],[0,1]],np.float32)
            tris = [[0,1,2],[0,2,3]]
        m = world_matrix(n, by_id, ov, cache)
        dstv = apply_m(m, verts)
        # Stretchy warp-deformer bounds are canvas-space. Apply ancestors after normal transforms.
        for wd in warp_ancestors(n, by_id):
            deformed = ov.get(wd["id"],{}).get("mesh_verts")
            if deformed:
                dstv = np.array([bilinear_grid(p,wd,deformed) for p in dstv],np.float32)
        srcv = uvs * np.array([tex.shape[1]-1, tex.shape[0]-1],np.float32)
        opacity = float(ov.get(n["id"],{}).get("opacity",n.get("opacity",1.0)))
        for tri in tris:
            idx = np.array(tri,dtype=int)
            triangle_blit(canvas, tex, srcv[idx], dstv[idx], opacity)
    return canvas


def render_video(path: str, animation_name: str | None, out: str, fps: int | None = None):
    p = StretchProject.open(path)
    try:
        anim = p.animation(animation_name)
        duration = int((anim or {}).get("duration", 2000))
        fps = int(fps or (anim or {}).get("fps",24) or 24)
        outp = Path(out); outp.parent.mkdir(parents=True,exist_ok=True)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg: raise RuntimeError("ffmpeg not found")
        W = int(p.data.get("canvas",{}).get("width",800)); H = int(p.data.get("canvas",{}).get("height",600))
        cmd = [ffmpeg,"-y","-f","rawvideo","-pix_fmt","rgba","-s",f"{W}x{H}","-r",str(fps),"-i","-","-an","-c:v","libx264","-pix_fmt","yuv420p",str(outp)]
        proc = subprocess.Popen(cmd,stdin=subprocess.PIPE)
        frames = max(1, math.ceil(duration/1000*fps))
        for i in range(frames):
            t = min(duration, i*1000/fps)
            proc.stdin.write(render_frame(p,anim,t).tobytes())
        proc.stdin.close(); rc = proc.wait()
        if rc != 0: raise RuntimeError(f"ffmpeg exited {rc}")
        print(json.dumps({"ok":True,"output":str(outp),"fps":fps,"frames":frames,"duration_ms":duration}))
    finally:
        p.close()


def inspect(path: str):
    p = StretchProject.open(path)
    try:
        print(json.dumps({
            "version": p.data.get("version"),
            "canvas": p.data.get("canvas"),
            "nodes": len(p.nodes),
            "parts": sum(n.get("type")=="part" for n in p.nodes),
            "warp_deformers": sum(n.get("type")=="warpDeformer" for n in p.nodes),
            "animations": [{"id":a.get("id"),"name":a.get("name"),"duration":a.get("duration"),"fps":a.get("fps"),"tracks":len(a.get("tracks",[]))} for a in p.animations],
        },indent=2))
    finally:
        p.close()


def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("inspect"); a.add_argument("project")
    r=sub.add_parser("render"); r.add_argument("project"); r.add_argument("--animation"); r.add_argument("--out",required=True); r.add_argument("--fps",type=int)
    ns=ap.parse_args()
    if ns.cmd=="inspect": inspect(ns.project)
    else: render_video(ns.project,ns.animation,ns.out,ns.fps)


if __name__ == "__main__": main()
