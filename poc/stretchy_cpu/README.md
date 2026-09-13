# Kid Studio Stretchy Studio CPU 2.5D POC

This branch is an isolated proof-of-concept. It does **not** modify the existing GPU worker on `main` and does **not** change Kid Studio production routing.

## Goal

Prove that an existing approved AI-generated Kid Studio character can become a reusable 2.5D animated character without CUDA/RunPod.

First acceptance test for Mutinta:

1. idle / breathing
2. walk
3. jump
4. wave / point
5. short dance

The test is considered successful only if the character remains recognisably the approved Mutinta and motion looks acceptable for a children's cartoon rather than like disconnected cut-out pieces.

## Why Stretchy Studio

Upstream: `MangoLion/stretchystudio` (MIT licensed).

Stretchy Studio's native `.stretch` format is a ZIP containing:

- `project.json`
- `textures/<part-id>.png`
- optional `audios/...`

Its project data contains nodes, meshes, UVs, animation tracks, parameters and warp deformers. The upstream animation track shape is:

```json
{
  "nodeId": "...",
  "property": "x | y | rotation | scaleX | scaleY | opacity | visible | mesh_verts",
  "keyframes": [
    {"time": 0, "value": 0, "easing": "ease-both"}
  ]
}
```

That gives Kid Studio a durable rig/animation interchange format instead of depending on Spine exports.

## Architecture for this POC

```text
Approved Kid Studio image
        |
        v
Layer decomposition (outside this renderer)
        |
        v
Stretchy Studio rig / mesh / animation authoring
        |
        v
.stretch project
        |
        v
stretchy_cpu.py
        |
        v
CPU-rendered PNG frames
        |
        v
FFmpeg MP4
```

The renderer is intentionally server-friendly. It reads the `.stretch` project directly and does not require the Stretchy browser/WebGL renderer at production time.

## Important limitation

Stretchy Studio's DWPose auto-rigging is an authoring convenience and may use WebGPU in-browser. Kid Studio production must **not depend on WebGPU or CUDA**. The saved rig is the production input. Manual/heuristic rigging remains possible if auto-rig is unavailable.

## Current scope

`stretchy_cpu.py` implements:

- `.stretch` ZIP loading
- texture extraction
- animation/keyframe interpolation
- hierarchical transforms
- CPU triangle texture warping for mesh parts
- basic support for Stretchy warp-deformer lattices
- PNG sequence rendering
- MP4 encoding via FFmpeg

This is deliberately a compatibility renderer, not a fork of Stretchy Studio's UI.

## Usage

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r poc/stretchy_cpu/requirements.txt
python poc/stretchy_cpu/stretchy_cpu.py inspect character.stretch
python poc/stretchy_cpu/stretchy_cpu.py render character.stretch --animation idle --out /tmp/mutinta-idle.mp4
```

The actual Mutinta test still requires a saved `.stretch` rig (or the original character PNG to prepare one in Stretchy Studio). No paid provider calls are part of this POC.
