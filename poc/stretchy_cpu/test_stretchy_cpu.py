import json
import tempfile
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

from stretchy_cpu import StretchProject, overrides, render_frame


def make_fixture(path: Path):
    node_id = "mutinta-test-part"
    project = {
        "version": "0.1",
        "canvas": {"width": 320, "height": 180, "bgEnabled": True, "bgColor": "#ffffff"},
        "textures": [{"id": node_id, "source": f"textures/{node_id}.png"}],
        "nodes": [{
            "id": node_id,
            "type": "part",
            "name": "Test Character",
            "parent": None,
            "draw_order": 0,
            "opacity": 1,
            "visible": True,
            "transform": {"x": 20, "y": 50, "rotation": 0, "scaleX": 1, "scaleY": 1, "pivotX": 0, "pivotY": 0},
            "mesh": None,
        }],
        "parameters": [],
        "physics_groups": [],
        "animations": [{
            "id": "walk",
            "name": "walk",
            "duration": 1000,
            "fps": 10,
            "tracks": [{
                "nodeId": node_id,
                "property": "x",
                "keyframes": [
                    {"time": 0, "value": 20, "easing": "linear"},
                    {"time": 1000, "value": 220, "easing": "linear"}
                ]
            }],
            "audioTracks": []
        }]
    }
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "textures").mkdir()
        img = Image.new("RGBA", (64, 96), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((16, 0, 48, 32), fill=(245, 180, 120, 255))
        d.rounded_rectangle((10, 28, 54, 90), radius=10, fill=(230, 80, 120, 255))
        img.save(td / "textures" / f"{node_id}.png")
        (td / "project.json").write_text(json.dumps(project))
        with zipfile.ZipFile(path, "w") as z:
            z.write(td / "project.json", "project.json")
            z.write(td / "textures" / f"{node_id}.png", f"textures/{node_id}.png")


def test_animation_interpolation_and_render():
    with tempfile.TemporaryDirectory() as td:
        fixture = Path(td) / "test.stretch"
        make_fixture(fixture)
        p = StretchProject.open(fixture)
        try:
            anim = p.animation("walk")
            ov = overrides(anim, 500)
            assert abs(ov["mutinta-test-part"]["x"] - 120) < 0.001
            frame = render_frame(p, anim, 500)
            assert frame.shape == (180, 320, 4)
            assert frame[..., 3].max() == 255
            # Character should add non-white pixels near the midpoint of its path.
            assert (frame[45:160, 100:210, :3] < 245).any()
        finally:
            p.close()


if __name__ == "__main__":
    test_animation_interpolation_and_render()
    print("stretchy CPU POC test: OK")
