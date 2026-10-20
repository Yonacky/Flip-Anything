"""Write an explicit move command JSON. Sends nothing by itself."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rot_x(a):
    import math
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def main():
    pose = json.loads(sys.argv[1])
    assert len(pose) == 6
    payload = {"action": "move", "pose": [float(v) for v in pose], "grip_required": False,
               "capture": True, "max_rotation_deg": float(sys.argv[2]),
               "max_translation_mm": float(sys.argv[3]),
               "max_joint_deg": float(sys.argv[4]), "speed": float(sys.argv[5]),
               "timeout_s": float(sys.argv[6])}
    out = ROOT / (sys.argv[7] if len(sys.argv) > 7 else "scene/next_action.json")
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf8")
    print("wrote", out)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
