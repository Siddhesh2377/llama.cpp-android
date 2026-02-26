#!/usr/bin/env python3
"""Pack a JSON file into AMXP binary format."""
import struct, sys, json, os

TYPES = {"arch": 0, "backend": 1, "quant": 2, "sampling": 3}
DIRS  = {"arch": "architectures", "backend": "backends", "quant": "quant_types", "sampling": "sampling"}

def pack(json_path, plugin_type):
    with open(json_path, "r") as f:
        body = json.dumps(json.load(f), separators=(",", ":")).encode("utf-8")

    header = struct.pack("<4sHHII", b"AMXP", 1, TYPES[plugin_type], 0, len(body))

    data = json.loads(body)
    outdir = os.path.join(os.path.dirname(json_path), "..", DIRS[plugin_type])
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, data["id"] + ".amxp")

    with open(outpath, "wb") as f:
        f.write(header + body)
    print(f"  {outpath} ({len(body)+16} bytes)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <type> <json_file> [json_file...]")
        print(f"Types: {', '.join(TYPES.keys())}")
        sys.exit(1)

    ptype = sys.argv[1]
    for jf in sys.argv[2:]:
        pack(jf, ptype)
