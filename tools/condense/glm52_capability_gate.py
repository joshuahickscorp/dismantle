#!/usr/bin/env python3.12
"""Run capability gates against an assembled Gravity or activation-aware artifact.

`odyssey/launch/SUBSTRATE_CAPABILITY.json` decides whether an artifact may be trained on,
and until now its entries were written by hand.  A hand-written capability verdict is the
same category of thing as a hand-written test result.  This produces them.

The gates run cheapest first, so a failure costs the least:

  G_math    tokens [17, 488, 220, 17, 284] -- "2 + 2 =". One forward pass. Math-Preserve
            fails this, and running it would have caught the collapse in minutes rather
            than after a full seal and six green infrastructure gates.
  G_live    two unrelated prompts must not produce identical output. Prompt-independent
            generation is what Math-Preserve does.
Runtime parity is deliberately a separate post-capability gate. This program does
not claim it: G_math and G_live answer only whether the artifact's own numpy
execution generates prompt-conditioned output. Rust/Metal must then be compared
against that oracle before speed or sustained-runtime claims.

Reconstruction error is not consulted at any point. It cannot promote, by the frozen
tournament and by the sub-bit law.

    python3.12 tools/condense/glm52_capability_gate.py --artifact <dir> --dry-run
    python3.12 tools/condense/glm52_capability_gate.py --artifact <dir> --run --out CAPABILITY.json
    python3.12 tools/condense/glm52_capability_gate.py --artifact <dir> --emit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / "odyssey/launch/SUBSTRATE_CAPABILITY.json"

# "2 + 2 =" and two unrelated prompts, in the artifact's own tokenizer ids.
G_MATH_TOKENS = [17, 488, 220, 17, 284]
G_MATH_EXPECT_TEXT = " 4"
G_LIVE_PROMPTS = [
    ("capital", [785, 6722, 315, 9621, 374]),      # "The capital of France is"
    ("python", [7984, 264, 13020, 729, 429, 17408, 288, 264, 914, 13]),
]


def index_sha256(artifact: Path) -> str | None:
    candidates = [
        artifact / "model.gravity.index.json",
        artifact / "model.activation_aware.index.json",
    ]
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        return None
    return hashlib.sha256(present[0].read_bytes()).hexdigest()


def run_oracle(artifact: Path, tokens: list[int]) -> dict:
    """The numpy oracle reading the same container through the same codec."""
    cmd = [str(ROOT / ".venv/glm52/bin/python"),
           str(ROOT / "tools/condense/glm52_flagship_oracle.py"),
           "--dir", str(artifact), "--tokens", *map(str, tokens)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        return {"ok": False, "error": (r.stderr or r.stdout)[-400:]}
    # The oracle prints a JSON object; take the last balanced one.
    out = r.stdout
    start = out.rfind("{\n \"tokens\"")
    if start < 0:
        start = out.find("{")
    try:
        return {"ok": True, **json.loads(out[start:])}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"unparsable oracle output: {e}"}


def gate_math(artifact: Path, decode) -> dict:
    r = run_oracle(artifact, G_MATH_TOKENS)
    if not r.get("ok"):
        return {"gate": "G_math", "status": "ERROR", "detail": r.get("error")}
    top1 = r["argmax"]
    text = decode(top1) if decode else None
    passed = (text or "").strip() == G_MATH_EXPECT_TEXT.strip()
    return {
        "gate": "G_math", "status": "PASS" if passed else "FAIL",
        "tokens": G_MATH_TOKENS, "argmax": top1, "decoded": text,
        "expected": G_MATH_EXPECT_TEXT,
        "why_it_is_first": "one forward pass, and it is inside the profile a math artifact claims to preserve",
    }


def gate_live(artifact: Path, decode) -> dict:
    outs = {}
    for name, toks in G_LIVE_PROMPTS:
        r = run_oracle(artifact, toks)
        if not r.get("ok"):
            return {"gate": "G_live", "status": "ERROR", "detail": r.get("error")}
        outs[name] = {"argmax": r["argmax"], "top5": r["top5"],
                      "decoded": decode(r["argmax"]) if decode else None}
    distinct = len({v["argmax"] for v in outs.values()}) > 1
    return {
        "gate": "G_live", "status": "PASS" if distinct else "FAIL",
        "outputs": outs,
        "criterion": "two unrelated prompts must not share an argmax",
        "note": "Math-Preserve fails this: it returned byte-identical output for both.",
    }


def load_decoder(artifact: Path):
    tok = artifact / "tokenizer/tokenizer.json"
    if not tok.is_file():
        return None
    vocab = json.loads(tok.read_text())["model"]["vocab"]
    inv = {v: k for k, v in vocab.items()}
    return lambda i: inv.get(i, f"<{i}>")


def planned_commands(artifact: Path) -> list[dict]:
    python = str(ROOT / ".venv/glm52/bin/python")
    oracle = str(ROOT / "tools/condense/glm52_flagship_oracle.py")
    cases = [("math", G_MATH_TOKENS), *G_LIVE_PROMPTS]
    return [
        {
            "name": name,
            "tokens": tokens,
            "command": [
                python,
                oracle,
                "--dir",
                str(artifact),
                "--tokens",
                *map(str, tokens),
            ],
        }
        for name, tokens in cases
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", required=True, type=Path)
    ap.add_argument("--name", help="name for the capability register entry")
    ap.add_argument("--emit", action="store_true", help="write the verdict into the register")
    ap.add_argument(
        "--run",
        action="store_true",
        help="execute whole-model capability inference without changing the register",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="plan only; retained as an explicit alias for the safe default",
    )
    ap.add_argument("--out", type=Path, help="write the executed gate receipt")
    a = ap.parse_args()

    artifact = a.artifact.expanduser()
    if not artifact.is_dir():
        print(f"no artifact directory at {artifact}", file=sys.stderr)
        return 2

    sha = index_sha256(artifact)
    if sha is None:
        print("artifact must contain exactly one supported model index", file=sys.stderr)
        return 2
    if a.dry_run and (a.run or a.emit):
        ap.error("--dry-run cannot be combined with --run or --emit")
    if not a.run and not a.emit:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "heavy_execution_started": False,
                    "artifact": str(artifact),
                    "artifact_index_sha256": sha,
                    "planned_gates": planned_commands(artifact),
                    "note": "rerun with --run only under a valid heavy window; --emit also writes the capability register",
                },
                indent=2,
            )
        )
        return 0

    decode = load_decoder(artifact)
    results = [gate_math(artifact, decode)]
    # Cheapest first: only run the more expensive gate if the cheap one survived.
    if results[0]["status"] == "PASS":
        results.append(gate_live(artifact, decode))
    else:
        results.append({"gate": "G_live", "status": "NOT_RUN",
                        "why": "G_math failed; the ordering exists so a failure costs the least"})

    passed = all(r["status"] == "PASS" for r in results)
    verdict = {
        "schema": "hawking.substrate.capability_gate_run.v1",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact": str(artifact),
        "artifact_index_sha256": sha,
        "gates": results,
        "capability_verdict": "APPROVED" if passed else "REFUSED",
        "artifact_verification": True,
        "law": "APPROVED means the artifact generates. It says nothing about quality, and it is "
               "not a substitute for the support halo or long context.",
        "reconstruction_error_consulted": False,
    }
    text = json.dumps(verdict, indent=2) + "\n"
    print(text, end="")
    if a.out:
        a.out.write_text(text)

    if a.emit:
        reg = json.loads(REGISTER.read_text())
        entry = {
            "name": a.name or artifact.name,
            "path": str(artifact),
            "artifact_index_sha256": sha,
            "capability_verdict": verdict["capability_verdict"],
            "capability_reason": "produced by tools/condense/glm52_capability_gate.py",
            "capability_evidence": {"gate_run": verdict},
        }
        reg["substrates"] = [s for s in reg["substrates"]
                             if s.get("artifact_index_sha256") != sha] + [entry]
        REGISTER.write_text(json.dumps(reg, indent=2) + "\n")
        print(f"\nwrote verdict {verdict['capability_verdict']} into {REGISTER}", file=sys.stderr)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
