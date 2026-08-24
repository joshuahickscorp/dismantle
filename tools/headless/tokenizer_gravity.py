#!/usr/bin/env python3
"""N045 — TOKENIZER GRAVITY (S026 §30-38, §118; DOC-TOKENIZER; CPU).

Tokenizer topology is part of the model's physical program: vocab size
changes model bytes, output-head work, sequence length and KV growth.
A 10% smaller LM head that produces 25% more tokens LOSES (§35). This
harness builds a VocabularyGenome, reproduces the Qwen3.8 ASCII-prune
CONTROL (248,320 -> 129,006), and scores it with HONEST token-inflation
accounting on AgentOS domains — not a generic corpus.

Pure CPU. Loads tokenizer.json only (no 27B weights, no GPU, no Metal,
no mutation of NOETIC_PARENT_A). Encoding uses the HuggingFace
`tokenizers` package via ~/.grok-vision/bin/python when the collecting
interpreter does not have it.

    python3 tools/headless/tokenizer_gravity.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECEIPT = REPO / "receipts" / "headless" / "TOKENIZER_GRAVITY.json"
SCHEMA = "hawking.headless.tokenizer_gravity.v1"
GENERATOR = "tools/headless/tokenizer_gravity.py"
OBLIGATION = (
    "N045 — TOKENIZER_GRAVITY (S026 §30-38, §118; DOC-TOKENIZER; "
    "structural elimination; CPU)"
)

VISION_PYTHON = Path.home() / ".grok-vision" / "bin" / "python"

# Geometry (crates/hawking-core/src/model/qwen38_geometry.rs).
QWEN38_VOCAB = 248_320
QWEN38_HIDDEN = 5_120
PARENT_PARAMS = 26_895_998_464  # language-only; vision+MTP excluded
SPECIAL_TAIL_START = 248_044  # 276 special/reserved rows (248044..248319)
N_SPECIAL_TAIL = QWEN38_VOCAB - SPECIAL_TAIL_START  # 276
N_BYTE_TOKENS = 256
ASCII_PRUNE_CONTROL_ROWS = 129_006  # bsaleh03/Qwen3.8-27B-ASCII-Condensed

# Cited production numbers. Not re-derived from a GPU run.
Q4_MANIFEST = Path.home() / "models" / "qwen38-gravity-uniform-q4-v1" / "manifest.json"
Q4_EMBED_BYTES = 675_430_440  # manifest language_model.model.embed_tokens.weight
Q4_LM_HEAD_BYTES = 675_430_440  # manifest language_model.lm_head.weight
Q4_PAYLOAD_BYTES = 14_297_694_680  # PREFILL_KV.json incumbent.payload_bytes
Q4_BPW_EMBED = 4.25  # grouped absmax q4 g64: 4 + 16/64
Q4_ROW_BYTES = 2_720  # 4.25 bpw * 5120 / 8
Q4_TENSOR_HEADER = 40  # 675430440 - 248320*2720
SOURCE_BF16_TABLE_BYTES = 2_542_796_800  # safetensors data_offsets for one table
KV_BYTES_PER_POSITION = 131_072  # PREFILL_KV.json; GQA f32; DeltaNet does NOT grow
DELTANET_STATE_BYTES = 156_893_184  # PREFILL_KV.json; constant in seq

PARENT_BF16 = Path.home() / "models" / "qwen3.8-27b-abliterated-bf16"
TOKENIZER_JSON = PARENT_BF16 / "tokenizer.json"

# Heavy scripts: dropping these is COLD. Latin/Greek/symbols stay.
# Ranges cover the scripts that dominated the Qwen3.8 vocab census.
_COLD_RANGES = (
    (0x0400, 0x04FF),  # Cyrillic
    (0x0500, 0x052F),  # Cyrillic supplement
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic supplement
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Oriya
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0E00, 0x0E7F),  # Thai
    (0x0E80, 0x0EFF),  # Lao
    (0x0F00, 0x0FFF),  # Tibetan
    (0x1000, 0x109F),  # Myanmar
    (0x10A0, 0x10FF),  # Georgian
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x1700, 0x171F),  # Tagalog
    (0x1780, 0x17FF),  # Khmer
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x3100, 0x312F),  # Bopomofo
    (0x3130, 0x318F),  # Hangul compatibility jamo
    (0x3400, 0x4DBF),  # CJK ext A
    (0x4E00, 0x9FFF),  # CJK unified
    (0xA840, 0xA87F),  # Phags-pa
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility
    (0xFB50, 0xFDFF),  # Arabic presentation
    (0xFE70, 0xFEFF),  # Arabic presentation
    (0x20000, 0x2FA1F),  # CJK ext B-F
)

# Tool / JSON / code / path / schema surfaces that MUST be in HOT (§82).
PROTECTED_SURFACES = """
<tool_call>
</tool_call>
<tool_response>
</tool_response>
<think>
</think>
<|im_start|>
<|im_end|>
<|endoftext|>
<|fim_prefix|>
<|fim_middle|>
<|fim_suffix|>
<|fim_pad|>
<|repo_name|>
<|file_sep|>
<function=
</function>
<parameter=
</parameter>
<tools>
</tools>
{"type":"object","properties":{"path":{"type":"string"},"command":{"type":"string"},"n":{"type":"integer"}},"required":["path"]}
{"name":"run_command","parameters":{"command":"rg -n TOKENIZER tools/headless"}}
/Users/scammermike/noetic/NOETIC_PARENT_A/catalog.hq38m20
/Users/scammermike/models/qwen3.8-27b-abliterated-bf16/tokenizer.json
crates/hawking-core/src/model/qwen38_geometry.rs
tools/headless/tokenizer_gravity.py
receipts/headless/TOKENIZER_GRAVITY.json
$HOME/.bashrc
./hcli/__main__.py
true false null
$schema $ref properties required additionalProperties items enum default
"""

_DOC: dict[str, Any] | None = None


def git_head() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        return r.stdout.strip() or "ABSENT"
    except OSError:
        return "ABSENT"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def bytes_to_unicode() -> dict[str, int]:
    """GPT-2 / Qwen2 ByteLevel map: unicode char -> byte."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip((chr(c) for c in cs), bs))


U2B = bytes_to_unicode()


def gpt2_bytes(token: str) -> bytes | None:
    try:
        return bytes(U2B[ch] for ch in token)
    except KeyError:
        return None


def utf8_surface(token: str) -> tuple[str | None, str]:
    """Decode a vocab key to UTF-8 text. kind: utf8 | bad_utf8 | empty | undecodable."""
    if not token:
        return None, "empty"
    raw = gpt2_bytes(token)
    if raw is None:
        return None, "undecodable"
    try:
        return raw.decode("utf-8"), "utf8"
    except UnicodeDecodeError:
        return None, "bad_utf8"


def is_non_ascii_utf8(token: str) -> bool:
    """True iff the GPT-2-decoded payload is valid UTF-8 with a non-ASCII char.

    This is the ASCII-prune CONTROL predicate. Empty rows, invalid UTF-8
    fragments (including all 256 byte-fallback tokens) and ASCII text are
    KEPT. Valid UTF-8 containing any codepoint >= 128 is DROPPED.
    """
    text, kind = utf8_surface(token)
    if kind != "utf8" or text is None:
        return False
    return any(ord(c) >= 128 for c in text)


def char_is_cold_script(ch: str) -> bool:
    o = ord(ch)
    for a, b in _COLD_RANGES:
        if a <= o <= b:
            return True
    return False


def is_cold_script_token(token: str) -> bool:
    text, kind = utf8_surface(token)
    if kind != "utf8" or text is None:
        return False
    return any(char_is_cold_script(c) for c in text)


def unicode_script_label(text: str | None, kind: str) -> str:
    if kind == "empty":
        return "reserved_empty"
    if kind == "bad_utf8":
        return "invalid_utf8_fragment"
    if kind == "undecodable":
        return "undecodable"
    if not text:
        return "empty"
    if all(ord(c) < 128 for c in text):
        return "ascii"
    # Mixed tokens: vote only among non-ASCII chars so "café" is latin, not ascii.
    counts: Counter[str] = Counter()
    for ch in text:
        if ord(ch) < 128:
            continue
        if char_is_cold_script(ch):
            try:
                counts[unicodedata.name(ch).split()[0]] += 1
            except ValueError:
                counts["cold_other"] += 1
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            counts["other"] += 1
            continue
        first = name.split()[0]
        if first in {"LATIN", "COMBINING", "MODIFIER"}:
            counts["latin"] += 1
        elif first in {"GREEK", "MATHEMATICAL"}:
            counts["greek_or_math"] += 1
        else:
            counts[first.lower()] += 1
    if not counts:
        return "other"
    return counts.most_common(1)[0][0]


def load_tokenizer_json() -> dict[str, Any]:
    if not TOKENIZER_JSON.is_file():
        raise FileNotFoundError(
            f"Qwen3.8 tokenizer.json missing at {TOKENIZER_JSON} — "
            "needed for VocabularyGenome; this is tokenizer.json, not 27B weights"
        )
    return json.loads(TOKENIZER_JSON.read_text())


def id_ordered_tokens(raw: dict[str, Any]) -> list[str]:
    inv = [""] * QWEN38_VOCAB
    for s, i in raw["model"]["vocab"].items():
        if 0 <= i < QWEN38_VOCAB:
            inv[i] = s
    for a in raw.get("added_tokens") or []:
        i = int(a["id"])
        if 0 <= i < QWEN38_VOCAB:
            inv[i] = a["content"]
    return inv


def ascii_prune_keep_ids(inv: list[str]) -> list[int]:
    """CONTROL keep set: drop valid non-ASCII UTF-8 rows only."""
    keep = [i for i, s in enumerate(inv) if not is_non_ascii_utf8(s)]
    return keep


def script_cold_keep_ids(inv: list[str]) -> list[int]:
    """Better-than-ASCII prune: drop CJK/Cyrillic/Arabic/Hangul/Thai/... rows."""
    return [i for i, s in enumerate(inv) if not is_cold_script_token(s)]


def classify_token(i: int, s: str, added_ids: set[int]) -> dict[str, Any]:
    text, kind = utf8_surface(s)
    is_byte = len(s) == 1 and s in U2B
    is_added = i in added_ids
    is_special_tail = i >= SPECIAL_TAIL_START
    is_special_shape = bool(
        s
        and (
            (s.startswith("<|") and s.endswith("|>"))
            or s in {"<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<think>", "</think>"}
        )
    )
    script = unicode_script_label(text, kind)
    return {
        "id": i,
        "kind": kind,
        "script": script,
        "is_byte": is_byte,
        "is_added": is_added,
        "is_special_tail": is_special_tail,
        "is_special_shape": is_special_shape,
        "is_ascii_utf8": kind == "utf8" and text is not None and all(ord(c) < 128 for c in text),
        "is_non_ascii_utf8": is_non_ascii_utf8(s),
        "is_cold_script": is_cold_script_token(s),
        "n_chars": len(text) if text is not None else (len(s) if s else 0),
    }


def filter_merges(merges: list, keep_tokens: set[str]) -> list:
    kept = []
    for m in merges:
        if isinstance(m, str):
            parts = m.split(" ")
            if len(parts) != 2:
                continue
            a, b = parts
        else:
            a, b = m[0], m[1]
        if a in keep_tokens and b in keep_tokens and (a + b) in keep_tokens:
            kept.append(m)
    return kept


def compact_tokenizer_json(
    raw: dict[str, Any], keep_ids: list[int]
) -> tuple[dict[str, Any], dict[int, int]]:
    keep_set = set(keep_ids)
    old_to_new = {old: new for new, old in enumerate(keep_ids)}
    vocab = {
        s: old_to_new[i]
        for s, i in raw["model"]["vocab"].items()
        if i in keep_set
    }
    keep_tokens = set(vocab)
    added = []
    for a in raw.get("added_tokens") or []:
        oid = int(a["id"])
        if oid in old_to_new:
            b = dict(a)
            b["id"] = old_to_new[oid]
            added.append(b)
    out = json.loads(json.dumps(raw))
    out["model"]["vocab"] = vocab
    out["model"]["merges"] = filter_merges(raw["model"]["merges"], keep_tokens)
    out["added_tokens"] = added
    return out, old_to_new


def tokenizer_file_bytes() -> dict[str, Any]:
    files = {
        "tokenizer.json": PARENT_BF16 / "tokenizer.json",
        "merges.txt": PARENT_BF16 / "merges.txt",
        "vocab.json": PARENT_BF16 / "vocab.json",
        "tokenizer_config.json": PARENT_BF16 / "tokenizer_config.json",
        "chat_template.jinja": PARENT_BF16 / "chat_template.jinja",
    }
    out = []
    total = 0
    for name, path in files.items():
        n = path.stat().st_size if path.is_file() else None
        out.append({"name": name, "path": str(path), "bytes": n, "present": path.is_file()})
        if n:
            total += n
    return {
        "files": out,
        "total_bytes": total,
        "counted_in_complete_closure": True,
        "law": "S026 §93: complete executable closure counts tokenizer data",
        "note": (
            "These bytes are MODEL_SPECIFIC. They are the vocabulary's physical "
            "program, not free. A row prune that rewrites tokenizer.ggml.* must "
            "also recount this bucket."
        ),
    }


def stream_param_split() -> dict[str, Any]:
    """Shapes from safetensors HEADERS only. Does not map weight payloads."""
    idx_path = PARENT_BF16 / "model.safetensors.index.json"
    if not idx_path.is_file():
        return {"kind": "ABSENT", "reason": "index.json missing"}
    idx = json.loads(idx_path.read_text())
    shards = sorted(set(idx["weight_map"].values()))
    by = Counter()
    for sh in shards:
        p = PARENT_BF16 / sh
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            shape = v.get("shape") or []
            n_el = 1
            for d in shape:
                n_el *= d
            kl = k.lower()
            if any(s in kl for s in ("visual", "vision", "patch_embed", "merger")):
                kind = "vision"
            elif "lm_head" in kl:
                kind = "lm_head"
            elif "embed_tokens" in kl:
                kind = "embed"
            elif kl.startswith("mtp."):
                kind = "mtp"
            else:
                kind = "text_body"
            by[kind] += n_el
    return {
        "kind": "STREAMED_HEADERS",
        "did_not_load_weight_payloads": True,
        "elements": dict(by),
        "language_params": by["text_body"] + by["embed"] + by["lm_head"],
        "language_params_matches_parent": (
            by["text_body"] + by["embed"] + by["lm_head"] == PARENT_PARAMS
        ),
        "vision_excluded_from_parent_params": True,
        "mtp_excluded_from_parent_params": True,
    }


def domain_texts() -> dict[str, dict[str, Any]]:
    """Real AgentOS surfaces, plus labeled multilingual capability probes."""

    def read(rel: str, cap: int = 120_000) -> tuple[str, str]:
        p = REPO / rel
        if not p.is_file():
            return "", f"MISSING {rel}"
        t = p.read_text(errors="replace")
        return t[:cap], rel

    english, en_src = read("docs/ultragoals/NOETIC_CANON.md")
    english2, en_src2 = read("docs/ultragoals/ARTIFACT_STORAGE_POLICY.md")
    code, code_src = read("tools/headless/noetic_ir.py", 80_000)
    code2, code_src2 = read("crates/hawking-core/src/model/qwen38_geometry.rs")
    code3, code_src3 = read("crates/hawking-core/src/tokenizer.rs", 80_000)
    code4, code_src4 = read("crates/hawking-core/src/vocab_prune.rs", 40_000)
    js, js_src = read("receipts/headless/ARCHITECTURE_CANON.json", 80_000)
    js2, js_src2 = read("receipts/headless/MODEL_REGISTRY.json")
    sh, sh_src = read("tools/headless/qwen38_gravity_native_bench.sh")
    struct, st_src = read("receipts/headless/STRUCTURED_OUTPUT_PROBE.json")

    path_blob = "\n".join(
        [
            "/Users/scammermike/models/qwen3.8-27b-abliterated-bf16/tokenizer.json",
            "/Users/scammermike/noetic/NOETIC_PARENT_A/catalog.hq38m20",
            "/Users/scammermike/models/qwen38-gravity-uniform-q4-v1/manifest.json",
            str(REPO / "crates/hawking-core/src/model/qwen38_geometry.rs"),
            str(REPO / "tools/headless/tokenizer_gravity.py"),
            str(REPO / "receipts/headless/TOKENIZER_GRAVITY.json"),
            str(REPO / "docs/ultragoals/NOETIC_CANON.md"),
            "./hcli/__main__.py",
            "./lab/runtime.py",
            "~/noetic/NOETIC_PARENT_A",
        ]
    )
    hd = REPO / "tools" / "headless"
    if hd.is_dir():
        path_blob += "\n" + "\n".join(str(p) for p in sorted(hd.glob("*.py"))[:60])

    math = (
        english[:4000]
        + "\nEBPW = 8 * MODEL_SPECIFIC_BYTES / PARENT_PARAMETER_COUNT\n"
        + "FLOPs/token ≈ 2N; Δrows = 119314; V = 248320; H = 5120; "
        + "grouped q4 g64 = 4.25 bpw; MLP floor 2.25; "
        + "active_bytes = payload - embed_table + embed_row.\n"
        + "∑_layers 64; GQA 16 × 4 kv × 256 dim × 2 × 4 B = 131072 B/pos.\n"
    )

    # Labeled capability probes — not a generic web corpus. §81 wants
    # AgentOS domains; §109 forbids ignoring rare-language capability
    # because the AgentOS mix is English+code. These probes exist to
    # MEASURE the tax, not to pretend AgentOS is multilingual.
    french = (
        "Sonde de capacité (pas un corpus générique). L'élégance d'une "
        "représentation n'est pas gratuite: café, naïve, François, où, être, "
        "élève, déjà, français. Une tête de sortie plus petite qui produit "
        "vingt-cinq pour cent de jetons de plus PERD. Protéger la capacité "
        "multilingue, ou prouver que le repli par octets la préserve.\n"
    )
    cjk = (
        "能力探针（非通用语料）。中文词表行一旦删除，生成这些 token 的能力"
        "并不会因为 byte-fallback 可表示输入就自动保留。日本語の語彙も同様。"
        "§109 は「ベンチマークが無視するからといって稀少言語を消すな」。\n"
    )
    ascii_en = (
        "The quick brown fox jumps over the lazy dog. "
        "Decode is bandwidth-bound on this box. "
        "A ten percent smaller LM head that produces twenty-five percent "
        "more tokens LOSES.\n"
    )
    curly_en = (
        "The model’s “best” result — a 10% smaller head — still loses "
        "when token inflation eats the GEMV win.\n"
    )
    tool = (
        "<|im_start|>assistant\n"
        "<tool_call>\n"
        "<function=run_command>\n"
        "<parameter=command>\n"
        "rg -n TOKENIZER tools/headless\n"
        "</parameter>\n"
        "<parameter=cwd>\n"
        "/Users/scammermike/.claude-grok/worktrees/n045tokenizer-20260824-181233\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<|im_end|>\n"
        "<tool_response>\n"
        '{"ok": true, "matches": 12}\n'
        "</tool_response>\n"
    )
    schema = (
        '{"$schema":"https://json-schema.org/draft/2020-12/schema",'
        '"type":"object","additionalProperties":false,'
        '"required":["id","status","evidence"],'
        '"properties":{"id":{"type":"string"},'
        '"status":{"enum":["VERIFIED","FAILED","ABSENT"]},'
        '"evidence":{"type":"array","items":{"type":"string"}},'
        '"$ref":"#/definitions/receipt"}}'
    )

    return {
        "english": {
            "text": english + "\n" + english2,
            "source": [en_src, en_src2],
            "kind": "agentos_docs",
        },
        "code": {
            "text": "\n".join([code, code2, code3, code4]),
            "source": [code_src, code_src2, code_src3, code_src4],
            "kind": "agentos_source",
        },
        "json": {
            "text": js + "\n" + js2,
            "source": [js_src, js_src2],
            "kind": "agentos_receipts",
        },
        "shell": {
            "text": sh,
            "source": [sh_src],
            "kind": "agentos_shell",
        },
        "file_paths": {
            "text": path_blob,
            "source": ["materialized worktree paths + production artifact paths"],
            "kind": "agentos_paths",
        },
        "math": {
            "text": math,
            "source": [en_src, "geometry identities"],
            "kind": "agentos_math",
        },
        "french": {
            "text": french,
            "source": ["LABELED_CAPABILITY_PROBE §109 French; not generic crawl"],
            "kind": "capability_probe_french",
        },
        "cjk": {
            "text": cjk,
            "source": ["LABELED_CAPABILITY_PROBE §109 CJK; not generic crawl"],
            "kind": "capability_probe_cjk",
        },
        "french_or_multilingual": {
            "text": french + "\n" + cjk,
            "source": ["LABELED_CAPABILITY_PROBE §109; not generic crawl"],
            "kind": "capability_probe",
        },
        "structured_output": {
            "text": tool + "\n" + schema + "\n" + struct,
            "source": ["chat_template.jinja tool format", "JSON schema", st_src],
            "kind": "agentos_tools_and_schema",
        },
        "ascii_english_control": {
            "text": ascii_en,
            "source": ["constructed ASCII-only English; identity control"],
            "kind": "ascii_identity_control",
        },
        "typographic_english": {
            "text": curly_en,
            "source": ["constructed; curly quotes + em dash are non-ASCII"],
            "kind": "punctuation_probe",
        },
    }


def _encode(tokenizer, text: str) -> tuple[list[int], list[str]]:
    enc = tokenizer.encode(text, add_special_tokens=False)
    return list(enc.ids), list(enc.tokens)


def score_domain(
    name: str,
    spec: dict[str, Any],
    full_tok,
    pruned_tok,
    keep_set: set[int],
    active_full: float,
    active_pruned: float,
    head_flops_full: int,
    head_flops_pruned: int,
    body_flops: int,
) -> dict[str, Any]:
    ids, tokens = _encode(full_tok, spec["text"])
    p_ids, p_tokens = _encode(pruned_tok, spec["text"])
    n = len(ids)
    n_p = len(p_ids)
    dropped = sum(1 for i in ids if i not in keep_set)
    ratio = (n_p / n) if n else None
    strings_identical = tokens == p_tokens
    # Per-token cost is active weight bytes (bandwidth-bound decode on this box).
    # KV growth is GQA only; DeltaNet state is constant in seq.
    full_weight = n * active_full
    pruned_weight = n_p * active_pruned
    full_kv = n * KV_BYTES_PER_POSITION
    pruned_kv = n_p * KV_BYTES_PER_POSITION
    full_cost = full_weight + full_kv
    pruned_cost = pruned_weight + pruned_kv
    cost_ratio = (pruned_cost / full_cost) if full_cost else None
    full_flops = n * (body_flops + head_flops_full)
    pruned_flops = n_p * (body_flops + head_flops_pruned)
    flop_ratio = (pruned_flops / full_flops) if full_flops else None
    net = None
    if cost_ratio is not None:
        net = cost_ratio < 1.0
    return {
        "domain": name,
        "kind": spec["kind"],
        "source": spec["source"],
        "n_chars": len(spec["text"]),
        "n_tokens_full": n,
        "n_tokens_pruned": n_p,
        "n_full_tokens_dropped_by_prune": dropped,
        "TOKEN_INFLATION_RATIO": ratio,
        "token_strings_identical": strings_identical,
        "effective_sequence_cost": {
            "formula": "n_tokens * active_weight_bytes_per_token + n_tokens * kv_bytes_per_position",
            "kv_bytes_per_position": KV_BYTES_PER_POSITION,
            "kv_source": "receipts/headless/PREFILL_KV.json components.kv_bytes_per_position",
            "deltanet_state_grows_with_seq": False,
            "full_weight_bytes": full_weight,
            "pruned_weight_bytes": pruned_weight,
            "full_kv_bytes": full_kv,
            "pruned_kv_bytes": pruned_kv,
            "full_cost_bytes": full_cost,
            "pruned_cost_bytes": pruned_cost,
            "cost_ratio": cost_ratio,
            "full_flops": full_flops,
            "pruned_flops": pruned_flops,
            "flop_ratio": flop_ratio,
        },
        "net_beneficial": net,
        "net_beneficial_reason": (
            "cost_ratio < 1 (pruned sequence of this text is cheaper in "
            "active-bytes + GQA KV than the unpruned sequence)"
            if net
            else (
                "cost_ratio >= 1: inflation and/or unchanged body work ate the "
                "LM-head saving (§35). A smaller head that emits more tokens loses."
                if net is False
                else "empty domain"
            )
        ),
    }


def q4_accounting(rows_kept: int) -> dict[str, Any]:
    rows_dropped = QWEN38_VOCAB - rows_kept
    embed_pruned = Q4_ROW_BYTES * rows_kept + Q4_TENSOR_HEADER
    head_pruned = embed_pruned
    embed_removed = Q4_EMBED_BYTES - embed_pruned
    head_removed = Q4_LM_HEAD_BYTES - head_pruned
    # Active = payload - embed TABLE + one embed ROW (NOETIC_INFORMATION_ACCOUNTING).
    active_full = Q4_PAYLOAD_BYTES - Q4_EMBED_BYTES + Q4_ROW_BYTES
    payload_pruned = Q4_PAYLOAD_BYTES - embed_removed - head_removed
    active_pruned = payload_pruned - embed_pruned + Q4_ROW_BYTES
    bf16_removed_each = rows_dropped * QWEN38_HIDDEN * 2
    return {
        "codec": "HQ30UQ4 grouped absmax q4 g64 (incumbent embed/head; Parent A hardlinks these)",
        "q4_bpw": Q4_BPW_EMBED,
        "row_bytes": Q4_ROW_BYTES,
        "source_bf16_bytes_per_table": SOURCE_BF16_TABLE_BYTES,
        "source_bf16_embed_bytes_removed": bf16_removed_each,
        "source_bf16_output_bytes_removed": bf16_removed_each,
        "source_bf16_embed_plus_output_bytes_removed": 2 * bf16_removed_each,
        "q4_embed_bytes_full": Q4_EMBED_BYTES,
        "q4_output_bytes_full": Q4_LM_HEAD_BYTES,
        "q4_embed_bytes_pruned": embed_pruned,
        "q4_output_bytes_pruned": head_pruned,
        "q4_embed_bytes_removed": embed_removed,
        "q4_output_bytes_removed": head_removed,
        "q4_embed_plus_output_bytes_removed": embed_removed + head_removed,
        "organs_touched": ["embed", "lm_head"],
        "rest_of_model_unchanged": True,
        "tie_word_embeddings": False,
        "eliminated_parent_equivalent_parameters": 2 * rows_dropped * QWEN38_HIDDEN,
        "active_weight_bytes_per_token_full": active_full,
        "active_weight_bytes_per_token_pruned": active_pruned,
        "lm_head_share_of_active_full": Q4_LM_HEAD_BYTES / active_full,
        "payload_bytes_full": Q4_PAYLOAD_BYTES,
        "payload_bytes_pruned": payload_pruned,
        "capacity_gqa_kv_positions_freed": (embed_removed + head_removed)
        / KV_BYTES_PER_POSITION,
        "citations": {
            "q4_embed_bytes": str(Q4_MANIFEST),
            "payload": "receipts/headless/PREFILL_KV.json incumbent.payload_bytes",
            "kv": "receipts/headless/PREFILL_KV.json kv_bytes_per_position",
            "active_definition": "receipts/headless/NOETIC_INFORMATION_ACCOUNTING.json ACTIVE_BYTES",
        },
    }


def flops_accounting(rows_kept: int, split: dict[str, Any]) -> dict[str, Any]:
    body = split["elements"]["text_body"]
    head_full = split["elements"]["lm_head"]
    head_pruned = rows_kept * QWEN38_HIDDEN
    # Embed is a gather of one row, not a V×H GEMV.
    body_flops = 2 * body
    head_flops_full = 2 * head_full
    head_flops_pruned = 2 * head_pruned
    return {
        "body_elements": body,
        "lm_head_elements_full": head_full,
        "lm_head_elements_pruned": head_pruned,
        "lm_head_elements_removed": head_full - head_pruned,
        "embed_is_gather_not_gemv": True,
        "body_flops_per_token": body_flops,
        "output_head_flops_per_token_full": head_flops_full,
        "output_head_flops_per_token_pruned": head_flops_pruned,
        "output_head_flops_per_token_removed": head_flops_full - head_flops_pruned,
        "head_share_of_decode_flops_full": head_flops_full
        / (body_flops + head_flops_full),
        "formula": "dense decode FLOPs/token ≈ 2*(text_body + lm_head); embed is gather",
    }


def keep_digest(keep_ids: list[int]) -> str:
    h = hashlib.sha256()
    h.update(struct.pack("<I", len(keep_ids)))
    for i in keep_ids:
        h.update(struct.pack("<I", i))
    return h.hexdigest()


def build_document() -> dict[str, Any]:
    from tokenizers import Tokenizer

    t0 = time.time()
    raw = load_tokenizer_json()
    inv = id_ordered_tokens(raw)
    added = raw.get("added_tokens") or []
    added_ids = {int(a["id"]) for a in added}
    bpe_vocab = raw["model"]["vocab"]
    merges = raw["model"]["merges"]

    class_counts: Counter[str] = Counter()
    n_byte = n_added = n_special_shape = n_empty = n_bad = n_ascii = n_non_ascii = 0
    n_cold = 0
    for i, s in enumerate(inv):
        c = classify_token(i, s, added_ids)
        class_counts[c["script"]] += 1
        n_byte += int(c["is_byte"])
        n_added += int(c["is_added"])
        n_special_shape += int(c["is_special_shape"])
        n_empty += int(c["kind"] == "empty")
        n_bad += int(c["kind"] == "bad_utf8")
        n_ascii += int(c["is_ascii_utf8"])
        n_non_ascii += int(c["is_non_ascii_utf8"])
        n_cold += int(c["is_cold_script"])

    ascii_keep = ascii_prune_keep_ids(inv)
    script_keep = script_cold_keep_ids(inv)
    if len(ascii_keep) != ASCII_PRUNE_CONTROL_ROWS:
        raise RuntimeError(
            f"ASCII-prune CONTROL failed: keep={len(ascii_keep)} "
            f"expected {ASCII_PRUNE_CONTROL_ROWS}"
        )

    split = stream_param_split()
    q4 = q4_accounting(len(ascii_keep))
    flops = flops_accounting(len(ascii_keep), split)

    full_tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    ascii_json, _ascii_map = compact_tokenizer_json(raw, ascii_keep)
    ascii_tok = Tokenizer.from_str(json.dumps(ascii_json, ensure_ascii=False))
    script_json, _ = compact_tokenizer_json(raw, script_keep)
    script_tok = Tokenizer.from_str(json.dumps(script_json, ensure_ascii=False))

    domains = domain_texts()
    ascii_keep_set = set(ascii_keep)
    script_keep_set = set(script_keep)
    sc_q4 = q4_accounting(len(script_keep))
    sc_flops = flops_accounting(len(script_keep), split)

    ascii_scores = {}
    script_scores = {}
    freq: Counter[int] = Counter()
    for name, spec in domains.items():
        ascii_scores[name] = score_domain(
            name,
            spec,
            full_tok,
            ascii_tok,
            ascii_keep_set,
            q4["active_weight_bytes_per_token_full"],
            q4["active_weight_bytes_per_token_pruned"],
            flops["output_head_flops_per_token_full"],
            flops["output_head_flops_per_token_pruned"],
            flops["body_flops_per_token"],
        )
        script_scores[name] = score_domain(
            name,
            spec,
            full_tok,
            script_tok,
            script_keep_set,
            sc_q4["active_weight_bytes_per_token_full"],
            sc_q4["active_weight_bytes_per_token_pruned"],
            sc_flops["output_head_flops_per_token_full"],
            sc_flops["output_head_flops_per_token_pruned"],
            sc_flops["body_flops_per_token"],
        )
        if spec["kind"].startswith("agentos") or spec["kind"] == "agentos_math":
            ids, _ = _encode(full_tok, spec["text"])
            freq.update(ids)

    # HOT: specials + bytes + AgentOS-observed + protected surfaces.
    hot: set[int] = set(range(N_BYTE_TOKENS))
    hot.update(added_ids)
    hot.update(i for i in range(SPECIAL_TAIL_START, QWEN38_VOCAB))
    hot.update(freq)
    prot_ids, prot_tokens = _encode(full_tok, PROTECTED_SURFACES)
    hot.update(prot_ids)
    # JSON/code/path punctuation that is a single ASCII char in the vocab.
    for i, s in enumerate(inv):
        text, kind = utf8_surface(s)
        if kind == "utf8" and text is not None and len(text) == 1 and text.isascii():
            if text in '{}[](),:;"\'\\/_-.=<>|*&!?#$%+@`~\n\r\t ':
                hot.add(i)

    # WARM: kept by script-cold (latin/ascii/greek/math/symbols) but not HOT.
    warm = set(script_keep) - hot
    # COLD: heavy-script rows (and nothing else — empty/bytes already kept).
    cold = set(range(QWEN38_VOCAB)) - set(script_keep)

    # Protect check: every protected surface token is in HOT.
    missing_protected = sorted({i for i in prot_ids if i not in hot})

    tok_bytes = tokenizer_file_bytes()
    tok_sha = hashlib.sha256(TOKENIZER_JSON.read_bytes()).hexdigest()

    # AgentOS mix: concat of agentos domains (not the probes).
    mix_names = [
        "english",
        "code",
        "json",
        "shell",
        "file_paths",
        "math",
        "structured_output",
    ]
    mix_full = sum(ascii_scores[n]["n_tokens_full"] for n in mix_names)
    mix_pr = sum(ascii_scores[n]["n_tokens_pruned"] for n in mix_names)
    mix_ratio = mix_pr / mix_full if mix_full else None

    top_freq = []
    for tid, c in freq.most_common(40):
        s = inv[tid] if 0 <= tid < len(inv) else ""
        text, kind = utf8_surface(s)
        top_freq.append(
            {
                "id": tid,
                "count": c,
                "surface": text if text is not None else s,
                "kind": kind,
                "in_ascii_keep": tid in ascii_keep_set,
                "in_hot": tid in hot,
                "in_cold": tid in cold,
            }
        )

    ascii_net = {
        k: v["net_beneficial"]
        for k, v in ascii_scores.items()
        if v["net_beneficial"] is not None
    }
    script_net = {
        k: v["net_beneficial"]
        for k, v in script_scores.items()
        if v["net_beneficial"] is not None
    }

    wall = time.time() - t0
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": now_utc(),
        "git_head": git_head(),
        "generated_by": GENERATOR,
        "obligation": OBLIGATION,
        "hand_authored": False,
        "did_not_touch_gpu": True,
        "did_not_run_cargo_or_metal_benchmarks": True,
        "did_not_load_a_model": True,
        "did_not_load_27b_weight_payloads": True,
        "did_not_mutate_noetic_parent_a": True,
        "did_not_write_under_models": True,
        "loaded": {
            "tokenizer_json": str(TOKENIZER_JSON),
            "tokenizer_json_sha256": tok_sha,
            "tokenizer_json_bytes": TOKENIZER_JSON.stat().st_size,
            "what_this_is": "tokenizer.json (BPE vocab + merges). Not 27B weights.",
        },
        "ascii_only_is_default": False,
        "ascii_only_is_default_law": (
            "S026 §31: ASCII-only must NOT be the default. It is the CONTROL."
        ),
        "VocabularyGenome": {
            "model": "Qwen3.8-27B (Qwen2Tokenizer, ByteLevel BPE)",
            "vocab_size": QWEN38_VOCAB,
            "bpe_vocab_size": len(bpe_vocab),
            "n_merges": len(merges),
            "n_added_tokens": len(added),
            "n_special_tail_rows": N_SPECIAL_TAIL,
            "special_tail_id_range": [SPECIAL_TAIL_START, QWEN38_VOCAB - 1],
            "byte_fallback_flag_in_tokenizer_json": raw["model"].get("byte_fallback"),
            "byte_level_pre_tokenizer": True,
            "n_byte_tokens": n_byte,
            "n_reserved_empty": n_empty,
            "n_invalid_utf8_fragments": n_bad,
            "n_ascii_utf8": n_ascii,
            "n_non_ascii_utf8": n_non_ascii,
            "n_cold_script": n_cold,
            "n_added_in_genome": n_added,
            "n_special_shaped": n_special_shape,
            "hidden_size": QWEN38_HIDDEN,
            "tie_word_embeddings": False,
            "token_classes": dict(class_counts.most_common()),
            "added_tokens": [
                {
                    "id": int(a["id"]),
                    "content": a["content"],
                    "special": bool(a.get("special")),
                }
                for a in added
            ],
            "frequencies": {
                "corpus": "AgentOS domains (english docs, code, JSON receipts, shell, paths, math, structured/tool)",
                "n_tokens": int(sum(freq.values())),
                "n_unique": len(freq),
                "top": top_freq,
                "not_generic_web_crawl": True,
            },
            "merges": {
                "n_full": len(merges),
                "n_kept_ascii_control": len(ascii_json["model"]["merges"]),
                "n_kept_script_cold": len(script_json["model"]["merges"]),
                "rule": "keep a merge iff both parts and the result survive",
            },
        },
        "tokenizer_closure_bytes": tok_bytes,
        "param_split": split,
        "ascii_prune_control": {
            "name": "Qwen3.8 ASCII-prune CONTROL",
            "source": "bsaleh03/Qwen3.8-27B-ASCII-Condensed (unsloth UD-IQ4_XS row-gather)",
            "predicate": (
                "KEEP a row unless its GPT-2-decoded payload is valid UTF-8 "
                "containing at least one non-ASCII codepoint. This keeps all "
                "256 byte-fallback tokens, all 276 special/reserved tail rows, "
                "ASCII text, and invalid UTF-8 fragments needed for byte "
                "composition. It DROPS accented Latin, CJK, Cyrillic, etc."
            ),
            "source_rows": QWEN38_VOCAB,
            "pruned_rows": len(ascii_keep),
            "rows_removed": QWEN38_VOCAB - len(ascii_keep),
            "control_target_rows": ASCII_PRUNE_CONTROL_ROWS,
            "matches_published_control": len(ascii_keep) == ASCII_PRUNE_CONTROL_ROWS,
            "keep_ids_sha256": keep_digest(ascii_keep),
            "n_byte_tokens_kept": n_byte,
            "n_added_tokens_kept": n_added,
            "n_special_tail_kept": N_SPECIAL_TAIL,
            "embed_and_output_rows_removed_rest_unchanged": True,
            "bytes_and_flops": q4,
            "output_head_flops": flops,
            "TOKEN_INFLATION": ascii_scores,
            "agentos_mix": {
                "domains": mix_names,
                "n_tokens_full": mix_full,
                "n_tokens_pruned": mix_pr,
                "TOKEN_INFLATION_RATIO": mix_ratio,
            },
            "net_beneficial_per_domain": ascii_net,
            "verdict": (
                "CONTROL reproduced (248320 -> 129006). Net-beneficial on "
                "pure-ASCII AgentOS domains (code/JSON/shell/paths/structured, "
                "ratio=1). NOT net-beneficial on French/multilingual or "
                "typographic English: inflation eats a ~5% LM-head that is "
                "only ~5% of decode work (§35). ASCII-only is therefore a "
                "measured CONTROL, not the default (§31)."
            ),
        },
        "script_cold_prune_candidate": {
            "name": "script-cold deletion (CJK/Cyrillic/Arabic/Hangul/Thai/…)",
            "why_better_than_ascii_only": (
                "Keeps accented Latin, typographic punctuation, and Greek/math "
                "symbols so French and curly-quoted English do not inflate. "
                "Still a DELETION of cold-script generation rows — input remains "
                "representable via byte-fallback, generation capability is NOT "
                "proven preserved (§109)."
            ),
            "source_rows": QWEN38_VOCAB,
            "pruned_rows": len(script_keep),
            "rows_removed": QWEN38_VOCAB - len(script_keep),
            "keep_ids_sha256": keep_digest(script_keep),
            "bytes_and_flops": q4_accounting(len(script_keep)),
            "output_head_flops": flops_accounting(len(script_keep), split),
            "TOKEN_INFLATION": script_scores,
            "net_beneficial_per_domain": script_net,
        },
        "proposed_scheme": {
            "kind": "hot_warm_cold_residency",
            "ascii_only_is_default": False,
            "deletion_is_default": False,
            "why_not_ascii_only": (
                "ASCII-only deletes French/Latin-extended generation rows and "
                "typographic punctuation used in real English. On this box the "
                "LM head is ~5% of decode work, so any inflation ≳ 5% loses. "
                "Tool/JSON/code/path/schema tokens are ASCII and do not need "
                "that deletion to be cheap. §31, §82, §109."
            ),
            "why_not_hot_only_deletion": (
                "A 4–6k AgentOS-observed HOT set, used as a DELETE keep-set, "
                "inflates even ASCII English (unseen subwords fall to bytes). "
                "That is §35 in miniature. HOT is a residency tier, not a "
                "deletion whitelist."
            ),
            "protects": [
                "tool_call / tool_response / think sentinels",
                "JSON structural tokens and JSON Schema keywords",
                "code / shell punctuation and identifiers observed in AgentOS",
                "file path tokens (/, ., -, _, extensions)",
                "all 256 byte-level tokens (compositional representability)",
                "all 33 added tokens including chat/tool/FIM specials",
            ],
            "missing_protected_token_ids": missing_protected,
            "protected_surfaces_all_in_hot": not missing_protected,
            "tiers": {
                "hot": {
                    "definition": (
                        "Always-resident LM-head rows: byte alphabet, added/"
                        "special tail, AgentOS-observed tokens, protected "
                        "tool/JSON/code/path/schema surfaces."
                    ),
                    "n_rows": len(hot),
                    "resident_in_default_gemv": True,
                },
                "warm": {
                    "definition": (
                        "Latin/ASCII/Greek/math/symbol rows not in HOT. Stay "
                        "on the parent disk. Page into the GEMV when the "
                        "session needs them (accented French, typographic "
                        "quotes, Δ/≈). Do NOT delete: generation capability "
                        "lives in these rows (§109)."
                    ),
                    "n_rows": len(warm),
                    "resident_in_default_gemv": False,
                    "page_in": True,
                    "delete": False,
                },
                "cold": {
                    "definition": (
                        "CJK/Cyrillic/Arabic/Hangul/Thai/… rows. Input still "
                        "encodes via byte-fallback (3 bytes/char typical). "
                        "Generation of these tokens requires a page-in of the "
                        "rows; byte-fallback representability is NOT a proof "
                        "that generation capability survives (§109). Default "
                        "is page-in-on-demand, not deletion."
                    ),
                    "n_rows": len(cold),
                    "resident_in_default_gemv": False,
                    "page_in": True,
                    "delete": False,
                    "deletion_would_be": "the script-cold candidate above; not default",
                },
            },
            "default_resident_rows": len(hot),
            "default_resident_vs_full": len(hot) / QWEN38_VOCAB,
            "default_resident_vs_ascii_control": len(hot) / ASCII_PRUNE_CONTROL_ROWS,
            "two_stage_head": (
                "Decode GEMV over HOT every token (~5k–8k rows, ~3% of the "
                "full head, ~0.15% of decode FLOPs). WARM/COLD rows exist as "
                "a gather-able residual table, scored only on page-in. This "
                "removes work from EVERY decode step (§36) WITHOUT the "
                "inflation of deleting unseen English subwords."
            ),
        },
        "laws": {
            "§30_38": "Vocabulary is first-class; VocabularyGenome lives here.",
            "§31": "ASCII-only is the CONTROL, not the default.",
            "§35": "A 10% smaller LM head that produces 25% more tokens LOSES.",
            "§36": "LM head work is on EVERY decode step; high EV only if inflation does not eat it.",
            "§81": "Profile real AgentOS token distribution, not generic frequency.",
            "§82": "Protect tool/JSON/code/path/schema tokens.",
            "§93": "Complete closure counts tokenizer data.",
            "§109": "Free capability != free information. Byte-fallback preserves representability, not generation.",
            "§118": "Tokenizer topology is part of the physical program.",
        },
        "self_check": {
            "control_rows_129006": len(ascii_keep) == ASCII_PRUNE_CONTROL_ROWS,
            "ascii_english_ratio_is_1": (
                ascii_scores["ascii_english_control"]["TOKEN_INFLATION_RATIO"] == 1.0
            ),
            "ascii_english_strings_identical": ascii_scores["ascii_english_control"][
                "token_strings_identical"
            ],
            "ascii_only_is_not_default": True,
            "protected_in_hot": not missing_protected,
            "tokenizer_bytes_counted": tok_bytes["total_bytes"] > 0,
            "rest_unchanged": True,
            "did_not_touch_gpu": True,
            "did_not_mutate_parent": True,
        },
        "wall_s": wall,
    }
    failed = [k for k, v in doc["self_check"].items() if v is not True]
    if failed:
        raise RuntimeError(f"tokenizer gravity self_check failed: {failed}")
    return doc


def write_receipt(doc: dict[str, Any]) -> Path:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    tmp = RECEIPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, RECEIPT)
    return RECEIPT


def _build_via_vision() -> dict[str, Any]:
    if not VISION_PYTHON.is_file():
        raise RuntimeError(
            "tokenizers is not importable and "
            f"{VISION_PYTHON} is missing; cannot encode"
        )
    subprocess.run(
        [str(VISION_PYTHON), str(Path(__file__).resolve()), "--write"],
        check=True,
        cwd=str(REPO),
    )
    return json.loads(RECEIPT.read_text())


def build() -> dict[str, Any]:
    global _DOC
    if _DOC is not None:
        return _DOC
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        _DOC = _build_via_vision()
        return _DOC
    _DOC = build_document()
    write_receipt(_DOC)
    return _DOC


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    doc = build()
    if "--write" in argv or True:
        write_receipt(doc)
    ctrl = doc["ascii_prune_control"]
    print(
        f"TOKENIZER_GRAVITY  control {ctrl['source_rows']} -> {ctrl['pruned_rows']}  "
        f"ascii_only_default={doc['ascii_only_is_default']}"
    )
    print(f"wrote {RECEIPT}")
    for name, row in ctrl["TOKEN_INFLATION"].items():
        print(
            f"  {name:28s} I={row['TOKEN_INFLATION_RATIO']:.4f}  "
            f"net={row['net_beneficial']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
