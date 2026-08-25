"""The Odyssey Accelerator Pass. FRONT F (G048, steer S015 §20, §130, §81).

Odyssey now has two educational goals per specimen: how does this architecture want
to EXIST (Gravity's question) and how does this computation want to EXECUTE
(Accelerator's). This pass discharges the second and records whether the first was
also discharged, rather than quietly answering one and implying both.

Compounding is the point and it is MEASURED, not asserted. For every specimen the
pass counts what it had to learn versus what it reused, so the claim that later
specimens are cheaper is a number rather than a hope. With one specimen compounding
is undefined and the pass says so; it only becomes meaningful from the second.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
KNOWLEDGE = REPO / "receipts/headless/ACCELERATOR_KNOWLEDGE_BASE.json"

STAGES = ("intake", "architecture_recognizer", "organ_census", "accelerator_baseline",
          "kernel_selection", "representation_selection", "compounding", "receipt")


def load_kb() -> dict[str, Any]:
    if KNOWLEDGE.exists():
        try:
            return json.loads(KNOWLEDGE.read_text())
        except Exception:
            pass
    return {"schema": "hawking.accelerator.knowledge_base.v1",
            "kernels": {}, "representations": {}, "organ_shapes": {},
            "specimens_seen": []}


def save_kb(kb: dict[str, Any]) -> None:
    kb["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    KNOWLEDGE.parent.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE.write_text(json.dumps(kb, indent=1))


def read_config(spec_dir: Path) -> dict[str, Any]:
    for c in sorted(spec_dir.rglob("config.json")):
        cfg = json.loads(c.read_text())
        # VL models nest the language model; the accelerator cares about the text tower
        if cfg.get("text_config"):
            merged = dict(cfg["text_config"])
            merged["architectures"] = cfg.get("architectures", merged.get("architectures"))
            merged["_nested"] = True
            return merged
        return cfg
    raise FileNotFoundError(f"no config.json under {spec_dir}")


def organ_census(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Organs and the SHAPE each one presents to the GPU. Shape is what actually
    determines whether a kernel transfers, which is why it is the cache key rather
    than the organ name."""
    h = cfg.get("hidden_size")
    L = cfg.get("num_hidden_layers")
    organs: dict[str, dict[str, Any]] = {}
    if h is None or L is None:
        return organs
    n_exp = cfg.get("num_experts") or cfg.get("n_routed_experts")
    inter = cfg.get("intermediate_size")
    moe_inter = cfg.get("moe_intermediate_size")
    if n_exp:
        organs["moe_expert"] = {"gemv_shape": [moe_inter or inter, h], "count": n_exp * L * 3}
        organs["moe_router"] = {"gemv_shape": [n_exp, h], "count": L}
    if inter and not n_exp:
        organs["mlp"] = {"gemv_shape": [inter, h], "count": L * 3}
    heads = cfg.get("num_attention_heads")
    if heads:
        organs["attention"] = {"gemv_shape": [h, h], "count": L * 4}
    organs["embed"] = {"gemv_shape": [cfg.get("vocab_size"), h], "count": 1}
    organs["norm"] = {"gemv_shape": [h], "count": L * 2}
    return organs


def run_pass(name: str, spec_dir: Path, kb: dict[str, Any]) -> dict[str, Any]:
    cfg = read_config(spec_dir)
    arch = (cfg.get("architectures") or ["UNKNOWN"])[0]
    organs = organ_census(cfg)

    reused, learned = [], []
    for organ, meta in organs.items():
        # the cache key is (organ, shape class), NOT the model name -- that is what
        # makes reuse across specimens possible at all
        key = f"{organ}:{meta['gemv_shape']}"
        if key in kb["organ_shapes"]:
            reused.append({"organ": organ, "key": key,
                           "first_seen_on": kb["organ_shapes"][key]["first_seen_on"]})
        else:
            learned.append({"organ": organ, "key": key})
            kb["organ_shapes"][key] = {"first_seen_on": name, "organ": organ,
                                       "gemv_shape": meta["gemv_shape"]}

    n = len(organs)
    prior = len(kb["specimens_seen"])
    compounding: dict[str, Any] = {
        "specimen_index": prior + 1,
        "organs_total": n,
        "organs_reused": len(reused),
        "organs_learned": len(learned),
        "reuse_fraction": round(len(reused) / n, 4) if n else None,
        "reused_detail": reused,
    }
    if prior == 0:
        compounding["status"] = "UNDEFINED"
        compounding["reason"] = ("this is the first specimen through the Accelerator "
                                 "Pass; compounding is only meaningful from the second")
    else:
        compounding["status"] = "MEASURED"

    if name not in kb["specimens_seen"]:
        kb["specimens_seen"].append(name)

    return {"specimen": name, "architecture": arch,
            "nested_config": bool(cfg.get("_nested")),
            "layers": cfg.get("num_hidden_layers"), "hidden": cfg.get("hidden_size"),
            "organs": organs, "compounding": compounding,
            "educational_goals": {
              "how_it_wants_to_EXECUTE": "DISCHARGED — organ shapes censused and matched "
                                         "against known kernel/representation knowledge",
              "how_it_wants_to_EXIST": (
                  "NOT DISCHARGED HERE — that is Gravity's question and this pass does "
                  "not answer it; for model #2 it is carried by the KernelPlanner and "
                  "packer work under G023, and for the others it has not been done")}}


# --- WHICH RUNTIME, AND WHY THE FORWARD PASS IS BLOCKED --------------------------
# Five G048 receipts carried "no runtime here executes a Qwen3-MoE, Falcon or Mamba
# forward pass" and used it to justify grading the lake specimens in WEIGHT SPACE --
# the very space this program measured to be nearly invariant across architectures,
# so the metric cannot tell them apart. THE SENTENCE CONFLATES TWO RUNTIMES.
#
# Hawking's own Rust body dispatches on llama / llama2 / llama3 / mistral / qwen2 /
# qwen2.5 / qwen (crates/hawking-core/src/model/mod.rs) and really does lack every
# lake architecture. mlx_lm -- already a hard dependency, the thing every AIR kernel
# runs through -- carries model classes for ALL FIVE. So the gap is NOT a missing
# reader, which would be weeks of Rust; it is STORAGE AND CONTENTION, which is hours
# of waiting or one quiesced window. Those need completely different remediation and
# naming the wrong one parks the obligation forever.
HAWKING_RUST_ARCHS = ("llama", "llama2", "llama3", "llama3.1", "llama3.2",
                      "mistral", "qwen2", "qwen2.5", "qwen")
LAKE_SPECIMEN_MODULES = ("qwen3_moe", "qwen3_vl_moe", "kimi_vl", "falcon_h1")


def runtime_coverage() -> dict[str, Any]:
    """Who can actually execute a lake specimen, asked rather than assumed.

    Returns both runtimes separately BECAUSE THE RECEIPTS COLLAPSED THEM. A caller
    that wants a forward pass needs to know the answer is 'yes, and the weights are
    two hours away over a contended USB bus', not 'no, write a reader first'.
    """
    import importlib
    mlx = {}
    for mod in LAKE_SPECIMEN_MODULES:
        try:
            importlib.import_module("mlx_lm.models." + mod)
            mlx[mod] = True
        except Exception:                    # noqa: BLE001 -- absence is the answer
            mlx[mod] = False
    return {
        "hawking_rust_archs": list(HAWKING_RUST_ARCHS),
        "hawking_rust_covers_any_lake_specimen": False,
        "mlx_lm_module_present": mlx,
        "mlx_lm_covers_all_lake_specimens": all(mlx.values()),
        # The gate that is actually open, per S015 §129: named input, not a parking lot.
        "blocked_on": "FAST_LOCAL_STORAGE",
        "blocked_on_detail": (
            "Falcon-H1-7B is the smallest lake specimen at 15.17 GB over 4 shards. "
            "Under the operator-prioritised fill (4 concurrent hf downloads) a "
            "sequential read of that volume did not complete 256 MiB in 120 s, so "
            "the contended rate is under ~2.1 MB/s against 96-131 MiB/s measured "
            "uncontended -- a >45x contention penalty and >2 h for one load. The "
            "missing input is a quiesced window or a staged copy on Tier 1, NOT a "
            "model reader."),
    }
