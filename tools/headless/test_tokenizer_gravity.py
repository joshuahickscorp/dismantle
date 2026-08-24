"""N045 TOKENIZER_GRAVITY: VocabularyGenome, ASCII-prune CONTROL, honest inflation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from tokenizer_gravity import (  # noqa: E402
    ASCII_PRUNE_CONTROL_ROWS,
    KV_BYTES_PER_POSITION,
    PARENT_PARAMS,
    QWEN38_HIDDEN,
    QWEN38_VOCAB,
    RECEIPT,
    SCHEMA,
    SPECIAL_TAIL_START,
    ascii_prune_keep_ids,
    build,
    id_ordered_tokens,
    is_non_ascii_utf8,
    load_tokenizer_json,
    write_receipt,
)

RECEIPT_DOC = None


def receipt() -> dict:
    global RECEIPT_DOC
    if RECEIPT_DOC is None:
        RECEIPT_DOC = build()
        write_receipt(RECEIPT_DOC)
    return RECEIPT_DOC


def test_harness_writes_receipt():
    doc = receipt()
    assert RECEIPT.is_file(), f"missing {RECEIPT}"
    on_disk = json.loads(RECEIPT.read_text())
    assert on_disk["schema"] == SCHEMA
    assert doc["schema"] == SCHEMA
    assert doc["generated_by"] == "tools/headless/tokenizer_gravity.py"
    assert doc["hand_authored"] is False


def test_cpu_discipline_flags():
    doc = receipt()
    assert doc["did_not_touch_gpu"] is True
    assert doc["did_not_run_cargo_or_metal_benchmarks"] is True
    assert doc["did_not_load_a_model"] is True
    assert doc["did_not_load_27b_weight_payloads"] is True
    assert doc["did_not_mutate_noetic_parent_a"] is True
    assert doc["did_not_write_under_models"] is True
    assert "tokenizer.json" in doc["loaded"]["what_this_is"]
    assert "Not 27B" in doc["loaded"]["what_this_is"]


def test_vocabulary_genome_shape():
    g = receipt()["VocabularyGenome"]
    assert g["vocab_size"] == QWEN38_VOCAB == 248320
    assert g["hidden_size"] == QWEN38_HIDDEN == 5120
    assert g["bpe_vocab_size"] == 248044
    assert g["n_merges"] > 200000
    assert g["n_added_tokens"] == 33
    assert g["n_special_tail_rows"] == 276
    assert g["n_byte_tokens"] == 256
    assert g["byte_fallback_flag_in_tokenizer_json"] is False
    assert g["byte_level_pre_tokenizer"] is True
    assert g["tie_word_embeddings"] is False
    assert g["token_classes"]
    assert "ascii" in g["token_classes"]
    contents = {a["content"] for a in g["added_tokens"]}
    assert "<tool_call>" in contents
    assert "</tool_call>" in contents
    assert "<tool_response>" in contents
    assert "<|im_end|>" in contents
    freq = g["frequencies"]
    assert freq["n_tokens"] > 0
    assert freq["n_unique"] > 0
    assert freq["not_generic_web_crawl"] is True
    assert freq["top"]


def test_ascii_prune_control_248320_to_129006():
    """The keep predicate must reproduce the published CONTROL row count."""
    raw = load_tokenizer_json()
    inv = id_ordered_tokens(raw)
    keep = ascii_prune_keep_ids(inv)
    assert len(inv) == QWEN38_VOCAB
    assert len(keep) == ASCII_PRUNE_CONTROL_ROWS == 129006
    assert QWEN38_VOCAB - len(keep) == 119314
    # Special tail and byte alphabet survive.
    keep_set = set(keep)
    assert set(range(256)).issubset(keep_set)
    assert set(range(SPECIAL_TAIL_START, QWEN38_VOCAB)).issubset(keep_set)
    dropped = [i for i in range(QWEN38_VOCAB) if i not in keep_set]
    assert dropped
    assert all(is_non_ascii_utf8(inv[i]) for i in dropped[:50])
    ctrl = receipt()["ascii_prune_control"]
    assert ctrl["source_rows"] == 248320
    assert ctrl["pruned_rows"] == 129006
    assert ctrl["matches_published_control"] is True
    assert ctrl["embed_and_output_rows_removed_rest_unchanged"] is True
    assert ctrl["n_byte_tokens_kept"] == 256
    assert ctrl["n_special_tail_kept"] == 276


def test_embed_and_output_bytes_and_flops_removed():
    ctrl = receipt()["ascii_prune_control"]
    b = ctrl["bytes_and_flops"]
    assert b["organs_touched"] == ["embed", "lm_head"]
    assert b["rest_of_model_unchanged"] is True
    assert b["tie_word_embeddings"] is False
    assert b["q4_embed_bytes_removed"] > 0
    assert b["q4_output_bytes_removed"] > 0
    assert b["q4_embed_plus_output_bytes_removed"] == (
        b["q4_embed_bytes_removed"] + b["q4_output_bytes_removed"]
    )
    assert b["source_bf16_embed_plus_output_bytes_removed"] == 2 * 119314 * 5120 * 2
    assert b["eliminated_parent_equivalent_parameters"] == 2 * 119314 * 5120
    fl = ctrl["output_head_flops"]
    assert fl["output_head_flops_per_token_removed"] == 2 * 119314 * 5120
    assert fl["embed_is_gather_not_gemv"] is True
    assert 0.04 < fl["head_share_of_decode_flops_full"] < 0.06
    split = receipt()["param_split"]
    assert split["language_params_matches_parent"] is True
    assert split["did_not_load_weight_payloads"] is True
    assert split["language_params"] == PARENT_PARAMS


def test_token_inflation_every_required_domain():
    inf = receipt()["ascii_prune_control"]["TOKEN_INFLATION"]
    required = {
        "english",
        "code",
        "json",
        "shell",
        "file_paths",
        "math",
        "french_or_multilingual",
        "structured_output",
    }
    assert required.issubset(inf.keys())
    for name in required:
        row = inf[name]
        assert row["TOKEN_INFLATION_RATIO"] is not None
        assert row["TOKEN_INFLATION_RATIO"] >= 1.0
        assert "effective_sequence_cost" in row
        esc = row["effective_sequence_cost"]
        assert esc["kv_bytes_per_position"] == KV_BYTES_PER_POSITION
        assert esc["deltanet_state_grows_with_seq"] is False
        assert "cost_ratio" in esc
        assert "net_beneficial" in row


def test_ascii_english_identity_and_code_json_paths_are_flat():
    inf = receipt()["ascii_prune_control"]["TOKEN_INFLATION"]
    ascii_en = inf["ascii_english_control"]
    assert ascii_en["TOKEN_INFLATION_RATIO"] == 1.0
    assert ascii_en["token_strings_identical"] is True
    assert ascii_en["net_beneficial"] is True
    # Real AgentOS files are almost ASCII. A stray non-ASCII glyph (©, ≈, …)
    # produces a tiny inflation; it must stay well under the ~5% head saving
    # or the domain is not net-beneficial. Exact 1.0 is the constructed
    # ASCII control above, not a promise about every source file.
    for name in ("code", "json", "shell", "file_paths", "structured_output"):
        row = inf[name]
        assert row["TOKEN_INFLATION_RATIO"] < 1.01, (
            name,
            row["TOKEN_INFLATION_RATIO"],
        )
        assert row["net_beneficial"] is True, name


def test_french_and_cjk_inflate_and_are_not_net_beneficial():
    inf = receipt()["ascii_prune_control"]["TOKEN_INFLATION"]
    fr = inf["french_or_multilingual"]
    assert fr["TOKEN_INFLATION_RATIO"] > 1.2
    assert fr["net_beneficial"] is False
    typo = inf["typographic_english"]
    assert typo["TOKEN_INFLATION_RATIO"] > 1.0
    assert typo["net_beneficial"] is False


def test_ascii_only_is_not_the_default():
    doc = receipt()
    assert doc["ascii_only_is_default"] is False
    scheme = doc["proposed_scheme"]
    assert scheme["ascii_only_is_default"] is False
    assert scheme["deletion_is_default"] is False
    assert scheme["kind"] == "hot_warm_cold_residency"
    assert scheme["kind"] != "ascii_only_deletion"
    tiers = scheme["tiers"]
    assert set(tiers) == {"hot", "warm", "cold"}
    assert tiers["hot"]["resident_in_default_gemv"] is True
    assert tiers["warm"]["delete"] is False
    assert tiers["cold"]["delete"] is False
    assert tiers["hot"]["n_rows"] > 256
    assert tiers["warm"]["n_rows"] > 0
    assert tiers["cold"]["n_rows"] > 0
    assert scheme["protected_surfaces_all_in_hot"] is True
    assert scheme["missing_protected_token_ids"] == []
    protects = " ".join(scheme["protects"]).lower()
    assert "tool" in protects
    assert "json" in protects
    assert "code" in protects
    assert "path" in protects
    assert "schema" in protects


def test_script_cold_candidate_keeps_french_flat():
    cand = receipt()["script_cold_prune_candidate"]
    assert cand["pruned_rows"] > ASCII_PRUNE_CONTROL_ROWS
    assert cand["pruned_rows"] < QWEN38_VOCAB
    inf = cand["TOKEN_INFLATION"]
    assert inf["french"]["TOKEN_INFLATION_RATIO"] == 1.0
    assert inf["typographic_english"]["TOKEN_INFLATION_RATIO"] == 1.0
    assert inf["code"]["TOKEN_INFLATION_RATIO"] < 1.01
    # CJK still inflates — those rows were the ones dropped.
    assert inf["cjk"]["TOKEN_INFLATION_RATIO"] > 1.0


def test_closure_counts_tokenizer_data():
    cl = receipt()["tokenizer_closure_bytes"]
    assert cl["counted_in_complete_closure"] is True
    assert cl["total_bytes"] > 10_000_000
    names = {f["name"] for f in cl["files"]}
    assert "tokenizer.json" in names
    assert "merges.txt" in names
    assert "vocab.json" in names


def test_self_check_all_true():
    failed = [k for k, v in receipt()["self_check"].items() if v is not True]
    assert failed == []


def test_effective_cost_formula_is_tokens_times_per_token_plus_kv():
    row = receipt()["ascii_prune_control"]["TOKEN_INFLATION"]["code"]
    esc = row["effective_sequence_cost"]
    n = row["n_tokens_full"]
    n_p = row["n_tokens_pruned"]
    # Reconstruct: cost = n * active + n * kv
    # The receipt stores the sums; check internal consistency.
    assert esc["full_kv_bytes"] == n * KV_BYTES_PER_POSITION
    assert esc["pruned_kv_bytes"] == n_p * KV_BYTES_PER_POSITION
    assert esc["full_cost_bytes"] == esc["full_weight_bytes"] + esc["full_kv_bytes"]
    assert esc["pruned_cost_bytes"] == esc["pruned_weight_bytes"] + esc["pruned_kv_bytes"]
    assert "n_tokens * active_weight_bytes_per_token" in esc["formula"]
