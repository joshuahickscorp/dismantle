//! Bounded native Flash-Next Noetic routed-expert body dispatch.
//!
//! This executable consumes the native router-selection receipt and the
//! persisted source-independent body/kernel receipts for the selected
//! experts.  It dispatches one bounded Q4/G64 body window per selected expert
//! through the existing Metal matvec kernel, then performs the selected-weight
//! gather on the host.  The gather is deliberately reported as host-side
//! composition: this is not a fused expert, complete expert, token, TPS, or
//! EBPW measurement.  The `--gate-up-swiglu` mode consumes full selected-expert
//! bodies and exercises the existing native gate+up/SwiGLU kernel, while still
//! stopping before down projection and the complete token graph.  The
//! `--expert-composition` mode chains that native gate+up/SwiGLU output directly
//! into independently persisted down-projection bodies, proving only the
//! bounded selected-expert composition and never a complete Flash token graph.

#![recursion_limit = "512"]

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("Flash Noetic routed dispatch requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
mod macos {
    use half::f16;
    use hawking_core::metal::{MetalContext, MetalDispatchTiming};
    use metal::Buffer;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    const SCHEMA: &str = "hawking.flash_noetic_routed_expert_dispatch_native.v1";
    const GATE_UP_SCHEMA: &str = "hawking.flash_noetic_routed_expert_gate_up_swiglu_native.v1";
    const COMPOSITION_SCHEMA: &str = "hawking.flash_noetic_routed_expert_composition_native.v1";
    const BODY_SCHEMA: &str = "hcli.agentos.flash_noetic_component_body.v1";
    const KERNEL_SCHEMA: &str = "hawking.flash_noetic_q4_kernel_parity.v1";
    const ROUTER_SCHEMA: &str = "hawking.flash_noetic_router_selection_native.v1";
    const CAMPAIGN_SCHEMA: &str = "hcli.agentos.flash_noetic_component_campaign.v1";
    const NOMENCLATURE_VERSION: &str = "HAWKING_NOMENCLATURE_V1";
    const REPO_ID: &str = "Qwen/Qwen3.8-Flash-Next";
    const PINNED_REVISION: &str = "34567a4712bc9766c4449e2e98e4468bfa24d915";
    const TENSOR_NAME: &str = "model.language_model.layers.0.mlp.experts.gate_up_proj";
    const DOWN_TENSOR_NAME: &str = "model.language_model.layers.0.mlp.experts.down_proj";
    const GROUP_SIZE: usize = 64;
    const CODE_BYTES_PER_GROUP: usize = GROUP_SIZE / 2;
    const KERNEL_NAME: &str = "qwen_uniform_q4_group64_matvec";
    const GATE_UP_KERNEL_NAME: &str =
        "qwen_uniform_q4_group64_matvec_gate_up_swiglu_geo_tpr64_tg128";
    const REFERENCE_MULTIPLIER: usize = 71;
    const REFERENCE_MODULUS: usize = 509;
    const REFERENCE_OFFSET: f32 = 254.0;
    const DEFAULT_WARMUP: usize = 2;
    const DEFAULT_REPS: usize = 7;
    const OUTPUT_ERROR_TOLERANCE: f32 = 2.0e-3;
    const GATE_UP_BODY_ROWS: usize = 1280;
    const GATE_UP_ROWS: usize = 640;
    const DOWN_BODY_ROWS: usize = 2560;
    const DOWN_COLUMNS: usize = 640;
    const MANIFEST_PATH: &str =
        "/Volumes/corpdrive/hawking-modellake/manifests/Qwen--Qwen3.8-Flash-Next@34567a4712bc.json";

    struct Args {
        root: PathBuf,
        router_receipt: PathBuf,
        campaign_receipt: PathBuf,
        warmup: usize,
        reps: usize,
        out: PathBuf,
        gate_up_swiglu: bool,
        expert_composition: bool,
    }

    struct PackedBody {
        path: PathBuf,
        receipt_sha256: String,
        body_sha256: String,
        expert_index: usize,
        row_start: usize,
        rows: usize,
        columns: usize,
        code_bytes: usize,
        scale_bytes: usize,
        codes: Vec<u8>,
        scales: Vec<u8>,
    }

    struct LoadedBody {
        body: PackedBody,
        body_receipt: Value,
        body_receipt_path: PathBuf,
        kernel_receipt: Value,
        kernel_receipt_path: PathBuf,
        kernel_receipt_sha256: String,
        codes: Buffer,
        scales: Buffer,
        output: Buffer,
        expected: Vec<f32>,
    }

    struct LoadedGateUp {
        body: PackedBody,
        gate: PackedBody,
        up: PackedBody,
        body_receipt: Value,
        body_receipt_path: PathBuf,
        kernel_receipt: Value,
        kernel_receipt_path: PathBuf,
        kernel_receipt_sha256: String,
        gate_codes: Buffer,
        gate_scales: Buffer,
        up_codes: Buffer,
        up_scales: Buffer,
        output: Buffer,
        expected: Vec<f32>,
    }

    struct LoadedComposition {
        gate_up_body: PackedBody,
        gate: PackedBody,
        gate_up_body_receipt: Value,
        gate_up_body_receipt_path: PathBuf,
        gate_up_kernel_receipt: Value,
        gate_up_kernel_receipt_path: PathBuf,
        gate_up_kernel_receipt_sha256: String,
        down: PackedBody,
        down_body_receipt: Value,
        down_body_receipt_path: PathBuf,
        down_kernel_receipt: Value,
        down_kernel_receipt_path: PathBuf,
        down_kernel_receipt_sha256: String,
        gate_codes: Buffer,
        gate_scales: Buffer,
        up_codes: Buffer,
        up_scales: Buffer,
        gate_up_output: Buffer,
        down_codes: Buffer,
        down_scales: Buffer,
        down_output: Buffer,
        expected_gate_up: Vec<f32>,
        expected_down: Vec<f32>,
    }

    fn repository_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
    }

    fn default_lake_root() -> PathBuf {
        if let Some(value) = env::var_os("HCLI_FLASH_NEXT_ROOT") {
            return PathBuf::from(value);
        }
        PathBuf::from(
            "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc",
        )
    }

    fn parse_usize(value: Option<String>, flag: &str) -> Result<usize, Box<dyn Error>> {
        value
            .ok_or_else(|| format!("missing value after {flag}"))?
            .parse::<usize>()
            .map_err(|error| format!("invalid {flag}: {error}").into())
    }

    fn parse_args() -> Result<Args, Box<dyn Error>> {
        let repo = repository_root();
        let mut args = Args {
            root: default_lake_root(),
            router_receipt: repo
                .join("receipts/headless/FLASH_NOETIC_ROUTER_SELECTION_NATIVE.json"),
            campaign_receipt: repo
                .join("receipts/headless/FLASH_NOETIC_ROUTED_EXPERT_COMPONENT_CAMPAIGN.json"),
            warmup: DEFAULT_WARMUP,
            reps: DEFAULT_REPS,
            out: repo.join("receipts/headless/FLASH_NOETIC_ROUTED_EXPERT_DISPATCH_NATIVE.json"),
            gate_up_swiglu: false,
            expert_composition: false,
        };
        let mut values = env::args().skip(1);
        while let Some(flag) = values.next() {
            match flag.as_str() {
                "--root" => args.root = PathBuf::from(values.next().ok_or("missing --root")?),
                "--router-receipt" => {
                    args.router_receipt =
                        PathBuf::from(values.next().ok_or("missing --router-receipt")?)
                }
                "--campaign-receipt" => {
                    args.campaign_receipt =
                        PathBuf::from(values.next().ok_or("missing --campaign-receipt")?)
                }
                "--warmup" => args.warmup = parse_usize(values.next(), &flag)?,
                "--reps" => args.reps = parse_usize(values.next(), &flag)?,
                "--out" => args.out = PathBuf::from(values.next().ok_or("missing --out")?),
                "--gate-up-swiglu" => args.gate_up_swiglu = true,
                "--expert-composition" => args.expert_composition = true,
                "--help" | "-h" => {
                    println!(
                        "usage: flash_noetic_routed_expert_dispatch [--root DIR] \
                         [--router-receipt FILE] [--campaign-receipt FILE] \
                         [--warmup N] [--reps N] [--out FILE] [--gate-up-swiglu] \
                         [--expert-composition]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other}").into()),
            }
        }
        if args.warmup > 32 {
            return Err("--warmup must be <= 32".into());
        }
        if args.reps == 0 || args.reps > 128 {
            return Err("--reps must be in 1..=128".into());
        }
        if args.gate_up_swiglu && args.expert_composition {
            return Err("--gate-up-swiglu and --expert-composition are mutually exclusive".into());
        }
        let default_out =
            repo.join("receipts/headless/FLASH_NOETIC_ROUTED_EXPERT_DISPATCH_NATIVE.json");
        if args.out == default_out {
            if args.expert_composition {
                args.out = repo
                    .join("receipts/headless/FLASH_NOETIC_ROUTED_EXPERT_COMPOSITION_NATIVE.json");
            } else if args.gate_up_swiglu {
                args.out = repo.join(
                    "receipts/headless/FLASH_NOETIC_ROUTED_EXPERT_GATE_UP_SWIGLU_NATIVE.json",
                );
            }
        }
        Ok(args)
    }

    fn sha256_bytes(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn read_json(path: &Path) -> Result<(Value, String), Box<dyn Error>> {
        let canonical = path.canonicalize()?;
        let bytes = fs::read(canonical)?;
        let value = serde_json::from_slice(&bytes)?;
        Ok((value, sha256_bytes(&bytes)))
    }

    fn string_field<'a>(value: &'a Value, field: &str) -> Result<&'a str, Box<dyn Error>> {
        value
            .get(field)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("receipt is missing string field {field}").into())
    }

    fn usize_field(value: &Value, field: &str) -> Result<usize, Box<dyn Error>> {
        value
            .get(field)
            .and_then(Value::as_u64)
            .and_then(|number| usize::try_from(number).ok())
            .ok_or_else(|| format!("receipt is missing integer field {field}").into())
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn validate_manifest(root: &Path) -> Result<Value, Box<dyn Error>> {
        let manifest_path = Path::new(MANIFEST_PATH);
        let (manifest, digest) = read_json(manifest_path)?;
        if string_field(&manifest, "repo")? != REPO_ID
            || string_field(&manifest, "revision")? != PINNED_REVISION
            || string_field(&manifest, "resolved_sha")? != PINNED_REVISION
        {
            return Err("ModelLake manifest is not the exact pinned Flash target".into());
        }
        let manifest_root = manifest
            .get("path")
            .and_then(Value::as_str)
            .ok_or("ModelLake manifest has no specimen path")?;
        if Path::new(manifest_root).canonicalize()? != root.canonicalize()? {
            return Err("selected specimen root does not match the pinned manifest".into());
        }
        Ok(json!({
            "path": manifest_path,
            "sha256": digest,
            "repo": REPO_ID,
            "revision": PINNED_REVISION,
            "resolved_sha": PINNED_REVISION,
            "n_files": manifest.get("n_files"),
            "bytes": manifest.get("bytes"),
            "label": "[V]",
        }))
    }

    fn validate_router(
        path: &Path,
    ) -> Result<(Value, String, Vec<usize>, Vec<f32>), Box<dyn Error>> {
        let (receipt, digest) = read_json(path)?;
        if string_field(&receipt, "schema")? != ROUTER_SCHEMA
            || string_field(&receipt, "status")? != "PASSED"
            || string_field(&receipt, "repo")? != REPO_ID
            || string_field(&receipt, "pinned_revision")? != PINNED_REVISION
            || string_field(&receipt, "nomenclature_version")? != NOMENCLATURE_VERSION
        {
            return Err("router receipt is not a PASSED pinned native Noetic receipt".into());
        }
        if receipt
            .get("native_selection_execution_observed")
            .and_then(Value::as_bool)
            != Some(true)
            || receipt.get("promotion_allowed").and_then(Value::as_bool) != Some(false)
        {
            return Err("router receipt fails native-observation or promotion guards".into());
        }
        let loader = receipt
            .get("native_loader")
            .ok_or("router receipt has no native_loader")?;
        if loader
            .get("source_independent_execution")
            .and_then(Value::as_bool)
            != Some(true)
            || loader
                .get("source_tensor_read_for_execution")
                .and_then(Value::as_bool)
                != Some(false)
        {
            return Err("router receipt is not source-independent".into());
        }
        let selection = receipt
            .get("selection")
            .ok_or("router receipt has no selection")?;
        let ids = selection
            .get("expert_ids")
            .and_then(Value::as_array)
            .ok_or("router receipt has no selection expert ids")?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|number| usize::try_from(number).ok())
                    .ok_or_else(|| "router receipt contains an invalid expert id".into())
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        let weights = selection
            .get("selected_weights")
            .and_then(Value::as_array)
            .ok_or("router receipt has no selected weights")?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|number| number as f32)
                    .filter(|number| number.is_finite() && *number >= 0.0)
                    .ok_or_else(|| "router receipt contains an invalid selected weight".into())
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        if ids.is_empty() || ids.len() != weights.len() || ids.len() > 64 {
            return Err("router receipt selection shape is outside bounded dispatch limits".into());
        }
        if ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("router receipt contains duplicate selected experts".into());
        }
        Ok((receipt, digest, ids, weights))
    }

    fn validate_campaign(path: &Path) -> Result<(Value, String), Box<dyn Error>> {
        let (campaign, digest) = read_json(path)?;
        if string_field(&campaign, "schema")? != CAMPAIGN_SCHEMA
            || string_field(&campaign, "status")? != "PASSED"
            || string_field(&campaign, "repo")? != REPO_ID
            || string_field(&campaign, "pinned_revision")? != PINNED_REVISION
            || string_field(&campaign, "nomenclature_version")? != NOMENCLATURE_VERSION
        {
            return Err("routed-expert campaign is not a PASSED pinned Noetic campaign".into());
        }
        if campaign
            .get("source_independent_execution")
            .and_then(Value::as_bool)
            != Some(true)
            || campaign
                .get("candidate_body_persisted")
                .and_then(Value::as_bool)
                != Some(true)
            || campaign.get("promotion_allowed").and_then(Value::as_bool) != Some(false)
        {
            return Err("routed-expert campaign fails source-independent guards".into());
        }
        Ok((campaign, digest))
    }

    fn validate_body_for(
        path: &Path,
        expected_tensor: &str,
        expected_shape: [usize; 3],
        max_rows: usize,
    ) -> Result<(PackedBody, Value), Box<dyn Error>> {
        let canonical_receipt = path.canonicalize()?;
        let (receipt, receipt_sha256) = read_json(&canonical_receipt)?;
        if string_field(&receipt, "schema")? != BODY_SCHEMA
            || string_field(&receipt, "status")? != "PASSED"
            || string_field(&receipt, "repo")? != REPO_ID
            || string_field(&receipt, "pinned_revision")? != PINNED_REVISION
            || string_field(&receipt, "nomenclature_version")? != NOMENCLATURE_VERSION
        {
            return Err("expert body receipt is not a PASSED pinned Noetic body".into());
        }
        if receipt.get("source_independent").and_then(Value::as_bool) != Some(true)
            || receipt
                .get("candidate_body_persisted")
                .and_then(Value::as_bool)
                != Some(true)
            || receipt.get("body_mutated").and_then(Value::as_bool) != Some(false)
            || receipt.get("model_loaded").and_then(Value::as_bool) != Some(false)
        {
            return Err("expert body receipt fails source-independent execution guards".into());
        }
        let source = receipt
            .get("source_block")
            .ok_or("expert body receipt has no source_block")?;
        if string_field(source, "tensor_name")? != expected_tensor
            || string_field(source, "dtype")?.to_uppercase() != "BF16"
        {
            return Err(format!(
                "expert body source block is not the pinned {expected_tensor} tensor"
            )
            .into());
        }
        let shape = source
            .get("shape")
            .and_then(Value::as_array)
            .ok_or("expert body source block has no shape")?;
        if shape.len() != 3
            || shape
                .iter()
                .zip(expected_shape)
                .any(|(value, expected)| value.as_u64() != Some(expected as u64))
        {
            return Err(format!(
                "expert body source shape is not [{},{},{}]",
                expected_shape[0], expected_shape[1], expected_shape[2]
            )
            .into());
        }
        let expert_index = usize_field(source, "expert_index")?;
        let row_start = usize_field(source, "row_start")?;
        let rows = usize_field(source, "row_count")?;
        let columns = shape[2]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("expert body column dimension is invalid")?;
        if expert_index >= 512 || rows == 0 || rows > max_rows || row_start + rows > max_rows {
            return Err("expert body window is outside bounded dispatch limits".into());
        }
        if usize_field(source, "bytes")? != rows * columns * 2 {
            return Err("expert body source block byte count is invalid".into());
        }
        let body_record = receipt
            .get("body")
            .ok_or("expert body receipt has no body")?;
        let body_path = Path::new(string_field(body_record, "path")?).canonicalize()?;
        let body_bytes = fs::read(&body_path)?;
        let body_sha256 = sha256_bytes(&body_bytes);
        if body_sha256 != string_field(body_record, "sha256")? {
            return Err("expert body bytes do not match the body receipt hash".into());
        }
        let code_bytes = usize_field(body_record, "code_bytes")?;
        let scale_bytes = usize_field(body_record, "scale_bytes")?;
        let expected_code_bytes = rows * (columns / 2);
        let expected_scale_bytes = rows * (columns / GROUP_SIZE) * 2;
        if code_bytes != expected_code_bytes
            || scale_bytes != expected_scale_bytes
            || body_bytes.len() != code_bytes + scale_bytes
        {
            return Err("expert body bytes do not match the Q4/G64 window".into());
        }
        Ok((
            PackedBody {
                path: body_path,
                receipt_sha256,
                body_sha256,
                expert_index,
                row_start,
                rows,
                columns,
                code_bytes,
                scale_bytes,
                codes: body_bytes[..code_bytes].to_vec(),
                scales: body_bytes[code_bytes..].to_vec(),
            },
            receipt,
        ))
    }

    fn validate_body(path: &Path) -> Result<(PackedBody, Value), Box<dyn Error>> {
        validate_body_for(
            path,
            TENSOR_NAME,
            [512, GATE_UP_BODY_ROWS, 2560],
            GATE_UP_BODY_ROWS,
        )
    }

    fn validate_down_body(path: &Path) -> Result<(PackedBody, Value), Box<dyn Error>> {
        validate_body_for(
            path,
            DOWN_TENSOR_NAME,
            [512, DOWN_BODY_ROWS, DOWN_COLUMNS],
            DOWN_BODY_ROWS,
        )
    }

    fn split_gate_up(body: &PackedBody) -> Result<(PackedBody, PackedBody), Box<dyn Error>> {
        if body.row_start != 0
            || body.rows != GATE_UP_BODY_ROWS
            || GATE_UP_BODY_ROWS != GATE_UP_ROWS * 2
        {
            return Err("gate/up activation requires a full 1280-row fused body".into());
        }
        let row_code_bytes = body.columns / 2;
        let row_scale_bytes = (body.columns / GROUP_SIZE) * 2;
        let half_code_bytes = GATE_UP_ROWS * row_code_bytes;
        let half_scale_bytes = GATE_UP_ROWS * row_scale_bytes;
        if body.code_bytes != GATE_UP_BODY_ROWS * row_code_bytes
            || body.scale_bytes != GATE_UP_BODY_ROWS * row_scale_bytes
            || body.codes.len() != body.code_bytes
            || body.scales.len() != body.scale_bytes
        {
            return Err("full gate/up body has an invalid packed layout".into());
        }
        let make_half = |codes: Vec<u8>, scales: Vec<u8>| PackedBody {
            path: body.path.clone(),
            receipt_sha256: body.receipt_sha256.clone(),
            body_sha256: body.body_sha256.clone(),
            expert_index: body.expert_index,
            row_start: 0,
            rows: GATE_UP_ROWS,
            columns: body.columns,
            code_bytes: half_code_bytes,
            scale_bytes: half_scale_bytes,
            codes,
            scales,
        };
        Ok((
            make_half(
                body.codes[..half_code_bytes].to_vec(),
                body.scales[..half_scale_bytes].to_vec(),
            ),
            make_half(
                body.codes[half_code_bytes..].to_vec(),
                body.scales[half_scale_bytes..].to_vec(),
            ),
        ))
    }

    fn validate_kernel_for(
        path: &Path,
        body: &PackedBody,
        expected_tensor: &str,
    ) -> Result<(Value, String), Box<dyn Error>> {
        let (receipt, digest) = read_json(path)?;
        if string_field(&receipt, "schema")? != KERNEL_SCHEMA
            || string_field(&receipt, "status")? != "PASSED"
            || string_field(&receipt, "repo")? != REPO_ID
            || string_field(&receipt, "pinned_revision")? != PINNED_REVISION
            || string_field(&receipt, "nomenclature_version")? != NOMENCLATURE_VERSION
        {
            return Err("expert kernel receipt is not a PASSED pinned parity receipt".into());
        }
        if receipt.get("body_mutated").and_then(Value::as_bool) != Some(false)
            || receipt.get("model_loaded").and_then(Value::as_bool) != Some(false)
        {
            return Err("expert kernel receipt fails mutation/model-load guards".into());
        }
        let source = receipt
            .get("source_tensor")
            .ok_or("expert kernel receipt has no source_tensor")?;
        if string_field(source, "tensor_name")? != expected_tensor
            || usize_field(source, "selected_expert")? != body.expert_index
            || usize_field(source, "selected_row_start")? != body.row_start
            || usize_field(source, "selected_row_count")? != body.rows
            || usize_field(source, "selected_block_bytes")? != body.rows * body.columns * 2
        {
            return Err(format!(
                "expert kernel source window does not match the {expected_tensor} body receipt"
            )
            .into());
        }
        let loader = receipt
            .get("native_loader")
            .ok_or("expert kernel receipt has no native_loader")?;
        if string_field(loader, "status")? != "BOUNDED_NOETIC_DESCRIPTOR_AND_BODY_LOAD"
            || loader
                .get("source_independent_execution")
                .and_then(Value::as_bool)
                != Some(true)
            || loader
                .get("candidate_body_persisted")
                .and_then(Value::as_bool)
                != Some(true)
        {
            return Err("expert kernel receipt does not prove persisted-body execution".into());
        }
        let native_kernel = receipt
            .get("native_kernel")
            .ok_or("expert kernel receipt has no native_kernel")?;
        if native_kernel
            .get("kernel_registered")
            .and_then(Value::as_bool)
            != Some(true)
            || string_field(native_kernel, "kernel")? != KERNEL_NAME
        {
            return Err("expert kernel receipt does not identify the expected Metal kernel".into());
        }
        if receipt
            .get("parity")
            .and_then(|value| value.get("within_tolerance"))
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err("expert kernel parity receipt is not within tolerance".into());
        }
        let candidate = receipt
            .get("candidate_body")
            .ok_or("expert kernel receipt has no candidate_body")?;
        if Path::new(string_field(candidate, "path")?).canonicalize()? != body.path
            || string_field(candidate, "sha256")? != body.body_sha256
            || usize_field(candidate, "bytes")? != body.code_bytes + body.scale_bytes
        {
            return Err("expert kernel candidate body does not match the persisted body".into());
        }
        Ok((receipt, digest))
    }

    fn validate_kernel(path: &Path, body: &PackedBody) -> Result<(Value, String), Box<dyn Error>> {
        validate_kernel_for(path, body, TENSOR_NAME)
    }

    fn validate_down_kernel(
        path: &Path,
        body: &PackedBody,
    ) -> Result<(Value, String), Box<dyn Error>> {
        validate_kernel_for(path, body, DOWN_TENSOR_NAME)
    }

    fn f32_bytes_for_hash(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn deterministic_input(columns: usize) -> Vec<f32> {
        (0..columns)
            .map(|index| {
                ((index * REFERENCE_MULTIPLIER % REFERENCE_MODULUS) as f32 - REFERENCE_OFFSET)
                    / REFERENCE_MODULUS as f32
            })
            .collect()
    }

    fn cpu_matvec(body: &PackedBody, input: &[f32]) -> Vec<f32> {
        let groups_per_row = body.columns / GROUP_SIZE;
        let mut output = vec![0.0f32; body.rows];
        for row in 0..body.rows {
            for group in 0..groups_per_row {
                let group_base = row * groups_per_row + group;
                let scale_offset = group_base * 2;
                let scale = f16::from_bits(u16::from_le_bytes([
                    body.scales[scale_offset],
                    body.scales[scale_offset + 1],
                ]))
                .to_f32();
                let code_base = group_base * CODE_BYTES_PER_GROUP;
                for local in 0..GROUP_SIZE {
                    let byte = body.codes[code_base + local / 2];
                    let nibble = if local % 2 == 0 {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    };
                    output[row] +=
                        (nibble as i32 - 8) as f32 * scale * input[group * GROUP_SIZE + local];
                }
            }
        }
        output
    }

    fn read_f32(buffer: &Buffer, count: usize) -> Vec<f32> {
        let pointer = buffer.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(pointer, count) }.to_vec()
    }

    fn dispatch(
        context: &MetalContext,
        body: &PackedBody,
        codes: &Buffer,
        scales: &Buffer,
        input: &Buffer,
        output: &Buffer,
    ) -> Result<MetalDispatchTiming, Box<dyn Error>> {
        let rows = body.rows as u32;
        let columns = body.columns as u32;
        let groups = (body.columns / GROUP_SIZE) as u32;
        Ok(
            context.dispatch_threads_timed(KERNEL_NAME, (rows, 1, 1), (1, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(codes), 0);
                encoder.set_buffer(1, Some(scales), 0);
                encoder.set_buffer(2, Some(input), 0);
                encoder.set_buffer(3, Some(output), 0);
                encoder.set_bytes(4, 4, &rows as *const u32 as *const _);
                encoder.set_bytes(5, 4, &columns as *const u32 as *const _);
                encoder.set_bytes(6, 4, &groups as *const u32 as *const _);
            })?,
        )
    }

    fn dispatch_gate_up_swiglu(
        context: &MetalContext,
        gate: &PackedBody,
        gate_codes: &Buffer,
        gate_scales: &Buffer,
        up_codes: &Buffer,
        up_scales: &Buffer,
        input: &Buffer,
        output: &Buffer,
    ) -> Result<MetalDispatchTiming, Box<dyn Error>> {
        let rows = gate.rows as u32;
        let columns = gate.columns as u32;
        let groups = (gate.columns / GROUP_SIZE) as u32;
        let threadgroup = 128u32;
        let grid = rows
            .div_ceil(2)
            .saturating_mul(threadgroup)
            .max(threadgroup);
        Ok(context.dispatch_threads_timed(
            GATE_UP_KERNEL_NAME,
            (grid, 1, 1),
            (threadgroup, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(gate_codes), 0);
                encoder.set_buffer(1, Some(gate_scales), 0);
                encoder.set_buffer(2, Some(up_codes), 0);
                encoder.set_buffer(3, Some(up_scales), 0);
                encoder.set_buffer(4, Some(input), 0);
                encoder.set_buffer(5, Some(output), 0);
                encoder.set_bytes(6, 4, &rows as *const u32 as *const _);
                encoder.set_bytes(7, 4, &columns as *const u32 as *const _);
                encoder.set_bytes(8, 4, &groups as *const u32 as *const _);
            },
        )?)
    }

    fn cpu_gate_up_swiglu(gate: &PackedBody, up: &PackedBody, input: &[f32]) -> Vec<f32> {
        let gate_values = cpu_matvec(gate, input);
        let up_values = cpu_matvec(up, input);
        gate_values
            .into_iter()
            .zip(up_values)
            .map(|(gate, up)| (gate / (1.0 + (-gate).exp())) * up)
            .collect()
    }

    fn gpu_ns(timing: MetalDispatchTiming) -> Result<u64, Box<dyn Error>> {
        match (timing.gpu_start_ns, timing.gpu_end_ns) {
            (Some(start), Some(end)) if end > start => Ok(end - start),
            _ => Err("Metal completed without a usable GPU timestamp".into()),
        }
    }

    fn output_metrics(reference: &[f32], observed: &[f32]) -> Value {
        let mut max_abs = 0.0f32;
        let mut squared = 0.0f64;
        let mut left_norm = 0.0f64;
        let mut right_norm = 0.0f64;
        let mut dot = 0.0f64;
        for (left, right) in reference.iter().zip(observed) {
            let error = *left - *right;
            max_abs = max_abs.max(error.abs());
            squared += f64::from(error) * f64::from(error);
            left_norm += f64::from(*left) * f64::from(*left);
            right_norm += f64::from(*right) * f64::from(*right);
            dot += f64::from(*left) * f64::from(*right);
        }
        let count = reference.len();
        json!({
            "count": count,
            "max_abs_error": max_abs,
            "rmse": (squared / count as f64).sqrt(),
            "cosine": if left_norm > 0.0 && right_norm > 0.0 { Some(dot / (left_norm.sqrt() * right_norm.sqrt())) } else { None },
            "finite": observed.iter().all(|value| value.is_finite()),
            "tolerance": OUTPUT_ERROR_TOLERANCE,
            "within_tolerance": max_abs <= OUTPUT_ERROR_TOLERANCE,
        })
    }

    fn percentile_median(values: &[u64]) -> u64 {
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        sorted[sorted.len() / 2]
    }

    fn component_ref(path: &Path, value: &Value, sha256: Option<&str>) -> Value {
        json!({
            "path": path,
            "sha256": sha256,
            "schema": value.get("schema"),
            "status": value.get("status"),
            "label": "[V]",
        })
    }

    fn write_atomic(path: &Path, value: &Value) -> Result<(), Box<dyn Error>> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension(format!("json.tmp-{}", std::process::id()));
        fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
        fs::rename(temporary, path)?;
        Ok(())
    }

    fn run(args: &Args) -> Result<Value, Box<dyn Error>> {
        let started = Instant::now();
        let root = args.root.canonicalize()?;
        let manifest = validate_manifest(&root)?;
        let (router, router_sha256, selected_ids, selected_weights) =
            validate_router(&args.router_receipt)?;
        let (campaign, campaign_sha256) = validate_campaign(&args.campaign_receipt)?;
        let components = campaign
            .get("components")
            .and_then(Value::as_array)
            .ok_or("routed-expert campaign has no components")?;

        let mut selected_specs = Vec::with_capacity(selected_ids.len());
        for expert_id in &selected_ids {
            let component = components
                .iter()
                .find(|component| {
                    component
                        .get("window")
                        .and_then(|window| window.get("expert_index"))
                        .and_then(Value::as_u64)
                        == Some(*expert_id as u64)
                })
                .ok_or_else(|| {
                    format!("campaign has no persisted body for selected expert {expert_id}")
                })?;
            let body_path = component
                .get("body_receipt")
                .and_then(|value| value.get("path"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    format!("campaign component for expert {expert_id} has no body receipt")
                })?;
            let kernel_path = component
                .get("kernel_receipt")
                .and_then(|value| value.get("path"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    format!("campaign component for expert {expert_id} has no kernel receipt")
                })?;
            let (body, body_receipt) = validate_body(Path::new(body_path))?;
            let (kernel, kernel_sha256) = validate_kernel(Path::new(kernel_path), &body)?;
            if body.expert_index != *expert_id {
                return Err(format!(
                    "selected expert {expert_id} does not match body expert {}",
                    body.expert_index
                )
                .into());
            }
            if let Some(input) = kernel.get("input") {
                let expected_input = sha256_bytes(&f32_bytes(&deterministic_input(body.columns)));
                if input.get("deterministic_sha256").and_then(Value::as_str)
                    != Some(expected_input.as_str())
                {
                    return Err(format!(
                        "expert {expert_id} kernel input does not match deterministic route input"
                    )
                    .into());
                }
            }
            selected_specs.push((
                body,
                body_receipt,
                PathBuf::from(body_path),
                kernel,
                PathBuf::from(kernel_path),
                kernel_sha256,
            ));
        }

        let columns = selected_specs
            .first()
            .map(|(body, _, _, _, _, _)| body.columns)
            .ok_or("no selected routed experts")?;
        let rows = selected_specs
            .first()
            .map(|(body, _, _, _, _, _)| body.rows)
            .ok_or("no selected routed experts")?;
        if selected_specs
            .iter()
            .any(|(body, _, _, _, _, _)| body.columns != columns || body.rows != rows)
        {
            return Err("selected expert body windows have different shapes".into());
        }
        let input = deterministic_input(columns);
        let input_sha256 = sha256_bytes(&f32_bytes(&input));
        let input_buffer_bytes = f32_bytes(&input);
        let context = MetalContext::new_with_trace(true)?;
        let input_buffer = context.new_buffer_with_bytes_checked(&input_buffer_bytes)?;
        let mut loaded = Vec::with_capacity(selected_specs.len());
        for (body, body_receipt, body_receipt_path, kernel, kernel_receipt_path, kernel_sha256) in
            selected_specs
        {
            let expected = cpu_matvec(&body, &input);
            loaded.push(LoadedBody {
                codes: context.new_buffer_with_bytes_checked(&body.codes)?,
                scales: context.new_buffer_with_bytes_checked(&body.scales)?,
                output: context.new_buffer_checked(body.rows * std::mem::size_of::<f32>())?,
                body,
                body_receipt,
                body_receipt_path,
                kernel_receipt: kernel,
                kernel_receipt_path,
                kernel_receipt_sha256: kernel_sha256,
                expected,
            });
        }

        let mut warmup_route_gpu_ns = Vec::with_capacity(args.warmup);
        for _ in 0..args.warmup {
            let mut total = 0u64;
            for item in &loaded {
                total = total.saturating_add(gpu_ns(dispatch(
                    &context,
                    &item.body,
                    &item.codes,
                    &item.scales,
                    &input_buffer,
                    &item.output,
                )?)?);
            }
            warmup_route_gpu_ns.push(total);
        }

        let mut per_body_gpu_ns = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut per_body_host_ns = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut per_body_output_hashes = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut route_gpu_ns = Vec::with_capacity(args.reps);
        let mut route_host_ns = Vec::with_capacity(args.reps);
        let mut mixed_hashes = Vec::with_capacity(args.reps);
        let mut last_outputs: Vec<Vec<f32>> = vec![Vec::new(); loaded.len()];
        for _ in 0..args.reps {
            let mut total_gpu = 0u64;
            let mut total_host = 0u64;
            let mut mixed = vec![0.0f32; loaded[0].body.rows];
            for (index, item) in loaded.iter().enumerate() {
                let timing = dispatch(
                    &context,
                    &item.body,
                    &item.codes,
                    &item.scales,
                    &input_buffer,
                    &item.output,
                )?;
                let observed = read_f32(&item.output, item.body.rows);
                if !observed.iter().all(|value| value.is_finite()) {
                    return Err(format!(
                        "native routed body {} output is non-finite",
                        item.body.expert_index
                    )
                    .into());
                }
                let parity = output_metrics(&item.expected, &observed);
                if parity.get("within_tolerance").and_then(Value::as_bool) != Some(true) {
                    return Err(format!(
                        "native routed body {} parity failed: {parity}",
                        item.body.expert_index
                    )
                    .into());
                }
                let gpu = gpu_ns(timing)?;
                let host = timing.host_wall_us.saturating_mul(1000);
                total_gpu = total_gpu.saturating_add(gpu);
                total_host = total_host.saturating_add(host);
                per_body_gpu_ns[index].push(gpu);
                per_body_host_ns[index].push(host);
                per_body_output_hashes[index].push(sha256_bytes(&f32_bytes_for_hash(&observed)));
                let weight = selected_weights[index];
                for (slot, value) in observed.iter().enumerate() {
                    mixed[slot] += weight * value;
                }
                last_outputs[index] = observed;
            }
            route_gpu_ns.push(total_gpu);
            route_host_ns.push(total_host);
            mixed_hashes.push(sha256_bytes(&f32_bytes_for_hash(&mixed)));
        }

        if per_body_output_hashes
            .iter()
            .any(|hashes| hashes.windows(2).any(|pair| pair[0] != pair[1]))
            || mixed_hashes.windows(2).any(|pair| pair[0] != pair[1])
        {
            return Err("native routed body outputs changed across repeated executions".into());
        }
        let mut expected_mixed = vec![0.0f32; loaded[0].body.rows];
        for (index, item) in loaded.iter().enumerate() {
            for (slot, value) in item.expected.iter().enumerate() {
                expected_mixed[slot] += selected_weights[index] * value;
            }
        }
        let observed_mixed = loaded.iter().enumerate().fold(
            vec![0.0f32; loaded[0].body.rows],
            |mut mixed, (index, _item)| {
                for (slot, value) in last_outputs[index].iter().enumerate() {
                    mixed[slot] += selected_weights[index] * value;
                }
                mixed
            },
        );
        let mixed_parity = output_metrics(&expected_mixed, &observed_mixed);
        if mixed_parity
            .get("within_tolerance")
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err(format!("weighted routed gather parity failed: {mixed_parity}").into());
        }

        let source_selection_parity = router
            .get("source_selection_parity")
            .cloned()
            .unwrap_or_else(|| json!({"status": "UNKNOWN"}));
        let source_mismatch = source_selection_parity
            .get("expert_ids_exact_match")
            .and_then(Value::as_bool)
            == Some(false);
        let mut physical_graph = json!({
            "schema": "hcli.physical_graph.v1",
            "semantic_type": "PhysicalGraph",
            "compiler_stage": "HawkingAccelerator",
            "component_scope": "native router receipt -> selected persisted routed-expert body windows -> Q4/G64 Metal matvecs -> host weighted gather; no complete expert/token graph",
            "device_placement": {"selected": "apple_metal", "candidates": ["apple_metal", "cpu"]},
            "native_kernel_execution_observed": true,
            "selected_expert_count": loaded.len(),
            "dispatches_per_route": loaded.len(),
            "source_selection_parity_status": source_selection_parity.get("status"),
            "source_selection_parity_qualified": source_selection_parity.get("expert_ids_exact_match"),
            "source_selection_mismatch_accepted_as_bounded_boundary": source_mismatch,
            "promotion_allowed": false,
        });
        let physical_graph_fingerprint = sha256_bytes(&serde_json::to_vec(&physical_graph)?);
        physical_graph["fingerprint"] = Value::String(physical_graph_fingerprint);

        let device_memory = context.device_memory_limits();
        let mut components_json = Vec::with_capacity(loaded.len());
        for (index, item) in loaded.iter().enumerate() {
            components_json.push(json!({
                "expert_index": item.body.expert_index,
                "row_start": item.body.row_start,
                "row_count": item.body.rows,
                "body_receipt": component_ref(&item.body_receipt_path, &item.body_receipt, Some(&item.body.receipt_sha256)),
                "kernel_receipt": component_ref(
                    &item.kernel_receipt_path,
                    &item.kernel_receipt,
                    Some(&item.kernel_receipt_sha256),
                ),
                "candidate_body": {
                    "path": item.body.path,
                    "sha256": item.body.body_sha256,
                    "bytes": item.body.code_bytes + item.body.scale_bytes,
                    "source_independent": true,
                    "candidate_body_persisted": true,
                    "label": "[D]",
                },
                "native_dispatch": {
                    "kernel": KERNEL_NAME,
                    "dispatches_per_route": 1,
                    "gpu_ns": per_body_gpu_ns[index],
                    "gpu_ns_median": percentile_median(&per_body_gpu_ns[index]),
                    "host_wall_ns": per_body_host_ns[index],
                    "host_wall_ns_median": percentile_median(&per_body_host_ns[index]),
                    "output_hashes": per_body_output_hashes[index],
                    "stable_output": true,
                },
                "parity": output_metrics(&item.expected, &last_outputs[index]),
                "source_reference_used_for_execution": false,
            }));
        }

        let router_selection = router
            .get("selection")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let route_gpu_median = percentile_median(&route_gpu_ns);
        let route_host_median = percentile_median(&route_host_ns);
        Ok(json!({
            "schema": SCHEMA,
            "nomenclature_version": NOMENCLATURE_VERSION,
            "semantic_type": "NoeticExecutableCandidate",
            "compiler_stage": "HawkingAccelerator",
            "status": "PASSED",
            "qualification": "BOUNDED_NATIVE_ROUTED_EXPERT_BODY_DISPATCH",
            "repo": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "root": root,
            "model_lake_manifest": manifest,
            "router_receipt": {
                "path": args.router_receipt.canonicalize()?,
                "sha256": router_sha256,
                "schema": ROUTER_SCHEMA,
                "status": "PASSED",
                "label": "[V]",
            },
            "campaign_receipt": {
                "path": args.campaign_receipt.canonicalize()?,
                "sha256": campaign_sha256,
                "schema": CAMPAIGN_SCHEMA,
                "status": "PASSED",
                "label": "[V]",
            },
            "selection": router_selection,
            "source_selection_parity": source_selection_parity,
            "components": components_json,
            "execution": {
                "provider": "apple-metal",
                "operation": "native_router_selection_receipt -> selected_persisted_routed_expert_body_load -> Q4_G64_matvec_per_selected_body -> host_selected_weight_gather",
                "selected_expert_ids": selected_ids,
                "selected_expert_count": loaded.len(),
                "dispatches_per_route": loaded.len(),
                "measured_routes": args.reps,
                "total_measured_dispatches": loaded.len() * args.reps,
                "native_routed_body_dispatch_observed": true,
                "source_reference_used_for_execution": false,
                "source_selection_mismatch_accepted_as_bounded_boundary": source_mismatch,
                "body_mutated": false,
                "model_loaded": false,
                "complete_expert_activation": false,
                "complete_token_runtime": false,
            },
            "input": {
                "definition": "((index * 71) mod 509 - 254) / 509",
                "values": columns,
                "deterministic_sha256": input_sha256,
                "label": "[V]",
            },
            "gpu_timing": {
                "device": context.device_name(),
                "warmup_runs": args.warmup,
                "measured_runs": args.reps,
                "warmup_route_gpu_ns": warmup_route_gpu_ns,
                "route_gpu_ns": route_gpu_ns,
                "route_gpu_ns_median": route_gpu_median,
                "route_host_wall_ns": route_host_ns,
                "route_host_wall_ns_median": route_host_median,
                "dispatches_per_route": loaded.len(),
                "output_hashes": mixed_hashes,
                "memory_limits": {
                    "max_buffer_length": device_memory.max_buffer_length,
                    "recommended_max_working_set_size": device_memory.recommended_max_working_set_size,
                    "current_allocated_size": device_memory.current_allocated_size,
                    "has_unified_memory": device_memory.has_unified_memory,
                },
                "timing_authority": "Metal completed-command-buffer GPUStartTime/GPUEndTime summed across selected body dispatches; host wall reported separately",
            },
            "gather": {
                "status": "BOUNDED_HOST_WEIGHTED_GATHER",
                "weights": selected_weights,
                "output_rows": loaded[0].body.rows,
                "output_sha256": mixed_hashes.last(),
                "parity": mixed_parity,
                "fused_native_gather": false,
            },
            "noetic_ir": {
                "schema": "hcli.noetic.ir.v1",
                "semantic_type": "NoeticIR",
                "operations": [
                    "consume_native_router_selection",
                    "resolve_selected_persisted_routed_expert_body_windows",
                    "load_source_independent_q4_g64_body_windows",
                    "execute_native_q4_g64_matvec_per_selected_body",
                    "apply_selected_router_weights_on_host",
                    "emit_bounded_routed_body_outputs",
                ],
                "source_independent": true,
                "complete_expert": false,
                "complete_model": false,
            },
            "physical_graph": physical_graph,
            "whole_model_capability": "NOT_TESTED",
            "complete_expert_runtime": "NOT_TESTED",
            "complete_token_runtime": "NOT_TESTED",
            "complete_system_ebpw": null,
            "flash_tps": null,
            "body_mutated": false,
            "model_loaded": false,
            "native_routed_body_dispatch_observed": true,
            "promotion_allowed": false,
            "claim_boundary": "PASSED bounded native source-independent dispatch of the selected persisted routed-expert Q4/G64 body windows with a host-side selected-weight gather. This is not full expert activation, fused route/gather, complete-model loading, complete-token runtime, Flash TPS, or EBPW evidence; source-selection mismatch remains explicit.",
            "next_action": "extend from bounded routed body windows to independently validated gate/up activation and native expert composition; do not measure or claim complete-token Flash TPS/EBPW until the full protected graph is capability-qualified",
            "elapsed_s": started.elapsed().as_secs_f64(),
        }))
    }

    fn run_gate_up_swiglu(args: &Args) -> Result<Value, Box<dyn Error>> {
        let started = Instant::now();
        let repo = repository_root();
        let root = args.root.canonicalize()?;
        let manifest = validate_manifest(&root)?;
        let (router, router_sha256, selected_ids, selected_weights) =
            validate_router(&args.router_receipt)?;
        let mut selected_specs = Vec::with_capacity(selected_ids.len());
        for expert_id in &selected_ids {
            let body_path = repo.join(format!(
                "receipts/headless/FLASH_NOETIC_GATE_UP_BODY_E{expert_id}_R0_1280.json"
            ));
            let kernel_path = repo.join(format!(
                "receipts/headless/FLASH_NOETIC_GATE_UP_KERNEL_E{expert_id}_R0_1280_PARITY.json"
            ));
            let (body, body_receipt) = validate_body(&body_path)?;
            if body.expert_index != *expert_id
                || body.row_start != 0
                || body.rows != GATE_UP_BODY_ROWS
            {
                return Err(format!(
                    "gate/up body receipt for expert {expert_id} is not the full 1280-row body"
                )
                .into());
            }
            let (kernel, kernel_sha256) = validate_kernel(&kernel_path, &body)?;
            let expected_input = sha256_bytes(&f32_bytes(&deterministic_input(body.columns)));
            if kernel
                .get("input")
                .and_then(|input| input.get("deterministic_sha256"))
                .and_then(Value::as_str)
                != Some(expected_input.as_str())
            {
                return Err(format!(
                    "gate/up kernel input for expert {expert_id} does not match the deterministic input"
                )
                .into());
            }
            selected_specs.push((
                body,
                body_receipt,
                body_path,
                kernel,
                kernel_path,
                kernel_sha256,
            ));
        }
        let columns = selected_specs
            .first()
            .map(|(body, _, _, _, _, _)| body.columns)
            .ok_or("no selected gate/up experts")?;
        if selected_specs.iter().any(|(body, _, _, _, _, _)| {
            body.columns != columns || body.row_start != 0 || body.rows != GATE_UP_BODY_ROWS
        }) {
            return Err("selected gate/up bodies have inconsistent full-body shapes".into());
        }

        let input = deterministic_input(columns);
        let input_sha256 = sha256_bytes(&f32_bytes(&input));
        let context = MetalContext::new_with_trace(true)?;
        let input_buffer = context.new_buffer_with_bytes_checked(&f32_bytes(&input))?;
        let mut loaded = Vec::with_capacity(selected_specs.len());
        for (body, body_receipt, body_receipt_path, kernel, kernel_receipt_path, kernel_sha256) in
            selected_specs
        {
            let (gate, up) = split_gate_up(&body)?;
            let expected = cpu_gate_up_swiglu(&gate, &up, &input);
            loaded.push(LoadedGateUp {
                gate_codes: context.new_buffer_with_bytes_checked(&gate.codes)?,
                gate_scales: context.new_buffer_with_bytes_checked(&gate.scales)?,
                up_codes: context.new_buffer_with_bytes_checked(&up.codes)?,
                up_scales: context.new_buffer_with_bytes_checked(&up.scales)?,
                output: context.new_buffer_checked(gate.rows * std::mem::size_of::<f32>())?,
                body,
                gate,
                up,
                body_receipt,
                body_receipt_path,
                kernel_receipt: kernel,
                kernel_receipt_path,
                kernel_receipt_sha256: kernel_sha256,
                expected,
            });
        }

        let mut warmup_route_gpu_ns = Vec::with_capacity(args.warmup);
        for _ in 0..args.warmup {
            let mut total = 0u64;
            for item in &loaded {
                total = total.saturating_add(gpu_ns(dispatch_gate_up_swiglu(
                    &context,
                    &item.gate,
                    &item.gate_codes,
                    &item.gate_scales,
                    &item.up_codes,
                    &item.up_scales,
                    &input_buffer,
                    &item.output,
                )?)?);
            }
            warmup_route_gpu_ns.push(total);
        }

        let mut per_expert_gpu_ns = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut per_expert_host_ns = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut per_expert_output_hashes = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut route_gpu_ns = Vec::with_capacity(args.reps);
        let mut route_host_ns = Vec::with_capacity(args.reps);
        let mut mixed_hashes = Vec::with_capacity(args.reps);
        let mut last_outputs: Vec<Vec<f32>> = vec![Vec::new(); loaded.len()];
        for _ in 0..args.reps {
            let mut total_gpu = 0u64;
            let mut total_host = 0u64;
            let mut mixed = vec![0.0f32; GATE_UP_ROWS];
            for (index, item) in loaded.iter().enumerate() {
                let timing = dispatch_gate_up_swiglu(
                    &context,
                    &item.gate,
                    &item.gate_codes,
                    &item.gate_scales,
                    &item.up_codes,
                    &item.up_scales,
                    &input_buffer,
                    &item.output,
                )?;
                let observed = read_f32(&item.output, GATE_UP_ROWS);
                if !observed.iter().all(|value| value.is_finite()) {
                    return Err(format!(
                        "native gate/up SwiGLU output for expert {} is non-finite",
                        item.body.expert_index
                    )
                    .into());
                }
                let parity = output_metrics(&item.expected, &observed);
                if parity.get("within_tolerance").and_then(Value::as_bool) != Some(true) {
                    return Err(format!(
                        "native gate/up SwiGLU parity failed for expert {}: {parity}",
                        item.body.expert_index
                    )
                    .into());
                }
                let gpu = gpu_ns(timing)?;
                let host = timing.host_wall_us.saturating_mul(1000);
                total_gpu = total_gpu.saturating_add(gpu);
                total_host = total_host.saturating_add(host);
                per_expert_gpu_ns[index].push(gpu);
                per_expert_host_ns[index].push(host);
                per_expert_output_hashes[index].push(sha256_bytes(&f32_bytes_for_hash(&observed)));
                for (slot, value) in observed.iter().enumerate() {
                    mixed[slot] += selected_weights[index] * value;
                }
                last_outputs[index] = observed;
            }
            route_gpu_ns.push(total_gpu);
            route_host_ns.push(total_host);
            mixed_hashes.push(sha256_bytes(&f32_bytes_for_hash(&mixed)));
        }
        if per_expert_output_hashes
            .iter()
            .any(|hashes| hashes.windows(2).any(|pair| pair[0] != pair[1]))
            || mixed_hashes.windows(2).any(|pair| pair[0] != pair[1])
        {
            return Err("native gate/up SwiGLU outputs changed across repeated executions".into());
        }
        let mut expected_mixed = vec![0.0f32; GATE_UP_ROWS];
        let mut observed_mixed = vec![0.0f32; GATE_UP_ROWS];
        for (index, item) in loaded.iter().enumerate() {
            for (slot, value) in item.expected.iter().enumerate() {
                expected_mixed[slot] += selected_weights[index] * value;
            }
            for (slot, value) in last_outputs[index].iter().enumerate() {
                observed_mixed[slot] += selected_weights[index] * value;
            }
        }
        let mixed_parity = output_metrics(&expected_mixed, &observed_mixed);
        if mixed_parity
            .get("within_tolerance")
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err(format!("weighted gate/up gather parity failed: {mixed_parity}").into());
        }

        let source_selection_parity = router
            .get("source_selection_parity")
            .cloned()
            .unwrap_or_else(|| json!({"status": "UNKNOWN"}));
        let source_mismatch = source_selection_parity
            .get("expert_ids_exact_match")
            .and_then(Value::as_bool)
            == Some(false);
        let mut physical_graph = json!({
            "schema": "hcli.physical_graph.v1",
            "semantic_type": "PhysicalGraph",
            "compiler_stage": "HawkingAccelerator",
            "component_scope": "native router receipt -> selected full persisted gate_up bodies -> split gate/up Q4/G64 buffers -> native Metal gate_up_swiglu -> host weighted gather; down projection and complete token graph excluded",
            "device_placement": {"selected": "apple_metal", "candidates": ["apple_metal", "cpu"]},
            "native_kernel": GATE_UP_KERNEL_NAME,
            "native_kernel_execution_observed": true,
            "selected_expert_count": loaded.len(),
            "dispatches_per_route": loaded.len(),
            "gate_rows": GATE_UP_ROWS,
            "full_body_rows": GATE_UP_BODY_ROWS,
            "source_selection_parity_status": source_selection_parity.get("status"),
            "source_selection_parity_qualified": source_selection_parity.get("expert_ids_exact_match"),
            "source_selection_mismatch_accepted_as_bounded_boundary": source_mismatch,
            "promotion_allowed": false,
        });
        let physical_graph_fingerprint = sha256_bytes(&serde_json::to_vec(&physical_graph)?);
        physical_graph["fingerprint"] = Value::String(physical_graph_fingerprint);

        let device_memory = context.device_memory_limits();
        let mut components_json = Vec::with_capacity(loaded.len());
        for (index, item) in loaded.iter().enumerate() {
            components_json.push(json!({
                "expert_index": item.body.expert_index,
                "body_rows": item.body.rows,
                "gate_rows": item.gate.rows,
                "up_rows": item.up.rows,
                "body_receipt": component_ref(&item.body_receipt_path, &item.body_receipt, Some(&item.body.receipt_sha256)),
                "kernel_receipt": component_ref(&item.kernel_receipt_path, &item.kernel_receipt, Some(&item.kernel_receipt_sha256)),
                "candidate_body": {
                    "path": item.body.path,
                    "sha256": item.body.body_sha256,
                    "bytes": item.body.code_bytes + item.body.scale_bytes,
                    "source_independent": true,
                    "candidate_body_persisted": true,
                    "layout": "fused gate_up body split into contiguous gate/up row halves",
                    "label": "[D]",
                },
                "native_dispatch": {
                    "kernel": GATE_UP_KERNEL_NAME,
                    "dispatches_per_route": 1,
                    "gpu_ns": per_expert_gpu_ns[index],
                    "gpu_ns_median": percentile_median(&per_expert_gpu_ns[index]),
                    "host_wall_ns": per_expert_host_ns[index],
                    "host_wall_ns_median": percentile_median(&per_expert_host_ns[index]),
                    "output_hashes": per_expert_output_hashes[index],
                    "stable_output": true,
                },
                "parity": output_metrics(&item.expected, &last_outputs[index]),
                "source_reference_used_for_execution": false,
            }));
        }
        let route_gpu_median = percentile_median(&route_gpu_ns);
        let route_host_median = percentile_median(&route_host_ns);
        Ok(json!({
            "schema": GATE_UP_SCHEMA,
            "nomenclature_version": NOMENCLATURE_VERSION,
            "semantic_type": "NoeticExecutableCandidate",
            "compiler_stage": "HawkingAccelerator",
            "status": "PASSED",
            "qualification": "BOUNDED_NATIVE_ROUTED_EXPERT_GATE_UP_SWIGLU_ACTIVATION",
            "repo": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "root": root,
            "model_lake_manifest": manifest,
            "router_receipt": {
                "path": args.router_receipt.canonicalize()?,
                "sha256": router_sha256,
                "schema": ROUTER_SCHEMA,
                "status": "PASSED",
                "label": "[V]",
            },
            "component_receipt_policy": {
                "directory": repo.join("receipts/headless"),
                "body_pattern": "FLASH_NOETIC_GATE_UP_BODY_E{expert}_R0_1280.json",
                "kernel_pattern": "FLASH_NOETIC_GATE_UP_KERNEL_E{expert}_R0_1280_PARITY.json",
                "body_rows": GATE_UP_BODY_ROWS,
                "gate_rows": GATE_UP_ROWS,
            },
            "selection": router.get("selection").cloned().unwrap_or_else(|| json!({})),
            "source_selection_parity": source_selection_parity,
            "components": components_json,
            "execution": {
                "provider": "apple-metal",
                "operation": "native_router_selection_receipt -> full persisted fused gate_up body load -> gate/up row split -> native Q4_G64 gate_up_swiglu -> host_selected_weight_gather",
                "selected_expert_ids": selected_ids,
                "selected_expert_count": loaded.len(),
                "dispatches_per_route": loaded.len(),
                "measured_routes": args.reps,
                "total_measured_dispatches": loaded.len() * args.reps,
                "native_gate_up_swiglu_observed": true,
                "native_expert_gate_up_activation_observed": true,
                "source_reference_used_for_execution": false,
                "source_selection_mismatch_accepted_as_bounded_boundary": source_mismatch,
                "body_mutated": false,
                "model_loaded": false,
                "complete_expert_activation": false,
                "complete_token_runtime": false,
                "down_projection": "NOT_TESTED",
            },
            "input": {
                "definition": "((index * 71) mod 509 - 254) / 509",
                "values": columns,
                "deterministic_sha256": input_sha256,
                "label": "[V]",
            },
            "gpu_timing": {
                "device": context.device_name(),
                "warmup_runs": args.warmup,
                "measured_runs": args.reps,
                "warmup_route_gpu_ns": warmup_route_gpu_ns,
                "route_gpu_ns": route_gpu_ns,
                "route_gpu_ns_median": route_gpu_median,
                "route_host_wall_ns": route_host_ns,
                "route_host_wall_ns_median": route_host_median,
                "dispatches_per_route": loaded.len(),
                "output_hashes": mixed_hashes,
                "memory_limits": {
                    "max_buffer_length": device_memory.max_buffer_length,
                    "recommended_max_working_set_size": device_memory.recommended_max_working_set_size,
                    "current_allocated_size": device_memory.current_allocated_size,
                    "has_unified_memory": device_memory.has_unified_memory,
                },
                "timing_authority": "Metal completed-command-buffer GPUStartTime/GPUEndTime for native gate_up_swiglu dispatches summed across selected experts; host wall reported separately",
            },
            "gather": {
                "status": "BOUNDED_HOST_WEIGHTED_GATHER",
                "weights": selected_weights,
                "output_rows": GATE_UP_ROWS,
                "output_sha256": mixed_hashes.last(),
                "parity": mixed_parity,
                "fused_native_gather": false,
            },
            "noetic_ir": {
                "schema": "hcli.noetic.ir.v1",
                "semantic_type": "NoeticIR",
                "operations": [
                    "consume_native_router_selection",
                    "resolve_selected_full_persisted_fused_gate_up_bodies",
                    "split_fused_gate_up_body_into_gate_and_up_q4_g64_buffers",
                    "execute_native_q4_g64_gate_up_swiglu_per_selected_expert",
                    "apply_selected_router_weights_on_host",
                    "emit_bounded_gate_up_activation_outputs",
                ],
                "source_independent": true,
                "complete_expert": false,
                "complete_model": false,
            },
            "physical_graph": physical_graph,
            "whole_model_capability": "NOT_TESTED",
            "complete_expert_runtime": "NOT_TESTED",
            "complete_token_runtime": "NOT_TESTED",
            "complete_system_ebpw": null,
            "flash_tps": null,
            "body_mutated": false,
            "model_loaded": false,
            "native_gate_up_swiglu_observed": true,
            "native_expert_gate_up_activation_observed": true,
            "promotion_allowed": false,
            "claim_boundary": "PASSED bounded native source-independent Q4/G64 gate+up/SwiGLU activation for the ten selected persisted expert bodies, with host-side selected-weight gather. Down projection, complete expert activation, complete-model loading, complete-token runtime, Flash TPS, and EBPW remain untested; source-selection mismatch remains explicit.",
            "next_action": "add and validate source-independent down-projection bodies, then compose a bounded native expert output before attempting any protected complete-token Flash graph",
            "elapsed_s": started.elapsed().as_secs_f64(),
        }))
    }

    fn run_expert_composition(args: &Args) -> Result<Value, Box<dyn Error>> {
        let started = Instant::now();
        let repo = repository_root();
        let root = args.root.canonicalize()?;
        let manifest = validate_manifest(&root)?;
        let (router, router_sha256, selected_ids, selected_weights) =
            validate_router(&args.router_receipt)?;
        let input = deterministic_input(2560);
        let input_sha256 = sha256_bytes(&f32_bytes(&input));
        let context = MetalContext::new_with_trace(true)?;
        let input_buffer = context.new_buffer_with_bytes_checked(&f32_bytes(&input))?;
        let mut loaded = Vec::with_capacity(selected_ids.len());

        for expert_id in &selected_ids {
            let gate_up_body_receipt_path = repo.join(format!(
                "receipts/headless/FLASH_NOETIC_GATE_UP_BODY_E{expert_id}_R0_1280.json"
            ));
            let gate_up_kernel_receipt_path = repo.join(format!(
                "receipts/headless/FLASH_NOETIC_GATE_UP_KERNEL_E{expert_id}_R0_1280_PARITY.json"
            ));
            let down_body_receipt_path = repo.join(format!(
                "receipts/headless/FLASH_NOETIC_DOWN_BODY_E{expert_id}_R0_2560.json"
            ));
            let down_kernel_receipt_path = repo.join(format!(
                "receipts/headless/FLASH_NOETIC_DOWN_KERNEL_E{expert_id}_R0_2560_PARITY.json"
            ));

            let (gate_up_body, gate_up_body_receipt) = validate_body(&gate_up_body_receipt_path)?;
            if gate_up_body.expert_index != *expert_id
                || gate_up_body.row_start != 0
                || gate_up_body.rows != GATE_UP_BODY_ROWS
                || gate_up_body.columns != 2560
            {
                return Err(format!(
                    "gate/up body receipt for expert {expert_id} is not the full [1280,2560] body"
                )
                .into());
            }
            let (gate_up_kernel, gate_up_kernel_receipt_sha256) =
                validate_kernel(&gate_up_kernel_receipt_path, &gate_up_body)?;
            let expected_gate_input =
                sha256_bytes(&f32_bytes(&deterministic_input(gate_up_body.columns)));
            if gate_up_kernel
                .get("input")
                .and_then(|input| input.get("deterministic_sha256"))
                .and_then(Value::as_str)
                != Some(expected_gate_input.as_str())
            {
                return Err(format!(
                    "gate/up kernel input for expert {expert_id} does not match the deterministic 2560-value input"
                )
                .into());
            }

            let (down, down_body_receipt) = validate_down_body(&down_body_receipt_path)?;
            if down.expert_index != *expert_id
                || down.row_start != 0
                || down.rows != DOWN_BODY_ROWS
                || down.columns != DOWN_COLUMNS
            {
                return Err(format!(
                    "down body receipt for expert {expert_id} is not the full [2560,640] body"
                )
                .into());
            }
            let (down_kernel, down_kernel_receipt_sha256) =
                validate_down_kernel(&down_kernel_receipt_path, &down)?;
            let expected_down_input = sha256_bytes(&f32_bytes(&deterministic_input(down.columns)));
            if down_kernel
                .get("input")
                .and_then(|input| input.get("deterministic_sha256"))
                .and_then(Value::as_str)
                != Some(expected_down_input.as_str())
            {
                return Err(format!(
                    "down kernel input for expert {expert_id} does not match the deterministic 640-value parity input"
                )
                .into());
            }

            let (gate, up) = split_gate_up(&gate_up_body)?;
            let expected_gate_up = cpu_gate_up_swiglu(&gate, &up, &input);
            let expected_down = cpu_matvec(&down, &expected_gate_up);
            loaded.push(LoadedComposition {
                gate_codes: context.new_buffer_with_bytes_checked(&gate.codes)?,
                gate_scales: context.new_buffer_with_bytes_checked(&gate.scales)?,
                up_codes: context.new_buffer_with_bytes_checked(&up.codes)?,
                up_scales: context.new_buffer_with_bytes_checked(&up.scales)?,
                gate_up_output: context
                    .new_buffer_checked(GATE_UP_ROWS * std::mem::size_of::<f32>())?,
                down_codes: context.new_buffer_with_bytes_checked(&down.codes)?,
                down_scales: context.new_buffer_with_bytes_checked(&down.scales)?,
                down_output: context
                    .new_buffer_checked(DOWN_BODY_ROWS * std::mem::size_of::<f32>())?,
                gate_up_body,
                gate,
                gate_up_body_receipt,
                gate_up_body_receipt_path: gate_up_body_receipt_path.canonicalize()?,
                gate_up_kernel_receipt: gate_up_kernel,
                gate_up_kernel_receipt_path: gate_up_kernel_receipt_path.canonicalize()?,
                gate_up_kernel_receipt_sha256,
                down,
                down_body_receipt,
                down_body_receipt_path: down_body_receipt_path.canonicalize()?,
                down_kernel_receipt: down_kernel,
                down_kernel_receipt_path: down_kernel_receipt_path.canonicalize()?,
                down_kernel_receipt_sha256,
                expected_gate_up,
                expected_down,
            });
        }

        if loaded.is_empty() {
            return Err("no selected experts were available for native composition".into());
        }

        let mut warmup_route_gpu_ns = Vec::with_capacity(args.warmup);
        for _ in 0..args.warmup {
            let mut total = 0u64;
            for item in &loaded {
                total = total.saturating_add(gpu_ns(dispatch_gate_up_swiglu(
                    &context,
                    &item.gate,
                    &item.gate_codes,
                    &item.gate_scales,
                    &item.up_codes,
                    &item.up_scales,
                    &input_buffer,
                    &item.gate_up_output,
                )?)?);
                total = total.saturating_add(gpu_ns(dispatch(
                    &context,
                    &item.down,
                    &item.down_codes,
                    &item.down_scales,
                    &item.gate_up_output,
                    &item.down_output,
                )?)?);
            }
            warmup_route_gpu_ns.push(total);
        }

        let mut per_expert_gate_up_gpu_ns = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut per_expert_down_gpu_ns = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut per_expert_total_gpu_ns = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut per_expert_host_ns = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut per_expert_gate_up_hashes = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut per_expert_down_hashes = vec![Vec::with_capacity(args.reps); loaded.len()];
        let mut route_gpu_ns = Vec::with_capacity(args.reps);
        let mut route_host_ns = Vec::with_capacity(args.reps);
        let mut mixed_hashes = Vec::with_capacity(args.reps);
        let mut last_gate_up_outputs: Vec<Vec<f32>> = vec![Vec::new(); loaded.len()];
        let mut last_outputs: Vec<Vec<f32>> = vec![Vec::new(); loaded.len()];
        for _ in 0..args.reps {
            let mut total_gpu = 0u64;
            let mut total_host = 0u64;
            let mut mixed = vec![0.0f32; DOWN_BODY_ROWS];
            for (index, item) in loaded.iter().enumerate() {
                let gate_up_timing = dispatch_gate_up_swiglu(
                    &context,
                    &item.gate,
                    &item.gate_codes,
                    &item.gate_scales,
                    &item.up_codes,
                    &item.up_scales,
                    &input_buffer,
                    &item.gate_up_output,
                )?;
                let down_timing = dispatch(
                    &context,
                    &item.down,
                    &item.down_codes,
                    &item.down_scales,
                    &item.gate_up_output,
                    &item.down_output,
                )?;
                let observed_gate_up = read_f32(&item.gate_up_output, GATE_UP_ROWS);
                let observed_down = read_f32(&item.down_output, DOWN_BODY_ROWS);
                if !observed_gate_up.iter().all(|value| value.is_finite())
                    || !observed_down.iter().all(|value| value.is_finite())
                {
                    return Err(format!(
                        "native expert composition output for expert {} is non-finite",
                        item.gate_up_body.expert_index
                    )
                    .into());
                }
                let gate_up_parity = output_metrics(&item.expected_gate_up, &observed_gate_up);
                if gate_up_parity
                    .get("within_tolerance")
                    .and_then(Value::as_bool)
                    != Some(true)
                {
                    return Err(format!(
                        "native composition gate/up parity failed for expert {}: {gate_up_parity}",
                        item.gate_up_body.expert_index
                    )
                    .into());
                }
                let down_parity = output_metrics(&item.expected_down, &observed_down);
                if down_parity.get("within_tolerance").and_then(Value::as_bool) != Some(true) {
                    return Err(format!(
                        "native composition down parity failed for expert {}: {down_parity}",
                        item.gate_up_body.expert_index
                    )
                    .into());
                }
                let gate_up_gpu = gpu_ns(gate_up_timing)?;
                let down_gpu = gpu_ns(down_timing)?;
                let total_expert_gpu = gate_up_gpu.saturating_add(down_gpu);
                let host = gate_up_timing
                    .host_wall_us
                    .saturating_add(down_timing.host_wall_us)
                    .saturating_mul(1000);
                total_gpu = total_gpu.saturating_add(total_expert_gpu);
                total_host = total_host.saturating_add(host);
                per_expert_gate_up_gpu_ns[index].push(gate_up_gpu);
                per_expert_down_gpu_ns[index].push(down_gpu);
                per_expert_total_gpu_ns[index].push(total_expert_gpu);
                per_expert_host_ns[index].push(host);
                per_expert_gate_up_hashes[index]
                    .push(sha256_bytes(&f32_bytes_for_hash(&observed_gate_up)));
                per_expert_down_hashes[index]
                    .push(sha256_bytes(&f32_bytes_for_hash(&observed_down)));
                for (slot, value) in observed_down.iter().enumerate() {
                    mixed[slot] += selected_weights[index] * value;
                }
                last_gate_up_outputs[index] = observed_gate_up;
                last_outputs[index] = observed_down;
            }
            route_gpu_ns.push(total_gpu);
            route_host_ns.push(total_host);
            mixed_hashes.push(sha256_bytes(&f32_bytes_for_hash(&mixed)));
        }

        if per_expert_gate_up_hashes
            .iter()
            .chain(per_expert_down_hashes.iter())
            .any(|hashes| hashes.windows(2).any(|pair| pair[0] != pair[1]))
            || mixed_hashes.windows(2).any(|pair| pair[0] != pair[1])
        {
            return Err(
                "native expert composition outputs changed across repeated executions".into(),
            );
        }

        let mut expected_mixed = vec![0.0f32; DOWN_BODY_ROWS];
        let mut observed_mixed = vec![0.0f32; DOWN_BODY_ROWS];
        for (index, item) in loaded.iter().enumerate() {
            for (slot, value) in item.expected_down.iter().enumerate() {
                expected_mixed[slot] += selected_weights[index] * value;
            }
            for (slot, value) in last_outputs[index].iter().enumerate() {
                observed_mixed[slot] += selected_weights[index] * value;
            }
        }
        let mixed_parity = output_metrics(&expected_mixed, &observed_mixed);
        if mixed_parity
            .get("within_tolerance")
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err(
                format!("weighted expert composition parity failed: {mixed_parity}").into(),
            );
        }

        let source_selection_parity = router
            .get("source_selection_parity")
            .cloned()
            .unwrap_or_else(|| json!({"status": "UNKNOWN"}));
        let source_mismatch = source_selection_parity
            .get("expert_ids_exact_match")
            .and_then(Value::as_bool)
            == Some(false);
        let mut physical_graph = json!({
            "schema": "hcli.physical_graph.v1",
            "semantic_type": "PhysicalGraph",
            "compiler_stage": "HawkingAccelerator",
            "component_scope": "native router receipt -> selected full persisted gate_up bodies -> native gate_up_swiglu -> device-resident 640-value activation -> selected full persisted down bodies -> native Q4/G64 down projection -> host weighted gather; no complete token graph",
            "device_placement": {"selected": "apple_metal", "candidates": ["apple_metal", "cpu"]},
            "native_gate_up_kernel": GATE_UP_KERNEL_NAME,
            "native_down_kernel": KERNEL_NAME,
            "native_kernel_execution_observed": true,
            "native_gate_up_swiglu_observed": true,
            "native_down_projection_observed": true,
            "native_expert_composition_observed": true,
            "device_intermediate_no_host_roundtrip": true,
            "selected_expert_count": loaded.len(),
            "dispatches_per_route": loaded.len() * 2,
            "gate_up_rows": GATE_UP_ROWS,
            "down_rows": DOWN_BODY_ROWS,
            "down_columns": DOWN_COLUMNS,
            "source_selection_parity_status": source_selection_parity.get("status"),
            "source_selection_parity_qualified": source_selection_parity.get("expert_ids_exact_match"),
            "source_selection_mismatch_accepted_as_bounded_boundary": source_mismatch,
            "promotion_allowed": false,
        });
        let physical_graph_fingerprint = sha256_bytes(&serde_json::to_vec(&physical_graph)?);
        physical_graph["fingerprint"] = Value::String(physical_graph_fingerprint);

        let device_memory = context.device_memory_limits();
        let mut components_json = Vec::with_capacity(loaded.len());
        for (index, item) in loaded.iter().enumerate() {
            components_json.push(json!({
                "expert_index": item.gate_up_body.expert_index,
                "gate_up": {
                    "body_receipt": component_ref(&item.gate_up_body_receipt_path, &item.gate_up_body_receipt, Some(&item.gate_up_body.receipt_sha256)),
                    "kernel_receipt": component_ref(&item.gate_up_kernel_receipt_path, &item.gate_up_kernel_receipt, Some(&item.gate_up_kernel_receipt_sha256)),
                    "candidate_body": {
                        "path": item.gate_up_body.path,
                        "sha256": item.gate_up_body.body_sha256,
                        "bytes": item.gate_up_body.code_bytes + item.gate_up_body.scale_bytes,
                        "source_independent": true,
                        "candidate_body_persisted": true,
                        "layout": "fused gate_up body split into contiguous gate/up row halves",
                        "label": "[D]",
                    },
                    "native_dispatch": {
                        "kernel": GATE_UP_KERNEL_NAME,
                        "gpu_ns": per_expert_gate_up_gpu_ns[index],
                        "gpu_ns_median": percentile_median(&per_expert_gate_up_gpu_ns[index]),
                        "output_hashes": per_expert_gate_up_hashes[index],
                        "stable_output": true,
                    },
                    "parity": output_metrics(&item.expected_gate_up, &last_gate_up_outputs[index]),
                },
                "down": {
                    "body_receipt": component_ref(&item.down_body_receipt_path, &item.down_body_receipt, Some(&item.down.receipt_sha256)),
                    "kernel_receipt": component_ref(&item.down_kernel_receipt_path, &item.down_kernel_receipt, Some(&item.down_kernel_receipt_sha256)),
                    "candidate_body": {
                        "path": item.down.path,
                        "sha256": item.down.body_sha256,
                        "bytes": item.down.code_bytes + item.down.scale_bytes,
                        "source_independent": true,
                        "candidate_body_persisted": true,
                        "label": "[D]",
                    },
                    "native_dispatch": {
                        "kernel": KERNEL_NAME,
                        "gpu_ns": per_expert_down_gpu_ns[index],
                        "gpu_ns_median": percentile_median(&per_expert_down_gpu_ns[index]),
                        "output_hashes": per_expert_down_hashes[index],
                        "stable_output": true,
                    },
                    "parity": output_metrics(&item.expected_down, &last_outputs[index]),
                },
                "native_dispatch": {
                    "dispatches_per_route": 2,
                    "gpu_ns": per_expert_total_gpu_ns[index],
                    "gpu_ns_median": percentile_median(&per_expert_total_gpu_ns[index]),
                    "host_wall_ns": per_expert_host_ns[index],
                    "host_wall_ns_median": percentile_median(&per_expert_host_ns[index]),
                    "device_intermediate": "gate_up_swiglu_output_buffer -> down_projection_input",
                    "host_intermediate_materialization": false,
                },
                "source_reference_used_for_execution": false,
            }));
        }

        let route_gpu_median = percentile_median(&route_gpu_ns);
        let route_host_median = percentile_median(&route_host_ns);
        Ok(json!({
            "schema": COMPOSITION_SCHEMA,
            "nomenclature_version": NOMENCLATURE_VERSION,
            "semantic_type": "NoeticExecutableCandidate",
            "compiler_stage": "HawkingAccelerator",
            "status": "PASSED",
            "qualification": "BOUNDED_NATIVE_ROUTED_EXPERT_GATE_UP_DOWN_COMPOSITION",
            "repo": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "root": root,
            "model_lake_manifest": manifest,
            "router_receipt": {
                "path": args.router_receipt.canonicalize()?,
                "sha256": router_sha256,
                "schema": ROUTER_SCHEMA,
                "status": "PASSED",
                "label": "[V]",
            },
            "component_receipt_policy": {
                "directory": repo.join("receipts/headless"),
                "gate_up_body_pattern": "FLASH_NOETIC_GATE_UP_BODY_E{expert}_R0_1280.json",
                "gate_up_kernel_pattern": "FLASH_NOETIC_GATE_UP_KERNEL_E{expert}_R0_1280_PARITY.json",
                "down_body_pattern": "FLASH_NOETIC_DOWN_BODY_E{expert}_R0_2560.json",
                "down_kernel_pattern": "FLASH_NOETIC_DOWN_KERNEL_E{expert}_R0_2560_PARITY.json",
                "gate_up_body_rows": GATE_UP_BODY_ROWS,
                "gate_up_rows": GATE_UP_ROWS,
                "down_body_rows": DOWN_BODY_ROWS,
                "down_columns": DOWN_COLUMNS,
            },
            "selection": router.get("selection").cloned().unwrap_or_else(|| json!({})),
            "source_selection_parity": source_selection_parity,
            "components": components_json,
            "execution": {
                "provider": "apple-metal",
                "operation": "native_router_selection_receipt -> full persisted gate_up body load -> native Q4_G64 gate_up_swiglu -> device-resident gate/up activation buffer -> full persisted down body load -> native Q4_G64 down_projection -> host_selected_weight_gather",
                "selected_expert_ids": selected_ids,
                "selected_expert_count": loaded.len(),
                "dispatches_per_expert": 2,
                "dispatches_per_route": loaded.len() * 2,
                "measured_routes": args.reps,
                "total_measured_dispatches": loaded.len() * args.reps * 2,
                "native_gate_up_swiglu_observed": true,
                "native_down_projection_observed": true,
                "native_expert_composition_observed": true,
                "bounded_selected_expert_output_observed": true,
                "device_intermediate_no_host_roundtrip": true,
                "source_reference_used_for_execution": false,
                "source_selection_mismatch_accepted_as_bounded_boundary": source_mismatch,
                "body_mutated": false,
                "model_loaded": false,
                "complete_expert_activation": false,
                "complete_token_runtime": false,
            },
            "input": {
                "definition": "((index * 71) mod 509 - 254) / 509",
                "values": input.len(),
                "deterministic_sha256": input_sha256,
                "label": "[V]",
            },
            "intermediate": {
                "semantic_type": "NoeticActivationBuffer",
                "shape": [GATE_UP_ROWS],
                "dtype": "F32",
                "producer": GATE_UP_KERNEL_NAME,
                "consumer": KERNEL_NAME,
                "device_resident": true,
                "host_roundtrip": false,
            },
            "gpu_timing": {
                "device": context.device_name(),
                "warmup_runs": args.warmup,
                "measured_runs": args.reps,
                "warmup_route_gpu_ns": warmup_route_gpu_ns,
                "route_gpu_ns": route_gpu_ns,
                "route_gpu_ns_median": route_gpu_median,
                "route_host_wall_ns": route_host_ns,
                "route_host_wall_ns_median": route_host_median,
                "dispatches_per_route": loaded.len() * 2,
                "output_hashes": mixed_hashes,
                "memory_limits": {
                    "max_buffer_length": device_memory.max_buffer_length,
                    "recommended_max_working_set_size": device_memory.recommended_max_working_set_size,
                    "current_allocated_size": device_memory.current_allocated_size,
                    "has_unified_memory": device_memory.has_unified_memory,
                },
                "timing_authority": "Metal completed-command-buffer GPUStartTime/GPUEndTime for native gate_up_swiglu plus down projection dispatches; host wall reported separately",
            },
            "gather": {
                "status": "BOUNDED_HOST_WEIGHTED_GATHER",
                "weights": selected_weights,
                "output_rows": DOWN_BODY_ROWS,
                "output_sha256": mixed_hashes.last(),
                "parity": mixed_parity,
                "fused_native_gather": false,
            },
            "noetic_ir": {
                "schema": "hcli.noetic.ir.v1",
                "semantic_type": "NoeticIR",
                "operations": [
                    "consume_native_router_selection",
                    "resolve_selected_full_persisted_fused_gate_up_bodies",
                    "execute_native_q4_g64_gate_up_swiglu_per_selected_expert",
                    "retain_gate_up_activation_in_device_buffer",
                    "resolve_selected_full_persisted_down_projection_bodies",
                    "execute_native_q4_g64_down_projection_per_selected_expert",
                    "apply_selected_router_weights_on_host",
                    "emit_bounded_selected_expert_outputs",
                ],
                "source_independent": true,
                "complete_expert": false,
                "complete_model": false,
            },
            "physical_graph": physical_graph,
            "whole_model_capability": "NOT_TESTED",
            "complete_expert_runtime": "NOT_TESTED",
            "complete_token_runtime": "NOT_TESTED",
            "complete_system_ebpw": null,
            "flash_tps": null,
            "body_mutated": false,
            "model_loaded": false,
            "native_gate_up_swiglu_observed": true,
            "native_down_projection_observed": true,
            "native_expert_composition_observed": true,
            "bounded_selected_expert_output_observed": true,
            "promotion_allowed": false,
            "claim_boundary": "PASSED bounded native source-independent composition of the ten selected persisted routed experts: Q4/G64 gate+up/SwiGLU writes a device-resident 640-value activation consumed directly by Q4/G64 down projection, followed by a host-side selected-weight gather. This is not a complete Flash model, complete-token runtime, Flash TPS, or EBPW result; source-selection mismatch remains explicit.",
            "next_action": "qualify the remaining Flash graph and protected complete-token capability; do not populate Flash TPS or EBPW until the full native token runtime is independently observed",
            "elapsed_s": started.elapsed().as_secs_f64(),
        }))
    }

    pub fn main() -> Result<(), Box<dyn Error>> {
        let args = parse_args()?;
        let destination = args.out.clone();
        let report = match if args.expert_composition {
            run_expert_composition(&args)
        } else if args.gate_up_swiglu {
            run_gate_up_swiglu(&args)
        } else {
            run(&args)
        } {
            Ok(report) => report,
            Err(error) => json!({
                "schema": if args.expert_composition {
                    COMPOSITION_SCHEMA
                } else if args.gate_up_swiglu {
                    GATE_UP_SCHEMA
                } else {
                    SCHEMA
                },
                "nomenclature_version": NOMENCLATURE_VERSION,
                "status": "FAILED",
                "repo": REPO_ID,
                "pinned_revision": PINNED_REVISION,
                "error": {"type": "RoutedExpertDispatchError", "message": error.to_string()},
                "body_mutated": false,
                "model_loaded": false,
                "native_routed_body_dispatch_observed": false,
                "native_gate_up_swiglu_observed": false,
                "native_expert_gate_up_activation_observed": false,
                "native_down_projection_observed": false,
                "native_expert_composition_observed": false,
                "whole_model_capability": "NOT_TESTED",
                "complete_expert_runtime": "NOT_TESTED",
                "complete_token_runtime": "NOT_TESTED",
                "promotion_allowed": false,
            }),
        };
        write_atomic(&destination, &report)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        if report.get("status").and_then(Value::as_str) == Some("PASSED") {
            Ok(())
        } else {
            Err("Flash Noetic native routed-expert dispatch failed; see the receipt".into())
        }
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::main()
}
