//! Bounded native Flash-Next Noetic router selection.
//!
//! This executable consumes the persisted source-independent Q4/G64 router
//! body, runs the existing Metal matvec over all pinned router rows, and
//! applies the pinned FP32 softmax/top-k contract to the resulting GPU logits.
//! The source-selection receipt is used only as a comparison reference.  The
//! executable does not load the whole model, read source weights for its
//! execution path, run a token loop, or claim Flash TPS/EBPW qualification.

#![recursion_limit = "512"]

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("Flash Noetic router selection requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
mod macos {
    use half::f16;
    use hawking_core::metal::{MetalContext, MetalDispatchTiming};
    use metal::Buffer;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::cmp::Ordering;
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    const SCHEMA: &str = "hawking.flash_noetic_router_selection_native.v1";
    const BODY_SCHEMA: &str = "hcli.agentos.flash_noetic_component_body.v1";
    const KERNEL_SCHEMA: &str = "hawking.flash_noetic_q4_kernel_parity.v1";
    const SELECTION_SCHEMA: &str = "hcli.agentos.flash_noetic_router_selection.v1";
    const NOMENCLATURE_VERSION: &str = "HAWKING_NOMENCLATURE_V1";
    const REPO_ID: &str = "Qwen/Qwen3.8-Flash-Next";
    const PINNED_REVISION: &str = "34567a4712bc9766c4449e2e98e4468bfa24d915";
    const TENSOR_NAME: &str = "model.language_model.layers.0.mlp.gate.weight";
    const GROUP_SIZE: usize = 64;
    const CODE_BYTES_PER_GROUP: usize = GROUP_SIZE / 2;
    const KERNEL_NAME: &str = "qwen_uniform_q4_group64_matvec";
    const REFERENCE_MULTIPLIER: usize = 71;
    const REFERENCE_MODULUS: usize = 509;
    const REFERENCE_OFFSET: f32 = 254.0;
    const DEFAULT_WARMUP: usize = 2;
    const DEFAULT_REPS: usize = 7;
    const OUTPUT_ERROR_TOLERANCE: f32 = 2.0e-3;
    const MANIFEST_PATH: &str =
        "/Volumes/corpdrive/hawking-modellake/manifests/Qwen--Qwen3.8-Flash-Next@34567a4712bc.json";

    struct Args {
        root: PathBuf,
        body_receipt: PathBuf,
        kernel_receipt: PathBuf,
        reference_receipt: PathBuf,
        warmup: usize,
        reps: usize,
        out: PathBuf,
    }

    struct PackedBody {
        path: PathBuf,
        receipt_sha256: String,
        body_sha256: String,
        rows: usize,
        columns: usize,
        code_bytes: usize,
        scale_bytes: usize,
        codes: Vec<u8>,
        scales: Vec<u8>,
    }

    struct Selection {
        expert_ids: Vec<usize>,
        router_probabilities: Vec<f32>,
        selected_weights: Vec<f32>,
        selected_probability_sum: f32,
        selected_weight_sum: f32,
        probabilities_sha256: String,
        logits_sha256: String,
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
            body_receipt: repo
                .join("receipts/headless/FLASH_NOETIC_ROUTER_COMPONENT_FULL_BODY.json"),
            kernel_receipt: repo
                .join("receipts/headless/FLASH_NOETIC_ROUTER_COMPONENT_FULL_KERNEL_PARITY.json"),
            reference_receipt: repo.join("receipts/headless/FLASH_NOETIC_ROUTER_SELECTION.json"),
            warmup: DEFAULT_WARMUP,
            reps: DEFAULT_REPS,
            out: repo.join("receipts/headless/FLASH_NOETIC_ROUTER_SELECTION_NATIVE.json"),
        };
        let mut values = env::args().skip(1);
        while let Some(flag) = values.next() {
            match flag.as_str() {
                "--root" => args.root = PathBuf::from(values.next().ok_or("missing --root")?),
                "--body-receipt" => {
                    args.body_receipt =
                        PathBuf::from(values.next().ok_or("missing --body-receipt")?)
                }
                "--kernel-receipt" => {
                    args.kernel_receipt =
                        PathBuf::from(values.next().ok_or("missing --kernel-receipt")?)
                }
                "--reference-receipt" => {
                    args.reference_receipt =
                        PathBuf::from(values.next().ok_or("missing --reference-receipt")?)
                }
                "--warmup" => args.warmup = parse_usize(values.next(), &flag)?,
                "--reps" => args.reps = parse_usize(values.next(), &flag)?,
                "--out" => args.out = PathBuf::from(values.next().ok_or("missing --out")?),
                "--help" | "-h" => {
                    println!(
                        "usage: flash_noetic_router_selection [--root DIR] \
                         [--body-receipt FILE] [--kernel-receipt FILE] \
                         [--reference-receipt FILE] [--warmup N] [--reps N] [--out FILE]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other}").into()),
            }
        }
        if args.warmup > 64 {
            return Err("--warmup must be <= 64".into());
        }
        if args.reps == 0 || args.reps > 128 {
            return Err("--reps must be in 1..=128".into());
        }
        Ok(args)
    }

    fn sha256_bytes(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn read_json(path: &Path) -> Result<(Value, String), Box<dyn Error>> {
        let bytes = fs::read(path.canonicalize()?)?;
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

    fn validate_body(path: &Path) -> Result<(PackedBody, Value), Box<dyn Error>> {
        let canonical_receipt = path.canonicalize()?;
        let (receipt, receipt_sha256) = read_json(&canonical_receipt)?;
        if string_field(&receipt, "schema")? != BODY_SCHEMA
            || string_field(&receipt, "status")? != "PASSED"
            || string_field(&receipt, "repo")? != REPO_ID
            || string_field(&receipt, "pinned_revision")? != PINNED_REVISION
            || string_field(&receipt, "nomenclature_version")? != NOMENCLATURE_VERSION
        {
            return Err("router body receipt is not a PASSED pinned Noetic body".into());
        }
        if receipt.get("source_independent").and_then(Value::as_bool) != Some(true)
            || receipt
                .get("candidate_body_persisted")
                .and_then(Value::as_bool)
                != Some(true)
            || receipt.get("body_mutated").and_then(Value::as_bool) != Some(false)
            || receipt.get("model_loaded").and_then(Value::as_bool) != Some(false)
        {
            return Err("router body receipt fails source-independent execution guards".into());
        }
        let source = receipt
            .get("source_block")
            .ok_or("router body receipt has no source_block")?;
        if string_field(source, "tensor_name")? != TENSOR_NAME
            || string_field(source, "dtype")?.to_uppercase() != "BF16"
            || usize_field(source, "row_start")? != 0
        {
            return Err("router body source block is not the pinned full gate matrix".into());
        }
        let shape = source
            .get("shape")
            .and_then(Value::as_array)
            .ok_or("router body source block has no shape")?;
        if shape.len() != 2 {
            return Err("router body source shape must be rank two".into());
        }
        let rows = shape[0]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("router body row dimension is invalid")?;
        let columns = shape[1]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("router body column dimension is invalid")?;
        if rows == 0 || rows > 4096 || columns == 0 || columns > 16384 || columns % GROUP_SIZE != 0
        {
            return Err("router body shape is outside the bounded native selection limits".into());
        }
        if usize_field(source, "row_count")? != rows
            || usize_field(source, "bytes")? != rows * columns * 2
        {
            return Err("router body source block is not complete".into());
        }
        let representation = receipt
            .get("representation_descriptor")
            .ok_or("router body receipt has no representation descriptor")?;
        if string_field(representation, "candidate_id")? != "independent_q4_g64" {
            return Err("router body candidate is not independent_q4_g64".into());
        }
        let body_record = receipt
            .get("body")
            .ok_or("router body receipt has no body")?;
        let body_path = Path::new(string_field(body_record, "path")?).canonicalize()?;
        let body = fs::read(&body_path)?;
        let body_sha256 = sha256_bytes(&body);
        if body_sha256 != string_field(body_record, "sha256")? {
            return Err("router body bytes do not match the body receipt hash".into());
        }
        let code_bytes = usize_field(body_record, "code_bytes")?;
        let scale_bytes = usize_field(body_record, "scale_bytes")?;
        let expected_code_bytes = rows * (columns / 2);
        let expected_scale_bytes = rows * (columns / GROUP_SIZE) * 2;
        if code_bytes != expected_code_bytes
            || scale_bytes != expected_scale_bytes
            || body.len() != code_bytes + scale_bytes
        {
            return Err("router body bytes do not match the Q4/G64 shape".into());
        }
        Ok((
            PackedBody {
                path: body_path,
                receipt_sha256,
                body_sha256,
                rows,
                columns,
                code_bytes,
                scale_bytes,
                codes: body[..code_bytes].to_vec(),
                scales: body[code_bytes..].to_vec(),
            },
            receipt,
        ))
    }

    fn validate_kernel(path: &Path, body: &PackedBody) -> Result<(Value, String), Box<dyn Error>> {
        let (receipt, digest) = read_json(path)?;
        if string_field(&receipt, "schema")? != KERNEL_SCHEMA
            || string_field(&receipt, "status")? != "PASSED"
            || string_field(&receipt, "repo")? != REPO_ID
            || string_field(&receipt, "pinned_revision")? != PINNED_REVISION
            || string_field(&receipt, "nomenclature_version")? != NOMENCLATURE_VERSION
        {
            return Err("router kernel receipt is not a PASSED pinned parity receipt".into());
        }
        if receipt.get("body_mutated").and_then(Value::as_bool) != Some(false)
            || receipt.get("model_loaded").and_then(Value::as_bool) != Some(false)
        {
            return Err("router kernel receipt fails mutation/model-load guards".into());
        }
        let loader = receipt
            .get("native_loader")
            .ok_or("router kernel receipt has no native_loader")?;
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
            return Err("router kernel receipt does not prove persisted-body execution".into());
        }
        let native_kernel = receipt
            .get("native_kernel")
            .ok_or("router kernel receipt has no native_kernel")?;
        if native_kernel
            .get("kernel_registered")
            .and_then(Value::as_bool)
            != Some(true)
            || string_field(native_kernel, "kernel")? != KERNEL_NAME
        {
            return Err("router kernel receipt does not identify the expected Metal kernel".into());
        }
        if receipt
            .get("parity")
            .and_then(|value| value.get("within_tolerance"))
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err("router kernel parity receipt is not within tolerance".into());
        }
        let candidate = receipt
            .get("candidate_body")
            .ok_or("router kernel receipt has no candidate_body")?;
        if Path::new(string_field(candidate, "path")?).canonicalize()? != body.path
            || string_field(candidate, "sha256")? != body.body_sha256
            || usize_field(candidate, "bytes")? != body.code_bytes + body.scale_bytes
        {
            return Err("router kernel candidate body does not match the persisted body".into());
        }
        Ok((receipt, digest))
    }

    fn validate_reference(path: &Path) -> Result<(Value, String), Box<dyn Error>> {
        let (receipt, digest) = read_json(path)?;
        if string_field(&receipt, "schema")? != SELECTION_SCHEMA
            || string_field(&receipt, "status")? != "PASSED"
            || string_field(&receipt, "repo")? != REPO_ID
            || string_field(&receipt, "pinned_revision")? != PINNED_REVISION
            || string_field(&receipt, "nomenclature_version")? != NOMENCLATURE_VERSION
        {
            return Err("router selection reference is not a PASSED pinned receipt".into());
        }
        if receipt.get("body_mutated").and_then(Value::as_bool) != Some(false)
            || receipt.get("model_loaded").and_then(Value::as_bool) != Some(false)
            || receipt
                .get("native_selection_execution_observed")
                .and_then(Value::as_bool)
                != Some(false)
        {
            return Err("router selection reference fails its derived-only guards".into());
        }
        let config = receipt
            .get("config")
            .and_then(|value| value.get("router"))
            .ok_or("router selection reference has no router config")?;
        let experts = usize_field(config, "num_experts")?;
        let top_k = usize_field(config, "num_experts_per_tok")?;
        if experts == 0 || top_k == 0 || top_k > experts {
            return Err("router selection reference has invalid expert counts".into());
        }
        for key in ["selection", "source_selection"] {
            let ids = receipt
                .get(key)
                .and_then(|value| value.get("expert_ids"))
                .and_then(Value::as_array)
                .ok_or_else(|| format!("router selection reference has no {key} expert ids"))?;
            if ids.len() != top_k
                || ids.iter().any(|value| {
                    value
                        .as_u64()
                        .map(|id| id as usize >= experts)
                        .unwrap_or(true)
                })
            {
                return Err(format!("router selection reference {key} ids are invalid").into());
            }
        }
        Ok((receipt, digest))
    }

    fn deterministic_input(columns: usize) -> Vec<f32> {
        (0..columns)
            .map(|index| {
                ((index * REFERENCE_MULTIPLIER % REFERENCE_MODULUS) as f32 - REFERENCE_OFFSET)
                    / REFERENCE_MODULUS as f32
            })
            .collect()
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
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

    fn select_router(
        logits: &[f32],
        top_k: usize,
        normalize: bool,
    ) -> Result<Selection, Box<dyn Error>> {
        if logits.is_empty() || top_k == 0 || top_k > logits.len() {
            return Err("router logits/top_k shape is invalid".into());
        }
        if !logits.iter().all(|value| value.is_finite()) {
            return Err("router logits contain a non-finite value".into());
        }
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probabilities = Vec::with_capacity(logits.len());
        for value in logits {
            probabilities.push((*value - maximum).exp());
        }
        let denominator: f32 = probabilities.iter().copied().sum();
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err("router softmax denominator is not finite and positive".into());
        }
        for value in &mut probabilities {
            *value /= denominator;
        }
        let mut order: Vec<usize> = (0..probabilities.len()).collect();
        order.sort_by(|left, right| {
            probabilities[*right]
                .partial_cmp(&probabilities[*left])
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.cmp(right))
        });
        order.truncate(top_k);
        let router_probabilities: Vec<f32> =
            order.iter().map(|index| probabilities[*index]).collect();
        let selected_probability_sum: f32 = router_probabilities.iter().copied().sum();
        if !selected_probability_sum.is_finite() || selected_probability_sum <= 0.0 {
            return Err("selected router probabilities are not finite and positive".into());
        }
        let selected_weights: Vec<f32> = if normalize {
            router_probabilities
                .iter()
                .map(|value| *value / selected_probability_sum)
                .collect()
        } else {
            router_probabilities.clone()
        };
        let selected_weight_sum: f32 = selected_weights.iter().copied().sum();
        Ok(Selection {
            expert_ids: order,
            router_probabilities,
            selected_weights,
            selected_probability_sum,
            selected_weight_sum,
            probabilities_sha256: sha256_bytes(&f32_bytes(&probabilities)),
            logits_sha256: sha256_bytes(&f32_bytes(logits)),
        })
    }

    fn selection_json(selection: &Selection) -> Value {
        json!({
            "expert_ids": selection.expert_ids,
            "router_probabilities": selection.router_probabilities,
            "selected_weights": selection.selected_weights,
            "selected_probability_sum": selection.selected_probability_sum,
            "selected_weight_sum": selection.selected_weight_sum,
            "probability_vector_sha256": selection.probabilities_sha256,
            "logits_sha256": selection.logits_sha256,
            "probabilities_finite": true,
        })
    }

    fn receipt_ids(receipt: &Value, field: &str) -> Result<Vec<usize>, Box<dyn Error>> {
        receipt
            .get(field)
            .and_then(|value| value.get("expert_ids"))
            .and_then(Value::as_array)
            .ok_or_else(|| -> Box<dyn Error> {
                format!("reference receipt has no {field}.expert_ids").into()
            })?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|number| usize::try_from(number).ok())
                    .ok_or_else(|| {
                        format!("reference receipt {field} contains an invalid id").into()
                    })
            })
            .collect()
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
        let (body, body_receipt) = validate_body(&args.body_receipt)?;
        let (kernel, kernel_sha256) = validate_kernel(&args.kernel_receipt, &body)?;
        let (reference, reference_sha256) = validate_reference(&args.reference_receipt)?;
        let input = deterministic_input(body.columns);
        let input_sha256 = sha256_bytes(&f32_bytes(&input));
        if let Some(native_input) = kernel.get("input") {
            if native_input
                .get("deterministic_sha256")
                .and_then(Value::as_str)
                != Some(input_sha256.as_str())
            {
                return Err("native parity input does not match the bounded router input".into());
            }
        }

        let expected = cpu_matvec(&body, &input);
        let context = MetalContext::new_with_trace(true)?;
        let codes_buffer = context.new_buffer_with_bytes_checked(&body.codes)?;
        let scales_buffer = context.new_buffer_with_bytes_checked(&body.scales)?;
        let input_buffer = context.new_buffer_with_bytes_checked(&f32_bytes(&input))?;
        let output_buffer = context.new_buffer_checked(body.rows * std::mem::size_of::<f32>())?;
        let mut warmup_gpu_ns = Vec::with_capacity(args.warmup);
        for _ in 0..args.warmup {
            warmup_gpu_ns.push(gpu_ns(dispatch(
                &context,
                &body,
                &codes_buffer,
                &scales_buffer,
                &input_buffer,
                &output_buffer,
            )?)?);
        }
        let mut gpu_ns_samples = Vec::with_capacity(args.reps);
        let mut host_ns_samples = Vec::with_capacity(args.reps);
        let mut output_hashes = Vec::with_capacity(args.reps);
        let mut last_output = Vec::new();
        for _ in 0..args.reps {
            let timing = dispatch(
                &context,
                &body,
                &codes_buffer,
                &scales_buffer,
                &input_buffer,
                &output_buffer,
            )?;
            let output = read_f32(&output_buffer, body.rows);
            if !output.iter().all(|value| value.is_finite()) {
                return Err("native router logits contain a non-finite value".into());
            }
            gpu_ns_samples.push(gpu_ns(timing)?);
            host_ns_samples.push(timing.host_wall_us.saturating_mul(1000));
            output_hashes.push(sha256_bytes(&f32_bytes(&output)));
            last_output = output;
        }
        let parity = output_metrics(&expected, &last_output);
        if parity.get("within_tolerance").and_then(Value::as_bool) != Some(true) {
            return Err(format!("native router matvec parity failed: {parity}").into());
        }
        if output_hashes.windows(2).any(|pair| pair[0] != pair[1]) {
            return Err("native router logits changed across repeated executions".into());
        }

        let reference_config = reference
            .get("config")
            .and_then(|value| value.get("router"))
            .ok_or("reference router config is missing")?;
        let top_k = usize_field(reference_config, "num_experts_per_tok")?;
        let normalize = reference_config
            .get("norm_topk_prob")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        if body.rows != usize_field(reference_config, "num_experts")? {
            return Err("router body row count does not match the pinned router config".into());
        }
        let selection = select_router(&last_output, top_k, normalize)?;
        let candidate_reference_ids = receipt_ids(&reference, "selection")?;
        let source_reference_ids = receipt_ids(&reference, "source_selection")?;
        let candidate_ids_match = selection.expert_ids == candidate_reference_ids;
        let source_ids_match = selection.expert_ids == source_reference_ids;
        let common_source_ids = selection
            .expert_ids
            .iter()
            .filter(|id| source_reference_ids.contains(id))
            .count();
        let device_memory = context.device_memory_limits();
        let mut gpu_sorted = gpu_ns_samples.clone();
        gpu_sorted.sort_unstable();
        let mut host_sorted = host_ns_samples.clone();
        host_sorted.sort_unstable();
        let source_block = body_receipt
            .get("source_block")
            .cloned()
            .ok_or("router body source block disappeared")?;
        let reference = json!({
            "receipt_path": args.reference_receipt.canonicalize()?,
            "receipt_sha256": reference_sha256,
            "candidate_expert_ids": candidate_reference_ids,
            "source_expert_ids": source_reference_ids,
            "candidate_selection_ids_match": candidate_ids_match,
            "source_selection_ids_match": source_ids_match,
            "source_top_k_overlap_count": common_source_ids,
            "source_top_k_overlap_fraction": common_source_ids as f64 / top_k as f64,
            "label": "[V]",
        });
        let source_selection_status = if source_ids_match {
            "MATCH"
        } else {
            "MISMATCH"
        };
        let mut physical_graph = json!({
            "schema": "hcli.physical_graph.v1",
            "semantic_type": "PhysicalGraph",
            "compiler_stage": "HawkingAccelerator",
            "component_scope": "full pinned Flash router matrix body through native Metal logits and derived FP32 selection; no token graph",
            "device_placement": {"selected": "apple_metal", "candidates": ["apple_metal", "cpu"]},
            "native_kernel_execution_observed": true,
            "source_selection_parity_status": source_selection_status,
            "source_selection_parity_qualified": source_ids_match,
            "promotion_allowed": false,
        });
        let physical_graph_fingerprint = sha256_bytes(&serde_json::to_vec(&physical_graph)?);
        physical_graph["fingerprint"] = Value::String(physical_graph_fingerprint);
        Ok(json!({
            "schema": SCHEMA,
            "nomenclature_version": NOMENCLATURE_VERSION,
            "semantic_type": "NoeticExecutableCandidate",
            "compiler_stage": "HawkingAccelerator",
            "status": "PASSED",
            "qualification": "BOUNDED_NATIVE_ROUTER_SELECTION",
            "repo": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "root": root,
            "model_lake_manifest": manifest,
            "body_receipt": {
                "path": args.body_receipt.canonicalize()?,
                "sha256": body.receipt_sha256,
                "schema": BODY_SCHEMA,
                "status": "PASSED",
                "label": "[V]",
            },
            "kernel_receipt": {
                "path": args.kernel_receipt.canonicalize()?,
                "sha256": kernel_sha256,
                "schema": KERNEL_SCHEMA,
                "status": "PASSED",
                "label": "[V]",
            },
            "source_block": source_block,
            "candidate_body": {
                "path": body.path,
                "sha256": body.body_sha256,
                "bytes": body.code_bytes + body.scale_bytes,
                "code_bytes": body.code_bytes,
                "scale_bytes": body.scale_bytes,
                "candidate_id": "independent_q4_g64",
                "execution_input": true,
                "source_independent": true,
                "label": "[D]",
            },
            "noetic_representation": {
                "candidate_id": "independent_q4_g64",
                "layout": "row-major [row, column]",
                "group_size": GROUP_SIZE,
                "code_dtype": "uint4_packed",
                "scale_dtype": "little_endian_float16",
                "effective_bits_per_value": (body.code_bytes + body.scale_bytes) as f64 * 8.0 / (body.rows * body.columns) as f64,
                "candidate_body_persisted": true,
                "dense_rematerialization": "forbidden",
                "label": "[D]",
            },
            "native_loader": {
                "status": "BOUNDED_NOETIC_PERSISTED_BODY_LOAD",
                "source_independent_execution": true,
                "body_loaded_directly": true,
                "source_tensor_read_for_execution": false,
                "dense_rematerialization": "forbidden",
                "whole_model_capability": "NOT_TESTED",
                "whole_model_runtime": "NOT_TESTED",
            },
            "native_kernel": {
                "kernel": KERNEL_NAME,
                "shader_family": "qwen_uniform_q4.metal",
                "kernel_registered": true,
                "dispatches_per_selection": 1,
                "grid": [body.rows, 1, 1],
                "threadgroup": [1, 1, 1],
                "selection_logits_are_gpu_output": true,
                "whole_model_capability": "NOT_TESTED",
                "whole_model_runtime": "NOT_TESTED",
                "label": "[V]/[D]",
            },
            "execution": {
                "provider": "apple-metal",
                "body_source_independent": true,
                "operation": "persisted_Q4_G64_body -> Metal_matvec -> FP32_softmax -> stable_top_k -> selected_weight_normalization",
                "model_loaded": false,
                "body_mutated": false,
                "native_selection_execution_observed": true,
                "source_reference_used_for_execution": false,
                "source_reference_used_for_comparison": true,
                "selected_expert_ids": selection.expert_ids,
            },
            "input": {
                "definition": "((index * 71) mod 509 - 254) / 509",
                "values": body.columns,
                "deterministic_sha256": input_sha256,
                "label": "[V]",
            },
            "selection_config": {
                "num_experts": body.rows,
                "num_experts_per_tok": top_k,
                "norm_topk_prob": normalize,
                "router_probability": "softmax(router_logits, dtype=float32, dim=-1)",
                "selection": "topk(router_probs, num_experts_per_tok)",
                "stable_tie_break": "expert_id_ascending",
                "label": "[V]",
            },
            "selection": selection_json(&selection),
            "reference": reference,
            "source_selection_parity": {
                "status": source_selection_status,
                "qualification": if source_ids_match { "SOURCE_ROUTER_SELECTION_MATCH" } else { "SOURCE_ROUTER_SELECTION_NOT_QUALIFIED" },
                "expert_ids_exact_match": source_ids_match,
                "top_k_overlap_count": common_source_ids,
                "top_k_overlap_fraction": common_source_ids as f64 / top_k as f64,
                "candidate_reference_ids_match": candidate_ids_match,
                "label": "[V]/[D]",
            },
            "parity": parity,
            "gpu_timing": {
                "device": context.device_name(),
                "warmup_runs": args.warmup,
                "measured_runs": args.reps,
                "warmup_gpu_ns": warmup_gpu_ns,
                "gpu_ns": gpu_ns_samples,
                "gpu_ns_min": gpu_sorted[0],
                "gpu_ns_median": gpu_sorted[gpu_sorted.len() / 2],
                "gpu_ns_max": *gpu_sorted.last().unwrap(),
                "host_wall_ns": host_ns_samples,
                "host_wall_ns_min": host_sorted[0],
                "host_wall_ns_median": host_sorted[host_sorted.len() / 2],
                "host_wall_ns_max": *host_sorted.last().unwrap(),
                "output_hashes": output_hashes,
                "timing_authority": "Metal completed-command-buffer GPUStartTime/GPUEndTime; host wall reported separately",
                "memory_limits": {
                    "max_buffer_length": device_memory.max_buffer_length,
                    "recommended_max_working_set_size": device_memory.recommended_max_working_set_size,
                    "current_allocated_size": device_memory.current_allocated_size,
                    "has_unified_memory": device_memory.has_unified_memory,
                },
            },
            "noetic_ir": {
                "schema": "hcli.noetic.ir.v1",
                "semantic_type": "NoeticIR",
                "operations": [
                    "load_persisted_source_independent_router_body",
                    "execute_native_q4_g64_router_matvec",
                    "compute_fp32_router_softmax",
                    "select_router_top_k",
                    "normalize_selected_router_weights",
                    "emit_selected_expert_ids_and_weights",
                ],
                "source_independent": true,
                "complete_model": false,
            },
            "physical_graph": physical_graph,
            "whole_model_capability": "NOT_TESTED",
            "complete_token_runtime": "NOT_TESTED",
            "complete_system_ebpw": null,
            "flash_tps": null,
            "body_mutated": false,
            "model_loaded": false,
            "native_selection_execution_observed": true,
            "promotion_allowed": false,
            "claim_boundary": "PASSED bounded native source-independent Flash router selection over a persisted Q4/G64 body. Native Metal logits and FP32 top-k are physically observed; source top-k parity is reported explicitly and is not qualified on mismatch. No whole-model loader, capability, complete-token runtime, EBPW, or Flash TPS claim is made.",
            "next_action": "connect native router selection to persisted routed-expert bodies and a protected complete-token graph only after source-selection parity is qualified or the mismatch is explicitly accepted as a representation boundary",
            "elapsed_s": started.elapsed().as_secs_f64(),
        }))
    }

    pub fn main() -> Result<(), Box<dyn Error>> {
        let args = parse_args()?;
        let destination = args.out.clone();
        let report = match run(&args) {
            Ok(report) => report,
            Err(error) => json!({
                "schema": SCHEMA,
                "nomenclature_version": NOMENCLATURE_VERSION,
                "status": "FAILED",
                "repo": REPO_ID,
                "pinned_revision": PINNED_REVISION,
                "error": {"type": "RouterSelectionError", "message": error.to_string()},
                "body_mutated": false,
                "model_loaded": false,
                "native_selection_execution_observed": false,
                "whole_model_capability": "NOT_TESTED",
                "complete_token_runtime": "NOT_TESTED",
                "promotion_allowed": false,
            }),
        };
        write_atomic(&destination, &report)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        if report.get("status").and_then(Value::as_str) == Some("PASSED") {
            Ok(())
        } else {
            Err("Flash Noetic native router selection failed; see the receipt".into())
        }
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::main()
}
