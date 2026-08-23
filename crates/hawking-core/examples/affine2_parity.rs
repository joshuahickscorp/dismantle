//! Native affine-2 group-32 kernel vs CPU reconstruction of one real MLX tensor.
//!
//! Loads `{name}.weight` (uint32, 16 LSB-first 2-bit codes/word),
//! `{name}.scales` and `{name}.biases` (fp16 or bf16, one pair per group of 32)
//! from the external 2-bit safetensors artifact. Dispatches the in-register
//! affine2 GEMV / dequant kernels and compares to the host oracle
//! `w = float(q) * scale + bias`. Does not call mlx or mlx_lm.
//!
//! ```text
//! cargo build -p hawking-core --example affine2_parity
//! cargo run -p hawking-core --example affine2_parity -- \
//!   --artifact /abs/path/to/abliterated-mlx-2bit/2bit
//! ```

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("affine2_parity requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

#[cfg(target_os = "macos")]
mod macos {
    use half::{bf16, f16};
    use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
    use serde_json::Value;
    use std::env;
    use std::error::Error;
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::path::{Path, PathBuf};

    const SHADER: &str = include_str!("../shaders/affine2_group32_matvec.metal");
    const GROUP_SIZE: usize = 32;
    const DEFAULT_TENSOR: &str = "language_model.model.layers.0.linear_attn.in_proj_a";
    const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
    const PASS_TOL: f32 = 1e-2;

    struct Args {
        artifact: PathBuf,
        tensor: String,
    }

    struct TensorBytes {
        dtype: String,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    struct Affine2Matrix {
        name: String,
        rows: usize,
        cols: usize,
        codes: Vec<u32>,
        scales: Vec<f32>,
        biases: Vec<f32>,
        scale_dtype: String,
        shard: PathBuf,
    }

    fn repository_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
    }

    fn default_artifact_candidates() -> Vec<PathBuf> {
        let rel = PathBuf::from("workspace/campaign/records/runs/qwen38-27b/abliterated-mlx-2bit/2bit");
        vec![
            repository_root().join(&rel),
            PathBuf::from("/Users/scammermike/Downloads/hawking").join(&rel),
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../").join(&rel),
        ]
    }

    fn resolve_default_artifact() -> Result<PathBuf, Box<dyn Error>> {
        for candidate in default_artifact_candidates() {
            if candidate.join("model.safetensors.index.json").is_file() {
                return Ok(candidate);
            }
        }
        Err(
            "2-bit artifact not found; pass --artifact /path/to/abliterated-mlx-2bit/2bit"
                .into(),
        )
    }

    fn parse_args() -> Result<Args, Box<dyn Error>> {
        let mut artifact = None;
        let mut tensor = DEFAULT_TENSOR.to_string();
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--artifact" => {
                    artifact = Some(PathBuf::from(
                        args.next().ok_or("missing value after --artifact")?,
                    ));
                }
                "--tensor" => {
                    tensor = args.next().ok_or("missing value after --tensor")?;
                    if let Some(stripped) = tensor.strip_suffix(".weight") {
                        tensor = stripped.to_string();
                    }
                }
                other => return Err(format!("unsupported option {other}").into()),
            }
        }
        Ok(Args {
            artifact: match artifact {
                Some(path) => path,
                None => resolve_default_artifact()?,
            },
            tensor,
        })
    }

    fn read_json(path: &Path) -> Result<Value, Box<dyn Error>> {
        let bytes = std::fs::read(path)?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    fn required_u64(value: &Value, field: &str) -> Result<u64, Box<dyn Error>> {
        value
            .get(field)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("missing unsigned {field}").into())
    }

    fn load_safetensors_tensor(shard: &Path, name: &str) -> Result<TensorBytes, Box<dyn Error>> {
        let mut file = File::open(shard)?;
        let mut header_len_bytes = [0u8; 8];
        file.read_exact(&mut header_len_bytes)?;
        let header_len = u64::from_le_bytes(header_len_bytes);
        if header_len == 0 || header_len > MAX_HEADER_BYTES {
            return Err(format!("invalid safetensors header length {header_len}").into());
        }
        let mut header_bytes = vec![0u8; header_len as usize];
        file.read_exact(&mut header_bytes)?;
        let header: Value = serde_json::from_slice(&header_bytes)?;
        let entry = header
            .get(name)
            .and_then(Value::as_object)
            .ok_or_else(|| format!("{} does not contain {name}", shard.display()))?;
        let dtype = entry
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or("tensor missing dtype")?
            .to_string();
        let shape = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or("tensor missing shape")?
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| "shape entry is not unsigned".into())
            })
            .collect::<Result<Vec<usize>, Box<dyn Error>>>()?;
        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or("tensor missing data_offsets")?;
        if offsets.len() != 2 {
            return Err("data_offsets must be [start, end]".into());
        }
        let start = offsets[0]
            .as_u64()
            .ok_or("data_offsets[0] is not unsigned")?;
        let end = offsets[1]
            .as_u64()
            .ok_or("data_offsets[1] is not unsigned")?;
        let nbytes = end
            .checked_sub(start)
            .ok_or("data_offsets end < start")? as usize;
        let file_off = 8u64
            .checked_add(header_len)
            .and_then(|o| o.checked_add(start))
            .ok_or("tensor file offset overflow")?;
        file.seek(SeekFrom::Start(file_off))?;
        let mut bytes = vec![0u8; nbytes];
        file.read_exact(&mut bytes)?;
        Ok(TensorBytes {
            dtype,
            shape,
            bytes,
        })
    }

    fn widen_f16x(bytes: &[u8], dtype: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        if bytes.len() % 2 != 0 {
            return Err(format!("{dtype} payload is not a multiple of 2 bytes").into());
        }
        let mut out = Vec::with_capacity(bytes.len() / 2);
        for chunk in bytes.chunks_exact(2) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            let value = match dtype {
                "F16" => f16::from_bits(bits).to_f32(),
                "BF16" => bf16::from_bits(bits).to_f32(),
                other => return Err(format!("unsupported 16-bit dtype {other}").into()),
            };
            if !value.is_finite() {
                return Err(format!("non-finite {dtype} scale/bias").into());
            }
            out.push(value);
        }
        Ok(out)
    }

    fn as_u32_le(bytes: &[u8]) -> Result<Vec<u32>, Box<dyn Error>> {
        if bytes.len() % 4 != 0 {
            return Err("uint32 payload is not a multiple of 4 bytes".into());
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn load_affine2(artifact: &Path, stem: &str) -> Result<Affine2Matrix, Box<dyn Error>> {
        let index_path = artifact.join("model.safetensors.index.json");
        let index = read_json(&index_path)?;
        let weight_map = index
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or("safetensors index missing weight_map")?;
        let mut shards = std::collections::BTreeSet::new();
        let weight_name = format!("{stem}.weight");
        let scales_name = format!("{stem}.scales");
        let biases_name = format!("{stem}.biases");
        for name in [&weight_name, &scales_name, &biases_name] {
            let shard = weight_map
                .get(name)
                .and_then(Value::as_str)
                .ok_or_else(|| format!("index does not map {name}"))?;
            shards.insert(artifact.join(shard));
        }
        if shards.len() != 1 {
            return Err("weight/scales/biases must live in one shard for this harness".into());
        }
        let shard = shards.into_iter().next().unwrap();
        let weight = load_safetensors_tensor(&shard, &weight_name)?;
        let scales = load_safetensors_tensor(&shard, &scales_name)?;
        let biases = load_safetensors_tensor(&shard, &biases_name)?;
        if weight.dtype != "U32" {
            return Err(format!("weight dtype is {}, expected U32", weight.dtype).into());
        }
        if weight.shape.len() != 2 || scales.shape.len() != 2 || biases.shape.len() != 2 {
            return Err("affine2 tensors must be rank-2".into());
        }
        if scales.dtype != biases.dtype {
            return Err("scales and biases dtypes differ".into());
        }
        if scales.shape != biases.shape {
            return Err("scales and biases shapes differ".into());
        }
        let rows = weight.shape[0];
        let packed_cols = weight.shape[1];
        let cols = packed_cols
            .checked_mul(16)
            .ok_or("packed column count overflow")?;
        if cols % GROUP_SIZE != 0 {
            return Err(format!("K={cols} is not divisible by {GROUP_SIZE}").into());
        }
        let groups_per_row = cols / GROUP_SIZE;
        if scales.shape != [rows, groups_per_row] {
            return Err(format!(
                "scales shape {:?} does not match weight [{rows}, {packed_cols}] (K={cols})",
                scales.shape
            )
            .into());
        }
        let codes = as_u32_le(&weight.bytes)?;
        if codes.len() != rows * packed_cols {
            return Err("uint32 word count does not match weight shape".into());
        }
        let scale_f32 = widen_f16x(&scales.bytes, &scales.dtype)?;
        let bias_f32 = widen_f16x(&biases.bytes, &biases.dtype)?;
        if scale_f32.len() != rows * groups_per_row || bias_f32.len() != rows * groups_per_row {
            return Err("widened scale/bias count does not match groups".into());
        }
        Ok(Affine2Matrix {
            name: stem.to_string(),
            rows,
            cols,
            codes,
            scales: scale_f32,
            biases: bias_f32,
            scale_dtype: scales.dtype,
            shard,
        })
    }

    fn cpu_reconstruct(matrix: &Affine2Matrix) -> Vec<f32> {
        let groups_per_row = matrix.cols / GROUP_SIZE;
        let packed_row = matrix.cols / 16;
        let mut out = vec![0.0f32; matrix.rows * matrix.cols];
        for row in 0..matrix.rows {
            for group in 0..groups_per_row {
                let rgb = row * groups_per_row + group;
                let scale = matrix.scales[rgb];
                let bias = matrix.biases[rgb];
                let word0 = matrix.codes[row * packed_row + 2 * group];
                let word1 = matrix.codes[row * packed_row + 2 * group + 1];
                let col0 = group * GROUP_SIZE;
                for i in 0..16 {
                    let q = ((word0 >> (2 * i)) & 3) as f32;
                    out[row * matrix.cols + col0 + i] = q * scale + bias;
                }
                for i in 0..16 {
                    let q = ((word1 >> (2 * i)) & 3) as f32;
                    out[row * matrix.cols + col0 + 16 + i] = q * scale + bias;
                }
            }
        }
        out
    }

    fn cpu_matvec(weights: &[f32], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows];
        for row in 0..rows {
            let mut sum = 0.0f32;
            let base = row * cols;
            for col in 0..cols {
                sum += weights[base + col] * input[col];
            }
            out[row] = sum;
        }
        out
    }

    fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    fn as_bytes_f32(values: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
    }

    fn as_bytes_u32(values: &[u32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
    }

    fn read_f32(buffer: &metal::Buffer, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, n).to_vec() }
    }

    fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
        encoder.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
    }

    fn deterministic_input(cols: usize) -> Vec<f32> {
        (0..cols)
            .map(|i| (i % 17) as f32 * 0.125 - 1.0)
            .collect()
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let args = parse_args()?;
        let config = read_json(&args.artifact.join("config.json"))?;
        let quant = config.get("quantization").ok_or("config.json missing quantization")?;
        let bits = required_u64(quant, "bits")?;
        let group_size = required_u64(quant, "group_size")?;
        let mode = quant
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("");
        if bits != 2 || group_size != 32 || mode != "affine" {
            return Err(format!(
                "artifact quantization is bits={bits} group_size={group_size} mode={mode}, expected 2/32/affine"
            )
            .into());
        }

        let matrix = load_affine2(&args.artifact, &args.tensor)?;
        let cpu_w = cpu_reconstruct(&matrix);
        let input = deterministic_input(matrix.cols);
        let cpu_y = cpu_matvec(&cpu_w, &input, matrix.rows, matrix.cols);

        let device = Device::system_default().ok_or("no Metal-capable GPU")?;
        let queue = device.new_command_queue();
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(SHADER, &opts)
            .map_err(|e| format!("affine2 shader compile: {e}"))?;

        let dequant_fn = library
            .get_function("affine2_group32_dequant", None)
            .map_err(|e| format!("affine2_group32_dequant: {e}"))?;
        let matvec_fn = library
            .get_function("affine2_group32_matvec", None)
            .map_err(|e| format!("affine2_group32_matvec: {e}"))?;
        let geo_fn = library
            .get_function("affine2_group32_matvec_geo_tpr64_tg128", None)
            .map_err(|e| format!("affine2_group32_matvec_geo_tpr64_tg128: {e}"))?;
        let dequant_pipe = device.new_compute_pipeline_state_with_function(&dequant_fn)?;
        let matvec_pipe = device.new_compute_pipeline_state_with_function(&matvec_fn)?;
        let geo_pipe = device.new_compute_pipeline_state_with_function(&geo_fn)?;

        let codes_buf = device.new_buffer_with_data(
            as_bytes_u32(&matrix.codes).as_ptr() as *const _,
            as_bytes_u32(&matrix.codes).len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let scales_buf = device.new_buffer_with_data(
            as_bytes_f32(&matrix.scales).as_ptr() as *const _,
            as_bytes_f32(&matrix.scales).len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let biases_buf = device.new_buffer_with_data(
            as_bytes_f32(&matrix.biases).as_ptr() as *const _,
            as_bytes_f32(&matrix.biases).len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let input_buf = device.new_buffer_with_data(
            as_bytes_f32(&input).as_ptr() as *const _,
            as_bytes_f32(&input).len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let dequant_buf = device.new_buffer(
            (matrix.rows * matrix.cols * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let y_serial_buf = device.new_buffer(
            (matrix.rows * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let y_geo_buf = device.new_buffer(
            (matrix.rows * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let rows = matrix.rows as u32;
        let cols = matrix.cols as u32;
        let group = GROUP_SIZE as u32;

        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&dequant_pipe);
            enc.set_buffer(0, Some(&codes_buf), 0);
            enc.set_buffer(1, Some(&scales_buf), 0);
            enc.set_buffer(2, Some(&biases_buf), 0);
            enc.set_buffer(3, Some(&dequant_buf), 0);
            set_u32(enc, 4, rows);
            set_u32(enc, 5, cols);
            set_u32(enc, 6, group);
            let n = (matrix.rows * matrix.cols) as u64;
            enc.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&matvec_pipe);
            enc.set_buffer(0, Some(&codes_buf), 0);
            enc.set_buffer(1, Some(&scales_buf), 0);
            enc.set_buffer(2, Some(&biases_buf), 0);
            enc.set_buffer(3, Some(&input_buf), 0);
            enc.set_buffer(4, Some(&y_serial_buf), 0);
            set_u32(enc, 5, rows);
            set_u32(enc, 6, cols);
            set_u32(enc, 7, group);
            enc.dispatch_threads(
                MTLSize::new(rows as u64, 1, 1),
                MTLSize::new(256.min(rows.max(1) as u64), 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&geo_pipe);
            enc.set_buffer(0, Some(&codes_buf), 0);
            enc.set_buffer(1, Some(&scales_buf), 0);
            enc.set_buffer(2, Some(&biases_buf), 0);
            enc.set_buffer(3, Some(&input_buf), 0);
            enc.set_buffer(4, Some(&y_geo_buf), 0);
            set_u32(enc, 5, rows);
            set_u32(enc, 6, cols);
            set_u32(enc, 7, group);
            let groups = (matrix.rows as u64).div_ceil(2);
            enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(128, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let gpu_w = read_f32(&dequant_buf, matrix.rows * matrix.cols);
        let gpu_y = read_f32(&y_serial_buf, matrix.rows);
        let geo_y = read_f32(&y_geo_buf, matrix.rows);
        let max_abs_diff_w = max_abs_diff(&gpu_w, &cpu_w);
        let max_abs_diff_y = max_abs_diff(&gpu_y, &cpu_y);
        let max_abs_diff_geo = max_abs_diff(&geo_y, &cpu_y);

        println!("codec: HGRAVF01 affine_q2_group32");
        println!("artifact: {}", args.artifact.display());
        println!("shard: {}", matrix.shard.display());
        println!("tensor: {}", matrix.name);
        println!("shape: [{}, {}]", matrix.rows, matrix.cols);
        println!("scale_dtype: {}", matrix.scale_dtype);
        println!("groups_per_row: {}", matrix.cols / GROUP_SIZE);
        println!("kernel: affine2_group32_matvec (in-register dequant, no dense W on GEMV)");
        println!("max_abs_diff: {:.6e}", max_abs_diff_w);
        println!("max_abs_diff_matvec: {:.6e}", max_abs_diff_y);
        println!("max_abs_diff_geo_tpr64: {:.6e}", max_abs_diff_geo);
        println!("tolerance: {:.6e}", PASS_TOL);

        if !max_abs_diff_w.is_finite() || max_abs_diff_w >= PASS_TOL {
            return Err(format!(
                "native affine2 reconstruction diverged: max_abs_diff={max_abs_diff_w} >= {PASS_TOL}"
            )
            .into());
        }
        if !max_abs_diff_y.is_finite() || max_abs_diff_y >= PASS_TOL {
            return Err(format!(
                "native affine2 matvec diverged: max_abs_diff_matvec={max_abs_diff_y} >= {PASS_TOL}"
            )
            .into());
        }
        println!("status: PASS");
        Ok(())
    }
}
