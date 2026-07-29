use std::fs;
use std::path::{Path, PathBuf};
const FORBIDDEN_CLAIMS: &[&str] = &[
    "faster",
    "speedup",
    "speed-up",
    "outperform",
    "lower latency",
    "higher throughput",
    "tok/s",
    "tokens per second",
    "better quality",
    "quality improvement",
    "improves quality",
    "more accurate",
    "state-of-the-art",
    "sota",
];
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().map(|e| e == "rs").unwrap_or(false) {
            out.push(path);
        }
    }
}
#[test]
fn source_makes_no_performance_or_quality_claim() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    collect_rs_files(&src, &mut files);
    assert!(!files.is_empty(), "expected source files under src/");
    let mut violations = Vec::new();
    for file in &files {
        let text = fs::read_to_string(file).unwrap().to_lowercase();
        for claim in FORBIDDEN_CLAIMS {
            if text.contains(claim) {
                violations.push(format!("{}: forbidden claim {:?}", file.display(), claim));
            }
        }
    }
    assert!(violations.is_empty());
}
