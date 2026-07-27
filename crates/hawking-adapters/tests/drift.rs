//! Drift test: regenerating schemas/types produces no diff against goldens.

use std::path::PathBuf;

use hawking_adapters::generate::generate_all;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn generated_artifacts_match_checked_in() {
    let root = crate_root();
    let mut failures = Vec::new();
    for art in generate_all() {
        let path = root.join(art.relative_path);
        if !path.exists() {
            failures.push(format!(
                "missing checked-in artifact {} — run: cargo run -p hawking-adapters --bin hawking-adapters-codegen",
                art.relative_path
            ));
            continue;
        }
        let on_disk = std::fs::read_to_string(&path).expect("read golden");
        if on_disk != art.contents {
            failures.push(format!(
                "drift in {} ({} bytes on disk vs {} generated) — re-run hawking-adapters-codegen",
                art.relative_path,
                on_disk.len(),
                art.contents.len()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "generated artifact drift:\n{}",
        failures.join("\n")
    );
}
