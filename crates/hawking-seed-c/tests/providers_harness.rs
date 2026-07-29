use hawking_seed_c::gravity::Rate;
use hawking_seed_c::pack::{CapabilityKind, Implementation, PackManifest, Profile, SEED_COMPAT};
use hawking_seed_c::providers::adapters::{self, AdapterProvider, Config};
use hawking_seed_c::providers::doctor::{DoctorProvider, SparseResidual};
use hawking_seed_c::providers::forge::{ForgeProvider, TernaryLatentFamily};
use hawking_seed_c::providers::metal::MetalOpProvider;
use hawking_seed_c::providers::provider::{Context, Provider};
use hawking_seed_c::providers::registry::Registry;
use hawking_seed_c::providers::source_decl::SourceRecord;
use hawking_seed_c::providers::validation::{FailurePolicy, TestKind, ValidationCase, ValidationManifest, ValidationProvider};
use hawking_seed_c::providers::verify;
use hawking_seed_c::record::sha256_hex;
use hawking_seed_c::state::{Event, Machine, State};
use std::io::Write;
use std::path::{Path, PathBuf};
fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("nucleus-harness-{}-{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}
fn write(dir: &Path, name: &str, content: &[u8]) -> String {
    std::fs::File::create(dir.join(name)).unwrap().write_all(content).unwrap();
    sha256_hex(content)
}
fn build_and_verify_pack(dir: &Path) -> (PackManifest, PathBuf) {
    let c_adapter = write(dir, "adapter.rs", b"// declarative gpt-oss adapter descriptor\n");
    let c_forge = write(dir, "forge.rs", b"// ternary latent family\n");
    let c_doctor = write(dir, "doctor.rs", b"// sparse residual treatment\n");
    let c_metal = write(dir, "metal.rs", b"// tied logits q8_0 op\n");
    let c_valid = write(dir, "validation.rs", b"// nucleus-core suite\n");
    let man = PackManifest::capability_pack("packs-nucleus-default", "1.0.0", Profile::Default)
        .with_source_commit("7f237ed3")
        .with_offline_cache(&dir.to_string_lossy())
        .with_rollback("git checkout packs-pre-collapse -- .")
        .add_content("adapter.rs", &c_adapter)
        .add_content("forge.rs", &c_forge)
        .add_content("doctor.rs", &c_doctor)
        .add_content("metal.rs", &c_metal)
        .add_content("validation.rs", &c_valid)
        .add_asset("weights", "safetensors", "openai/gpt-oss-120b@b5c939de")
        .add_implementation(Implementation { id: "gpt-oss".into(), kind: CapabilityKind::ModelAdapter, loc: 30, bytes: c_adapter.len(), entry: "adapters::gpt_oss".into(), tests: vec!["gpt-oss-plan".into()] })
        .add_implementation(Implementation { id: "ternary-latent".into(), kind: CapabilityKind::ForgeFamily, loc: 60, bytes: 4096, entry: "forge::TernaryLatentFamily".into(), tests: vec!["subbit".into()] })
        .add_implementation(Implementation { id: "sparse-residual".into(), kind: CapabilityKind::DoctorTreatment, loc: 40, bytes: 3000, entry: "doctor::SparseResidual".into(), tests: vec!["doctor".into()] })
        .add_implementation(Implementation { id: "tied-logits".into(), kind: CapabilityKind::MetalImpl, loc: 90, bytes: 3000, entry: "metal::TiedLogitsOp".into(), tests: vec!["parity".into()] })
        .add_implementation(Implementation { id: "nucleus-core".into(), kind: CapabilityKind::ValidationSuite, loc: 50, bytes: 2000, entry: "validation::nucleus_core".into(), tests: vec![] })
        .add_capability("adapter.gpt_oss", CapabilityKind::ModelAdapter, "gpt-oss")
        .add_capability("forge.ternary_latent", CapabilityKind::ForgeFamily, "ternary-latent")
        .add_capability("doctor.sparse_residual", CapabilityKind::DoctorTreatment, "sparse-residual")
        .add_capability("metal.tied_logits", CapabilityKind::MetalImpl, "tied-logits")
        .add_capability("validation.nucleus-core", CapabilityKind::ValidationSuite, "nucleus-core");
    let mpath = dir.join("manifest.json");
    std::fs::write(&mpath, serde_json::to_string_pretty(&man).unwrap()).unwrap();
    (man, mpath)
}
#[test]
fn full_integration_under_real_hawking_seed() {
    let dir = scratch("full");
    let run_root = dir.join("state");
    let mut machine = Machine::open(&run_root).unwrap();
    machine.apply(Event::Prepare, serde_json::json!({})).unwrap();
    let (man, mpath) = build_and_verify_pack(&dir);
    let report = verify::verify_pack(&man, &mpath, SEED_COMPAT, &[]).unwrap();
    assert!(report.ok(), "pack must verify: {report:?}");
    assert_eq!(verify::offline_hydrate(&man, &mpath).unwrap(), 5, "offline hydration reconstitutes all 5 files");
    machine.record(verify::seal_receipt(&report).unwrap()).unwrap();
    machine.apply(Event::Admit, serde_json::json!({"verified": report.pack})).unwrap();
    machine.apply(Event::Run, serde_json::json!({})).unwrap();
    let mut reg = Registry::new();
    reg.insert(&man).unwrap();
    let mut admissions = Vec::new();
    for cap in ["adapter.gpt_oss", "forge.ternary_latent", "doctor.sparse_residual", "metal.tied_logits", "validation.nucleus-core"] {
        let sel = reg.activate_sole(cap).unwrap();
        assert_eq!(sel.hawking_compat, SEED_COMPAT);
        assert!(!sel.reason.is_empty());
        admissions.push(sel.admission_receipt());
    }
    assert_eq!(reg.active_selections().len(), 5);
    for rec in admissions {
        machine.record(rec).unwrap();
    }
    let source = SourceRecord::hf("openai/gpt-oss-120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a")
        .with_format("safetensors")
        .with_tensor_type("MXFP4");
    let ctx = Context::new("harness-run", &run_root.to_string_lossy(), source, Rate::new(4, 5));
    let gpt = AdapterProvider::new(adapters::gpt_oss());
    let cfg = Config { n_layers: 2, hidden: 64, n_ff: 128, n_heads: 4, n_kv_heads: 2, head_dim: 16, vocab: 100, rms_eps: 1e-5, rope_base: 10000.0, quant: "MXFP4".into() };
    let plan = adapters::gpt_oss().build_plan(&cfg).unwrap();
    assert!(plan.ops.len() > 10, "adapter emits a Seed IR Plan");
    assert!(adapters::gpt_oss().moe_ops(&cfg).is_some(), "gpt-oss emits MoE ops");
    let out = gpt.run(&ctx, serde_json::to_value(&cfg).unwrap()).unwrap();
    assert!(out.evidence.verify().is_ok() && out.evidence.kind == "evaluation");
    machine.record(out.evidence).unwrap();
    let forge = ForgeProvider::new(TernaryLatentFamily::new(32));
    let out = forge.run(&ctx, serde_json::json!({"rows": 256, "cols": 256})).unwrap();
    assert!(out.metrics["subbit"].as_bool().unwrap(), "forge produces a sub-bit artifact");
    machine.record(out.evidence).unwrap();
    let doctor = DoctorProvider::new(SparseResidual);
    let out = doctor.run(&ctx, serde_json::json!({"rows": 256, "cols": 256, "budget_bpw": 0.99})).unwrap();
    assert!(out.metrics["within_budget"].as_bool().unwrap(), "doctor stays within the physical budget");
    assert!(out.metrics["improved"].as_bool().unwrap(), "doctor reduces divergence");
    machine.record(out.evidence).unwrap();
    let metal = MetalOpProvider::default();
    let out = metal.run(&ctx, serde_json::json!({"vocab": 256, "hidden": 64})).unwrap();
    assert!(out.metrics["cpu_metal_parity"].as_bool().unwrap(), "metal op matches the CPU reference");
    machine.record(out.evidence).unwrap();
    let suite = ValidationManifest::new("nucleus-core").add(ValidationCase {
        id: "subbit-property".into(), kind: TestKind::Property, fixture: String::new(), source_identity: String::new(),
        required_packs: vec!["packs-nucleus-default".into()], expected: "whole_bpw < 1.0".into(), tolerance: 0.0,
        resource_class: "cpu".into(), failure_policy: FailurePolicy::FailClosed,
    });
    let vp = ValidationProvider::new(suite);
    let out = vp.run(&ctx, serde_json::json!({"available": ["packs-nucleus-default"]})).unwrap();
    assert!(out.metrics["green"].as_bool().unwrap(), "validation suite green");
    machine.record(out.evidence).unwrap();
    machine.apply(Event::Evaluate, serde_json::json!({})).unwrap();
    machine.apply(Event::Seal, serde_json::json!({})).unwrap();
    let resumed = Machine::open(&run_root).unwrap();
    assert_eq!(resumed.state, State::Sealed, "the Seed's controller reaches Sealed");
    assert!(resumed.log.iter().all(|r| r.verify().is_ok()), "every evidence record is sealed + untampered");
    let admits = resumed.log.iter().filter(|r| r.kind == "admission").count();
    assert_eq!(admits, 5, "one sealed admission record per activation (got {admits})");
    let evals = resumed.log.iter().filter(|r| r.kind == "evaluation").count();
    assert!(evals >= 5, "the Seed received evidence from all 5 providers (got {evals})");
}
#[test]
fn tamper_refusal_and_rollback_under_seed() {
    let dir = scratch("tamper");
    let (man, mpath) = build_and_verify_pack(&dir);
    assert!(verify::verify_pack(&man, &mpath, SEED_COMPAT, &[]).unwrap().ok());
    std::fs::write(dir.join("forge.rs"), b"// TAMPERED\n").unwrap();
    let r = verify::verify_pack(&man, &mpath, SEED_COMPAT, &[]).unwrap();
    assert!(r.tamper_detected && !r.ok(), "tamper must be refused");
    std::fs::write(dir.join("forge.rs"), b"// ternary latent family\n").unwrap();
    assert!(verify::verify_pack(&man, &mpath, SEED_COMPAT, &[]).unwrap().ok(), "rollback restores green");
}
#[test]
fn provider_cannot_mutate_hawking_state() {
    let dir = scratch("nomutate");
    let run_root = dir.join("state");
    let mut machine = Machine::open(&run_root).unwrap();
    for ev in [Event::Prepare, Event::Admit, Event::Run] {
        machine.apply(ev, serde_json::json!({})).unwrap();
    }
    let before = machine.log.len();
    let ctx = Context::new("r", &run_root.to_string_lossy(), SourceRecord::local("/tmp"), Rate::new(4, 5));
    let _ = ForgeProvider::new(TernaryLatentFamily::new(32))
        .run(&ctx, serde_json::json!({"rows": 64, "cols": 64}))
        .unwrap();
    let reopened = Machine::open(&run_root).unwrap();
    assert_eq!(reopened.log.len(), before, "provider must not mutate Seed state");
}
