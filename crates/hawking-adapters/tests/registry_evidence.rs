//! Registry honesty: every family's support level is backed by named evidence.
//! A level asserted without a present evidence path fails this test.

use hawking_adapters::evidence::{validate_family_evidence, workspace_root};
use hawking_adapters::registry::builtin_registry;
use hawking_adapters::support_level::SupportLevel;
use hawking_adapters::{Evidence, FamilyDescriptor};

#[test]
fn every_family_evidence_present() {
    let r = builtin_registry();
    r.validate_all_evidence()
        .unwrap_or_else(|errs| panic!("evidence failures:\n{}", errs.join("\n")));
}

#[test]
fn level_without_evidence_fails() {
    let root = workspace_root();
    let bogus = FamilyDescriptor {
        id: "bogus",
        display_name: "Bogus",
        level: SupportLevel::SyntheticParity,
        evidence: &[], // empty — must fail
        module: "nowhere",
        executes: false,
        serve_registered: false,
        gaps: &[],
    };
    let err = validate_family_evidence(&root, &bogus).unwrap_err();
    assert!(
        err.contains("requires at least one named evidence"),
        "got: {err}"
    );
}

#[test]
fn production_always_fails() {
    let root = workspace_root();
    static EV: &[Evidence] = &[Evidence {
        path: "Cargo.toml",
        claim: "exists but PRODUCTION is still forbidden",
    }];
    let prod = FamilyDescriptor {
        id: "fake_prod",
        display_name: "Fake",
        level: SupportLevel::Production,
        evidence: EV,
        module: "x",
        executes: true,
        serve_registered: true,
        gaps: &[],
    };
    let err = validate_family_evidence(&root, &prod).unwrap_err();
    assert!(err.contains("PRODUCTION"), "got: {err}");
}

#[test]
fn missing_evidence_path_fails() {
    let root = workspace_root();
    static EV: &[Evidence] = &[Evidence {
        path: "this/path/does/not/exist.rs",
        claim: "phantom",
    }];
    let d = FamilyDescriptor {
        id: "missing",
        display_name: "Missing",
        level: SupportLevel::SmallRealCheckpoint,
        evidence: EV,
        module: "x",
        executes: false,
        serve_registered: false,
        gaps: &[],
    };
    let err = validate_family_evidence(&root, &d).unwrap_err();
    assert!(err.contains("does not exist"), "got: {err}");
}
