//! Generate family documentation, JSON schemas, CLI validation, SDK types,
//! HIDE capability declarations, and Fabric declarations from the registry.
//!
//! Pattern mirrors `hide-sdk-codegen`: pure deterministic strings, checked-in
//! goldens under `generated/`, drift test fails on diff.

use std::path::{Path, PathBuf};

use crate::export::adapter_registry_json;
use crate::registry::builtin_registry;
use crate::support_level::SupportLevel;

/// One generated artifact (relative path under the crate + contents).
#[derive(Debug, Clone)]
pub struct GeneratedArtifact {
    pub relative_path: &'static str,
    pub contents: String,
}

/// Produce every generated artifact in a stable order.
pub fn generate_all() -> Vec<GeneratedArtifact> {
    vec![
        GeneratedArtifact {
            relative_path: "generated/families.md",
            contents: family_docs_md(),
        },
        GeneratedArtifact {
            relative_path: "generated/registry.schema.json",
            contents: registry_json_schema(),
        },
        GeneratedArtifact {
            relative_path: "generated/cli_validate.json",
            contents: cli_validate_json(),
        },
        GeneratedArtifact {
            relative_path: "generated/sdk_types.d.ts",
            contents: sdk_types_ts(),
        },
        GeneratedArtifact {
            relative_path: "generated/hide_capabilities.json",
            contents: hide_capabilities_json(),
        },
        GeneratedArtifact {
            relative_path: "generated/fabric_declarations.json",
            contents: fabric_declarations_json(),
        },
        GeneratedArtifact {
            relative_path: "generated/HAWKING_ADAPTER_REGISTRY.json",
            contents: adapter_registry_json(),
        },
        GeneratedArtifact {
            relative_path: "generated/HAWKING_CANONICAL_EVENTS.json",
            contents: hawking_events::canonical_events_json(),
        },
    ]
}

/// Write all artifacts under `crate_root` (the hawking-adapters package dir).
pub fn write_all(crate_root: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut written = Vec::new();
    for art in generate_all() {
        let path = crate_root.join(art.relative_path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, &art.contents)?;
        written.push(path);
    }
    Ok(written)
}

fn family_docs_md() -> String {
    let r = builtin_registry();
    let mut out = String::from(
        "# Hawking model-family adapter registry\n\n\
         Generated from `hawking-adapters` — do not hand-edit.\n\n\
         **No family is PRODUCTION today.**\n\n\
         | Family | Level | Executes | Serve-registered | Module |\n\
         |---|---|---|---|---|\n",
    );
    for d in r.families() {
        out.push_str(&format!(
            "| {} | {} | {} | {} | `{}` |\n",
            d.display_name,
            d.level.as_str(),
            d.executes,
            d.serve_registered,
            d.module
        ));
    }
    out.push_str("\n## Gaps\n\n");
    for d in r.families() {
        out.push_str(&format!("### {}\n\n", d.id));
        for g in d.gaps {
            out.push_str(&format!("- {g}\n"));
        }
        out.push('\n');
    }
    out
}

fn registry_json_schema() -> String {
    let schema = serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "hawking.adapters.registry.v1",
        "title": "HawkingAdapterRegistry",
        "type": "object",
        "required": ["schema", "families"],
        "properties": {
            "schema": { "const": "hawking.adapters.registry.v1" },
            "families": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "level", "evidence", "module", "executes", "serve_registered", "gaps"],
                    "properties": {
                        "id": { "type": "string" },
                        "level": {
                            "type": "string",
                            "enum": [
                                "DECLARED",
                                "SYNTHETIC_PARITY",
                                "SMALL_REAL_CHECKPOINT",
                                "FULL_PARENT_VALIDATED",
                                "PRODUCTION"
                            ]
                        },
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["path", "claim"],
                                "properties": {
                                    "path": { "type": "string" },
                                    "claim": { "type": "string" }
                                }
                            }
                        },
                        "module": { "type": "string" },
                        "executes": { "type": "boolean" },
                        "serve_registered": { "type": "boolean" },
                        "gaps": {
                            "type": "array",
                            "items": { "type": "string" }
                        }
                    }
                }
            }
        }
    });
    let mut s = serde_json::to_string_pretty(&schema).unwrap();
    s.push('\n');
    s
}

fn cli_validate_json() -> String {
    // Machine-readable rules a CLI can load to reject inflated levels / missing evidence.
    let r = builtin_registry();
    let rules: Vec<_> = r
        .families()
        .map(|d| {
            serde_json::json!({
                "family": d.id,
                "max_level": d.level.as_str(),
                "require_evidence_when_above": SupportLevel::Declared.as_str(),
                "forbid_production": true,
                "executes": d.executes,
                "serve_registered": d.serve_registered,
                "evidence_paths": d.evidence.iter().map(|e| e.path).collect::<Vec<_>>(),
            })
        })
        .collect();
    let doc = serde_json::json!({
        "schema": "hawking.adapters.cli_validate.v1",
        "rules": rules,
        "global": {
            "forbid_production": true,
            "note": "A level asserted without backing evidence must fail validation."
        }
    });
    let mut s = serde_json::to_string_pretty(&doc).unwrap();
    s.push('\n');
    s
}

fn sdk_types_ts() -> String {
    // Mirrors hide-sdk's protocol.d.ts style: deterministic hand-shaped TS from the registry.
    let r = builtin_registry();
    let mut out = String::from(
        "/* Generated by hawking-adapters-codegen — do not hand-edit. */\n\n\
         export type SupportLevel =\n\
           | \"DECLARED\"\n\
           | \"SYNTHETIC_PARITY\"\n\
           | \"SMALL_REAL_CHECKPOINT\"\n\
           | \"FULL_PARENT_VALIDATED\"\n\
           | \"PRODUCTION\";\n\n\
         export type FamilyId =\n",
    );
    let ids: Vec<_> = r.families().map(|d| d.id).collect();
    for (i, id) in ids.iter().enumerate() {
        let sep = if i + 1 == ids.len() { ";" } else { "" };
        out.push_str(&format!("  | \"{id}\"{sep}\n"));
    }
    out.push_str(
        "\nexport interface FamilyEvidence {\n\
         \tpath: string;\n\
         \tclaim: string;\n\
         }\n\n\
         export interface FamilyAdapterEntry {\n\
         \tid: FamilyId;\n\
         \tdisplayName: string;\n\
         \tlevel: SupportLevel;\n\
         \tevidence: FamilyEvidence[];\n\
         \tmodule: string;\n\
         \texecutes: boolean;\n\
         \tserveRegistered: boolean;\n\
         \tgaps: string[];\n\
         }\n\n\
         export const FAMILY_ADAPTERS: FamilyAdapterEntry[] = [\n",
    );
    for d in r.families() {
        out.push_str("  {\n");
        out.push_str(&format!("    id: \"{}\",\n", d.id));
        out.push_str(&format!(
            "    displayName: \"{}\",\n",
            d.display_name.replace('"', "\\\"")
        ));
        out.push_str(&format!("    level: \"{}\",\n", d.level.as_str()));
        out.push_str("    evidence: [\n");
        for e in d.evidence {
            out.push_str(&format!(
                "      {{ path: \"{}\", claim: \"{}\" }},\n",
                e.path,
                e.claim.replace('"', "\\\"")
            ));
        }
        out.push_str("    ],\n");
        out.push_str(&format!("    module: \"{}\",\n", d.module.replace('"', "\\\"")));
        out.push_str(&format!("    executes: {},\n", d.executes));
        out.push_str(&format!("    serveRegistered: {},\n", d.serve_registered));
        out.push_str("    gaps: [\n");
        for g in d.gaps {
            out.push_str(&format!("      \"{}\",\n", g.replace('"', "\\\"")));
        }
        out.push_str("    ],\n");
        out.push_str("  },\n");
    }
    out.push_str("];\n");
    out
}

fn hide_capabilities_json() -> String {
    let r = builtin_registry();
    let caps: Vec<_> = r
        .families()
        .map(|d| {
            serde_json::json!({
                "id": format!("model_family.{}", d.id),
                "kind": "model_family",
                "level": d.level.as_str(),
                "executes": d.executes,
                "serve_registered": d.serve_registered,
                "description": format!("{} — {}", d.display_name, d.level.as_str()),
            })
        })
        .collect();
    let doc = serde_json::json!({
        "schema": "hawking.hide.model_family_capabilities.v1",
        "capabilities": caps,
    });
    let mut s = serde_json::to_string_pretty(&doc).unwrap();
    s.push('\n');
    s
}

fn fabric_declarations_json() -> String {
    // Fabric-facing only: declare event categories + family placement hooks.
    // Do not implement Fabric itself.
    let cats: Vec<_> = hawking_events::all_categories()
        .iter()
        .map(|c| {
            serde_json::json!({
                "category": c.as_str(),
                "kind": hawking_events::kind_for_category(*c),
            })
        })
        .collect();
    let r = builtin_registry();
    let families: Vec<_> = r
        .families()
        .map(|d| {
            serde_json::json!({
                "family": d.id,
                "serve_registered": d.serve_registered,
                "placement": if d.serve_registered { "local_serve_eligible" } else { "not_serve_registered" },
            })
        })
        .collect();
    let doc = serde_json::json!({
        "schema": "hawking.fabric.declarations.v1",
        "note": "Declarations only — Fabric implementation is a parallel lane.",
        "event_categories": cats,
        "family_placement": families,
    });
    let mut s = serde_json::to_string_pretty(&doc).unwrap();
    s.push('\n');
    s
}
