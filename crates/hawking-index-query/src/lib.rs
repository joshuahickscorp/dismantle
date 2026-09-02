//! JSON query surface over the hawking-index Python grammar.
//!
//! Lane r1 (reachability) and lane r2 (roadmap auditor) both need a once-built
//! symbol/import/call index of git-tracked Python. This crate walks the same
//! `tree-sitter-python` grammar `hawking-index` compiles in
//! (`GrammarRegistry::bundle(LangId::Python)`). It does not add a second parser.
//!
//! Wire format (`hawking.index.python_facts.v1`) is the merge contract if r1
//! later inlines the same CLI into the `hawking-index` binary.

pub mod python_facts;

pub use python_facts::{
    dump_python_facts_from_overlay, dump_python_facts_git_head, extract_python_facts,
    extract_python_facts_many, read_overlay_ndjson, CallFact, DefFact, ImportFact, ImportedName,
    NameUseFact, PythonFactsDump, PythonFileFacts, SubprocessLitFact, PYTHON_FACTS_SCHEMA,
};
