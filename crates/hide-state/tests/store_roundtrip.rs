use hide_state::{
    Ancestry, Capsule, CapsuleBuilder, CapsuleStore, CapsuleType, DiskStore, IdentityBinding,
    IntegrityAlgo, MemoryStore,
};
fn identity() -> IdentityBinding {
    IdentityBinding {
        model_weights_id: "weights".into(),
        arch_id: "arch".into(),
        tokenizer_id: "tok".into(),
        prompt_abi_version: "abi-1".into(),
        tool_registry_id: "reg".into(),
        engine_build_id: "build".into(),
        security_domain: "domain".into(),
    }
}
fn synthetic(seed: u8, len: usize, algo: IntegrityAlgo) -> Capsule {
    let payload: Vec<u8> = (0..len).map(|i| ((i as u8).wrapping_add(seed))).collect();
    CapsuleBuilder::new(CapsuleType::Kv, "model-fixture", identity())
        .runtime_version("rt-1")
        .dtype("f16")
        .device("cpu")
        .position(seed as u64)
        .context_pack_hash("ctx")
        .integrity_algo(algo)
        .seal(payload)
}
fn exercise_store<S: CapsuleStore>(store: &mut S) {
    let capsule = synthetic(3, 512, IntegrityAlgo::Blake3);
    let payload = capsule.payload().to_vec();
    let id = store.save(&capsule).unwrap();
    let loaded = store.load(&id).unwrap();
    assert_eq!(loaded, capsule);
    assert_eq!(loaded.payload(), payload.as_slice());
    let meta = store.inspect(&id).unwrap();
    assert_eq!(meta.header, *capsule.header());
    assert_eq!(meta.header.bytes, payload.len() as u64);
    let child_id = store.fork(&id).unwrap();
    assert_ne!(child_id, id);
    let child = store.load(&child_id).unwrap();
    assert_eq!(child.payload(), payload.as_slice());
    assert_eq!(child.parent_capsule_id(), Some(&id));
    let cmp = store.compare(&id, &child_id).unwrap();
    assert_eq!(cmp.ancestry, Ancestry::ParentToChild);
    assert!(cmp.payload_identical);
    assert!(cmp.identity_identical);
    assert!(!cmp.same_capsule_id);
    assert!(!cmp.header_identical); // ids and ancestry differ
    let cmp_self = store.compare(&id, &id).unwrap();
    assert_eq!(cmp_self.ancestry, Ancestry::Same);
    assert!(cmp_self.header_identical);
    store.release(&id).unwrap();
    assert!(store.load(&id).is_err());
    assert!(store.load(&child_id).is_ok());
    assert!(store.release(&id).is_err());
}
#[test]
fn memory_store_full_surface() {
    let mut store = MemoryStore::new();
    exercise_store(&mut store);
}
#[test]
fn disk_store_full_surface() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = DiskStore::open(dir.path()).unwrap();
    exercise_store(&mut store);
}
#[test]
fn disk_store_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let id = {
        let mut store = DiskStore::open(dir.path()).unwrap();
        let capsule = synthetic(7, 256, IntegrityAlgo::Sha256);
        store.save(&capsule).unwrap()
    };
    let store = DiskStore::open(dir.path()).unwrap();
    let loaded = store.load(&id).unwrap();
    assert_eq!(loaded.header().integrity.algo, IntegrityAlgo::Sha256);
 assert!(loaded .header() .integrity .verify(loaded.payload()));
}
#[test]
fn disk_store_rejects_corrupted_object() {
    use std::fs;
    let dir = tempfile::tempdir().unwrap();
    let mut store = DiskStore::open(dir.path()).unwrap();
    let capsule = synthetic(1, 128, IntegrityAlgo::Blake3);
    let id = store.save(&capsule).unwrap();
    let objects = dir.path().join("objects");
    let object_file = fs::read_dir(&objects)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().map(|x| x == "capsule").unwrap_or(false))
        .expect("object file present");
    let mut bytes = fs::read(&object_file).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    fs::write(&object_file, &bytes).unwrap();
    assert!(store.load(&id).is_err());
}
#[test]
fn disk_store_deduplicates_identical_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = DiskStore::open(dir.path()).unwrap();
    let capsule = synthetic(5, 100, IntegrityAlgo::Blake3);
    let id = store.save(&capsule).unwrap();
    let child_id = store.fork(&id).unwrap();
    let objects = dir.path().join("objects");
    let count = std::fs::read_dir(&objects).unwrap().count();
    assert_eq!(count, 2);
    store.release(&id).unwrap();
    let count_after = std::fs::read_dir(&objects).unwrap().count();
    assert_eq!(count_after, 1);
    assert!(store.load(&child_id).is_ok());
}
