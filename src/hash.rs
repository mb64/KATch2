//! Fast non-cryptographic hash maps and sets (FxHash, via `rustc-hash`).
//!
//! FxHash beats ahash on our small integer / tuple-of-integer keys, and its
//! fixed seed makes iteration order deterministic across runs.

pub type HashMap<K, V> = rustc_hash::FxHashMap<K, V>;
pub type HashSet<K> = rustc_hash::FxHashSet<K>;

/// `new` / `with_capacity` constructors for [`HashMap`].
pub trait HashMapExt {
    fn new() -> Self;
    fn with_capacity(capacity: usize) -> Self;
}

/// `new` / `with_capacity` constructors for [`HashSet`].
pub trait HashSetExt {
    fn new() -> Self;
    fn with_capacity(capacity: usize) -> Self;
}

impl<K, V> HashMapExt for HashMap<K, V> {
    fn new() -> Self {
        Self::default()
    }

    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, Default::default())
    }
}

impl<K> HashSetExt for HashSet<K> {
    fn new() -> Self {
        Self::default()
    }

    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, Default::default())
    }
}
