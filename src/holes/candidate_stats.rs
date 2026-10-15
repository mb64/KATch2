//! Per-guess SPP DAG measurements; ordered maps avoid consuming hash seeds.
use crate::spp::{SPP, SPPstore};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Write},
};

pub(super) struct CandidateStats {
    guess: usize,
    writer: Option<BufWriter<File>>,
}
impl CandidateStats {
    pub(super) fn new() -> Self {
        let writer = crate::experiment::candidate_stats_path().map(|path| {
            let mut writer = BufWriter::new(File::create(path).expect("create candidate-size log"));
            writeln!(writer,"guess,slot,root,internal_nodes,terminal_nodes,total_nodes,accepted_pairs,positive_examples,negative_examples,forbidden_nodes,example_selection_ns,learning_ns,input_positive_nodes,input_negative_nodes,input_shared_nodes,input_trie_nodes,merged_dag_nodes").unwrap();
            writer
        });
        Self { guess: 0, writer }
    }
    pub(super) fn enabled(&self) -> bool {
        self.writer.is_some()
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record(
        &mut self,
        slot: u32,
        candidate: SPP,
        forbidden: Option<SPP>,
        store: &SPPstore,
        positive: usize,
        negative: usize,
        selection_ns: u128,
        learning_ns: u128,
    ) {
        let Some(writer) = &mut self.writer else {
            return;
        };
        let mut memo = BTreeMap::new();
        let pairs = count(candidate, store, &mut memo);
        let terminals = memo.keys().filter(|s| s.as_u32() <= 1).count();
        let nodes = memo.len();
        let forbidden_nodes = forbidden.map_or(0, |s| {
            let mut m = BTreeMap::new();
            count(s, store, &mut m);
            m.len()
        });
        let [ip, ine, shared, trie, merged] = crate::spp::learner::input_sizes();
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.guess,
            slot,
            candidate.as_u32(),
            nodes - terminals,
            terminals,
            nodes,
            pairs,
            positive,
            negative,
            forbidden_nodes,
            selection_ns,
            learning_ns,
            ip,
            ine,
            shared,
            trie,
            merged
        )
        .unwrap();
        if self.guess.is_multiple_of(256) {
            writer.flush().unwrap();
        }
    }
    pub(super) fn next(&mut self) {
        self.guess += 1;
    }
}
fn count(s: SPP, store: &SPPstore, memo: &mut BTreeMap<SPP, u128>) -> u128 {
    if let Some(&n) = memo.get(&s) {
        return n;
    }
    let n = if s.as_u32() <= 1 {
        s.as_u32() as u128
    } else {
        let x = store.get(s);
        [x.x00, x.x01, x.x10, x.x11]
            .into_iter()
            .fold(0u128, |n, c| n.saturating_add(count(c, store, memo)))
    };
    memo.insert(s, n);
    n
}
