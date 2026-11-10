//! Symbolic evidence over the complete Cand predicate, kept as shared DAGs.
use super::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Operation {
    Union,
    Intersect,
    Difference,
}

impl<'a> Cand<'a> {
    /// Project a fixed internal trace's feasible boundary pairs through this
    /// predicate. Correlations remain enforced by the predicate in the learner.
    pub fn boundary_packets(
        &self,
        store: &mut spp::SPPstore,
        first: &[bool],
        states: &[usize],
        last: &[bool],
        incoming: crate::sp::SP,
        outgoing: crate::sp::SP,
    ) -> (crate::sp::SP, crate::sp::SP) {
        use crate::sp::SP;
        let last = store.sp.singleton(last);
        let last = store.ibwd(last);
        let outgoing_relation = store.ifwd(outgoing);
        let mask = store.intersect(last, outgoing_relation);
        fn visit(
            cand: &Cand<'_>,
            id: CandIdx,
            depth: usize,
            input: SP,
            first: &[bool],
            states: &[usize],
            mask: spp::SPP,
            store: &mut spp::SPPstore,
            memo: &mut HashMap<(CandIdx, usize, SP), (SP, SP)>,
        ) -> (SP, SP) {
            if let Some(&result) = memo.get(&(id, depth, input)) {
                return result;
            }
            let result = match &cand.nodes[id as usize] {
                CandNode::Root(r) => {
                    if input == SP::new(0) {
                        (SP::new(0), store.sp.zero)
                    } else {
                        let r = store.intersect(*r, mask);
                        let post = store.fwd(r);
                        (SP::new(u32::from(!store.sp.is_zero(post))), post)
                    }
                }
                CandNode::NextState(children) => visit(
                    cand,
                    children[states[depth - first.len()]],
                    depth + 1,
                    input,
                    first,
                    states,
                    mask,
                    store,
                    memo,
                ),
                CandNode::NextField { b00, b01, b10, b11 } => {
                    let inputs = store.sp.get(input);
                    let (a, b) = if first[depth] {
                        (*b01, *b11)
                    } else {
                        (*b00, *b10)
                    };
                    let (p0, q0) = visit(
                        cand,
                        a,
                        depth + 1,
                        inputs.x0,
                        first,
                        states,
                        mask,
                        store,
                        memo,
                    );
                    let (p1, q1) = visit(
                        cand,
                        b,
                        depth + 1,
                        inputs.x1,
                        first,
                        states,
                        mask,
                        store,
                        memo,
                    );
                    (store.sp.mk(p0, p1), store.sp.union(q0, q1))
                }
            };
            memo.insert((id, depth, input), result);
            result
        }
        visit(
            self,
            self.head,
            0,
            incoming,
            first,
            states,
            mask,
            store,
            &mut HashMap::new(),
        )
    }

    pub fn constant(store: &mut spp::SPPstore, dfa: &'a ExplicitDFA, value: bool) -> Self {
        let relation = if value { store.top } else { store.zero };
        Self::product(store, dfa, relation, None, relation)
    }

    /// `(in,start) ∈ incoming ∧ states = vector ∧ (end,out) ∈ outgoing`.
    /// `None` accepts every state vector. No concrete packets are enumerated.
    pub fn product(
        store: &mut spp::SPPstore,
        dfa: &'a ExplicitDFA,
        incoming: spp::SPP,
        states: Option<&[usize]>,
        outgoing: spp::SPP,
    ) -> Self {
        let ns = dfa.num_states();
        if let Some(states) = states {
            assert_eq!(states.len(), ns);
            assert!(states.iter().all(|&q| q < ns));
        }
        let mut builder = Builder::default();
        let mut yes = builder.mk(CandNode::Root(outgoing));
        let mut no = builder.mk(CandNode::Root(store.zero));
        for i in (0..ns).rev() {
            let mut children = vec![yes; ns];
            if let Some(states) = states {
                children.fill(no);
                children[states[i]] = yes;
            }
            yes = builder.mk(CandNode::NextState(children));
            no = builder.mk(CandNode::NextState(vec![no; ns]));
        }
        fn fields(
            builder: &mut Builder,
            store: &spp::SPPstore,
            relation: spp::SPP,
            depth: usize,
            yes: CandIdx,
            no: CandIdx,
            memo: &mut HashMap<(spp::SPP, usize), CandIdx>,
        ) -> CandIdx {
            if depth == 0 {
                return if relation == spp::SPP::new(1) {
                    yes
                } else {
                    no
                };
            }
            if let Some(&node) = memo.get(&(relation, depth)) {
                return node;
            }
            let r = store.get(relation);
            let [b00, b01, b10, b11] = [r.x00, r.x01, r.x10, r.x11]
                .map(|r| fields(builder, store, r, depth - 1, yes, no, memo));
            let node = builder.mk(CandNode::NextField { b00, b01, b10, b11 });
            memo.insert((relation, depth), node);
            node
        }
        let head = fields(
            &mut builder,
            store,
            incoming,
            store.num_vars() as usize,
            yes,
            no,
            &mut HashMap::new(),
        );
        Self {
            nodes: builder.nodes.into(),
            head,
            dfa,
        }
    }

    pub fn singleton(store: &mut spp::SPPstore, dfa: &'a ExplicitDFA, input: &Input) -> Self {
        let incoming = pair(store, &input.pkt_in, &input.pkt_start);
        let outgoing = pair(store, &input.pkt_end, &input.pkt_out);
        Self::product(store, dfa, incoming, Some(&input.states), outgoing)
    }

    pub fn union(&self, other: &Self, store: &mut spp::SPPstore) -> Self {
        self.combine(other, store, Operation::Union)
    }
    pub fn intersect(&self, other: &Self, store: &mut spp::SPPstore) -> Self {
        self.combine(other, store, Operation::Intersect)
    }
    pub fn difference(&self, other: &Self, store: &mut spp::SPPstore) -> Self {
        self.combine(other, store, Operation::Difference)
    }
    fn combine(&self, other: &Self, store: &mut spp::SPPstore, op: Operation) -> Self {
        assert!(
            std::ptr::eq(self.dfa, other.dfa),
            "Cand bounds must share their reference DFA"
        );
        let mut arena = Arena::default();
        let a = arena.import(self);
        let b = arena.import(other);
        let head = arena.apply(a, b, op, store);
        arena.builder.finish(head, self.dfa)
    }

    pub fn is_empty(&self, store: &spp::SPPstore) -> bool {
        let mut seen = vec![false; self.nodes.len()];
        let mut todo = vec![self.head];
        while let Some(id) = todo.pop() {
            if std::mem::replace(&mut seen[id as usize], true) {
                continue;
            }
            match &self.nodes[id as usize] {
                CandNode::Root(r) => {
                    if !store.is_zero(*r) {
                        return false;
                    }
                }
                node => todo.extend(children(node)),
            }
        }
        true
    }

    /// Learn from shared positive/negative diagrams. Prefix cells are merged
    /// only if their entire symbolic suffix evidence is compatible. At each
    /// output root the SPP learner receives whole required/forbidden relations.
    pub fn from_examples_with_bounds(
        store: &mut spp::SPPstore,
        dfa: &'a ExplicitDFA,
        examples: &[(Input, bool)],
        required: &Self,
        forbidden: &Self,
    ) -> Result<Self, ConflictError> {
        let mut arena = Arena::default();
        assert!(std::ptr::eq(dfa, required.dfa) && std::ptr::eq(dfa, forbidden.dfa));
        let mut p = arena.import(required);
        let mut n = arena.import(forbidden);
        for (input, positive) in examples {
            let singleton = Self::singleton(store, dfa, input);
            let s = arena.import(&singleton);
            if *positive {
                p = arena.apply(p, s, Operation::Union, store);
            } else {
                n = arena.apply(n, s, Operation::Union, store);
            }
        }
        if !arena.disjoint(p, n, store) {
            return Err(ConflictError);
        }
        let nv = store.num_vars() as usize;
        let ns = dfa.num_states();
        let mut cells = vec![(p, n, Vec::<Option<usize>>::new())];
        let mut levels = vec![vec![0]];
        for depth in 0..nv + ns {
            let mut classes = Vec::<usize>::new();
            let mut assigned = HashMap::new();
            for &parent in &levels[depth] {
                let (p, n, _) = &cells[parent];
                let pc = children(&arena.builder.nodes[*p as usize]);
                let nc = children(&arena.builder.nodes[*n as usize]);
                let mut row = Vec::new();
                for (p, n) in pc.into_iter().zip(nc) {
                    if arena.empty(p, store) && arena.empty(n, store) {
                        row.push(None);
                        continue;
                    }
                    let id = if let Some(&id) = assigned.get(&(p, n)) {
                        id
                    } else {
                        let target = classes.iter().copied().find(|&id| {
                            arena.disjoint(p, cells[id].1, store)
                                && arena.disjoint(cells[id].0, n, store)
                        });
                        let id = if let Some(id) = target {
                            cells[id].0 = arena.apply(cells[id].0, p, Operation::Union, store);
                            cells[id].1 = arena.apply(cells[id].1, n, Operation::Union, store);
                            id
                        } else {
                            let id = cells.len();
                            cells.push((p, n, Vec::new()));
                            classes.push(id);
                            id
                        };
                        assigned.insert((p, n), id);
                        id
                    };
                    row.push(Some(id));
                }
                cells[parent].2 = row;
            }
            levels.push(classes);
        }
        let mut result = Builder::default();
        let mut done = vec![0; cells.len()];
        // A completely unconstrained suffix may occur only for empty evidence.
        if levels.last().unwrap().is_empty() {
            return Ok(Self::constant(store, dfa, true));
        }
        for &id in levels.last().unwrap() {
            let CandNode::Root(p) = arena.builder.nodes[cells[id].0 as usize] else {
                unreachable!()
            };
            let CandNode::Root(n) = arena.builder.nodes[cells[id].1 as usize] else {
                unreachable!()
            };
            let r =
                spp::learner::learn_spp_from_relations(p, n, store).map_err(|_| ConflictError)?;
            done[id] = result.mk(CandNode::Root(r));
        }
        for depth in (0..nv + ns).rev() {
            for &id in &levels[depth] {
                let row = &cells[id].2;
                let fill = done[row.iter().flatten().next().copied().unwrap()];
                let row: Vec<_> = row.iter().map(|x| x.map_or(fill, |i| done[i])).collect();
                let node = if depth < nv {
                    CandNode::NextField {
                        b00: row[0],
                        b01: row[1],
                        b10: row[2],
                        b11: row[3],
                    }
                } else {
                    CandNode::NextState(row)
                };
                done[id] = result.mk(node);
            }
        }
        Ok(Self {
            nodes: result.nodes.into(),
            head: done[0],
            dfa,
        })
    }
}

fn pair(store: &mut spp::SPPstore, a: &[bool], b: &[bool]) -> spp::SPP {
    let a = store.sp.singleton(a);
    let b = store.sp.singleton(b);
    let a = store.ibwd(a);
    let b = store.ifwd(b);
    store.intersect(a, b)
}

fn children(node: &CandNode) -> Vec<CandIdx> {
    match node {
        CandNode::NextField { b00, b01, b10, b11 } => vec![*b00, *b01, *b10, *b11],
        CandNode::NextState(xs) => xs.clone(),
        CandNode::Root(_) => unreachable!("root has no Cand children"),
    }
}

impl Builder {
    /// Retain only the resulting predicate, not input diagrams or intermediate
    /// Boolean-operation nodes. Cloning the finished immutable arena is cheap.
    fn finish(self, head: CandIdx, dfa: &ExplicitDFA) -> Cand<'_> {
        fn copy(
            id: CandIdx,
            source: &[CandNode],
            target: &mut Builder,
            memo: &mut HashMap<CandIdx, CandIdx>,
        ) -> CandIdx {
            if let Some(&id) = memo.get(&id) {
                return id;
            }
            let node = match &source[id as usize] {
                CandNode::Root(r) => CandNode::Root(*r),
                CandNode::NextState(xs) => {
                    CandNode::NextState(xs.iter().map(|&x| copy(x, source, target, memo)).collect())
                }
                CandNode::NextField { b00, b01, b10, b11 } => {
                    let [b00, b01, b10, b11] =
                        [b00, b01, b10, b11].map(|&x| copy(x, source, target, memo));
                    CandNode::NextField { b00, b01, b10, b11 }
                }
            };
            let result = target.mk(node);
            memo.insert(id, result);
            result
        }
        let mut target = Builder::default();
        let head = copy(head, &self.nodes, &mut target, &mut HashMap::new());
        Cand {
            nodes: target.nodes.into(),
            head,
            dfa,
        }
    }
}

#[derive(Default)]
struct Arena {
    builder: Builder,
    memo: HashMap<(CandIdx, CandIdx, Operation), CandIdx>,
    empty: HashMap<CandIdx, bool>,
}
impl Arena {
    fn import(&mut self, cand: &Cand<'_>) -> CandIdx {
        // Builders always allocate children before parents.
        let mut ids = Vec::new();
        for node in cand.nodes.iter() {
            let node = match node {
                CandNode::Root(r) => CandNode::Root(*r),
                CandNode::NextState(xs) => {
                    CandNode::NextState(xs.iter().map(|&i| ids[i as usize]).collect())
                }
                CandNode::NextField { b00, b01, b10, b11 } => CandNode::NextField {
                    b00: ids[*b00 as usize],
                    b01: ids[*b01 as usize],
                    b10: ids[*b10 as usize],
                    b11: ids[*b11 as usize],
                },
            };
            ids.push(self.builder.mk(node));
        }
        ids[cand.head as usize]
    }
    fn apply(
        &mut self,
        a: CandIdx,
        b: CandIdx,
        op: Operation,
        store: &mut spp::SPPstore,
    ) -> CandIdx {
        if let Some(&id) = self.memo.get(&(a, b, op)) {
            return id;
        }
        let x = self.builder.nodes[a as usize].clone();
        let y = self.builder.nodes[b as usize].clone();
        let node = match (x, y) {
            (CandNode::Root(a), CandNode::Root(b)) => CandNode::Root(match op {
                Operation::Union => store.union(a, b),
                Operation::Intersect => store.intersect(a, b),
                Operation::Difference => store.difference(a, b),
            }),
            (x @ CandNode::NextField { .. }, y @ CandNode::NextField { .. }) => {
                let row: Vec<_> = children(&x)
                    .into_iter()
                    .zip(children(&y))
                    .map(|(a, b)| self.apply(a, b, op, store))
                    .collect();
                CandNode::NextField {
                    b00: row[0],
                    b01: row[1],
                    b10: row[2],
                    b11: row[3],
                }
            }
            (CandNode::NextState(x), CandNode::NextState(y)) => CandNode::NextState(
                x.into_iter()
                    .zip(y)
                    .map(|(a, b)| self.apply(a, b, op, store))
                    .collect(),
            ),
            _ => panic!("Cand diagram levels must agree"),
        };
        let id = self.builder.mk(node);
        self.memo.insert((a, b, op), id);
        id
    }
    fn empty(&mut self, id: CandIdx, store: &spp::SPPstore) -> bool {
        if let Some(&empty) = self.empty.get(&id) {
            return empty;
        }
        let node = self.builder.nodes[id as usize].clone();
        let empty = match node {
            CandNode::Root(r) => store.is_zero(r),
            node => children(&node).into_iter().all(|i| self.empty(i, store)),
        };
        self.empty.insert(id, empty);
        empty
    }
    fn disjoint(&mut self, a: CandIdx, b: CandIdx, store: &mut spp::SPPstore) -> bool {
        let intersection = self.apply(a, b, Operation::Intersect, store);
        self.empty(intersection, store)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> Vec<Input> {
        (0..64)
            .map(|i| Input {
                pkt_in: vec![i & 1 != 0],
                pkt_start: vec![i & 2 != 0],
                states: vec![(i >> 4) & 1, (i >> 5) & 1],
                pkt_end: vec![i & 4 != 0],
                pkt_out: vec![i & 8 != 0],
            })
            .collect()
    }

    #[test]
    fn symbolic_learning_and_boolean_operations_match_finite_truth_tables() {
        let mut store = spp::SPPstore::new(1);
        let dfa = ExplicitDFA {
            start: 0,
            transitions: vec![vec![(store.top, 0)], vec![(store.top, 1)]],
            outputs: vec![store.zero; 2],
        };
        let domain = inputs();
        for seed in 0..24usize {
            let mut p = Cand::constant(&mut store, &dfa, false);
            let mut n = p.clone();
            let mut examples = Vec::new();
            for (i, input) in domain.iter().enumerate() {
                let single = Cand::singleton(&mut store, &dfa, input);
                match (i * 13 + seed * 7) % 9 {
                    0 => p = p.union(&single, &mut store),
                    1 => n = n.union(&single, &mut store),
                    _ => {}
                }
            }
            // Mix symbolic evidence with ordinary positive and negative examples.
            for (i, input) in domain.iter().enumerate() {
                if (i + seed) % 7 == 0 {
                    let label = !n.accepts_input(&store, input);
                    examples.push((
                        Input {
                            pkt_in: input.pkt_in.clone(),
                            pkt_start: input.pkt_start.clone(),
                            states: input.states.clone(),
                            pkt_end: input.pkt_end.clone(),
                            pkt_out: input.pkt_out.clone(),
                        },
                        label,
                    ));
                }
            }
            let result =
                Cand::from_examples_with_bounds(&mut store, &dfa, &examples, &p, &n).unwrap();
            let union = p.union(&n, &mut store);
            let difference = union.difference(&n, &mut store);
            assert!(p.intersect(&n, &mut store).is_empty(&store));
            for input in &domain {
                let yes = p.accepts_input(&store, input);
                let no = n.accepts_input(&store, input);
                assert_eq!(union.accepts_input(&store, input), yes || no);
                assert_eq!(difference.accepts_input(&store, input), yes);
                if yes {
                    assert!(result.accepts_input(&store, input));
                }
                if no {
                    assert!(!result.accepts_input(&store, input));
                }
            }
            for (input, label) in &examples {
                assert_eq!(result.accepts_input(&store, input), *label);
            }
            assert!(Cand::from_examples_with_bounds(&mut store, &dfa, &[], &p, &p).is_err());
        }
    }

    #[test]
    fn wide_regions_stay_shared_and_tighten_boundary_packets() {
        let mut store = spp::SPPstore::new(64);
        let dfa = ExplicitDFA {
            start: 0,
            transitions: vec![vec![(store.top, 0)]],
            outputs: vec![store.zero],
        };
        let pre = store.test(0, false);
        let pre = store.bwd(pre);
        let incoming = store.ibwd(pre);
        let outgoing = store.test(1, true);
        let outgoing = store.bwd(outgoing);
        let outgoing = store.ibwd(outgoing);
        let region = Cand::product(&mut store, &dfa, incoming, Some(&[0]), outgoing);
        assert!(store.accepts(incoming, &[false; 64], &[false; 64]));
        assert!(store.accepts(outgoing, &[true; 64], &[false; 64]));
        assert!(region.accepts_input(
            &store,
            &Input {
                pkt_in: vec![false; 64],
                pkt_start: vec![false; 64],
                states: vec![0],
                pkt_end: vec![true; 64],
                pkt_out: vec![false; 64]
            }
        ));
        assert!(region.nodes.len() < 300);
        let one = store.sp.one;
        let (pre, post) =
            region.boundary_packets(&mut store, &[false; 64], &[0], &[true; 64], one, one);
        assert_eq!(pre, store.bwd(incoming));
        assert_eq!(post, store.sp.one);
        let (pre, post) =
            region.boundary_packets(&mut store, &[false; 64], &[0], &[false; 64], one, one);
        assert!(store.sp.is_zero(pre));
        assert!(store.sp.is_zero(post));
    }
}
