//! Event Horizon — `SuffixAutomaton`: online SAM-backed exact-match proposer.
//!
//! A second long-match proposer that complements `SuffixArrayDraft`.
//! Where `SuffixArrayDraft` does a backwards linear scan O(stream·H), this
//! module builds an *online suffix automaton* (SAM) over the emitted token
//! stream, giving O(1) amortised construction per token and O(k) lookup.
//!
//! Gate: env var `HAWKING_EH_SAM` (default unset / disabled).
//!       If the var is not set, `propose()` returns an empty `TokenLine`.
//!
//! Lossless by construction: every proposed token is confirmed by the verifier
//! before it is emitted.
//!
//! ## SAM basics (adapted for token ids)
//! A suffix automaton over a string `s` is a DAWG that accepts every suffix of
//! `s`.  Online construction (Blumer et al. / Ukkonen/DAWG variant) adds each
//! character in O(1) amortised.  Each state `q` represents an equivalence class
//! of substrings (all substrings of equal right-extension sets).
//!
//! Key fields per state:
//! - `len`   — length of the longest substring in the class.
//! - `link`  — suffix link (parent in the suffix-link tree; link of root is 0).
//! - `next`  — transition function: token_id → state_index.
//!
//! After construction, the *endpos* of any accepted suffix can be found by
//! following transitions from state 0.

use crate::proposal::{Budget, Ctx, Proposal, Proposer, Telemetry};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Sub-flag
// ---------------------------------------------------------------------------

/// Returns `true` when `HAWKING_EH_SAM` is set in the environment.
pub fn sam_enabled() -> bool {
    std::env::var("HAWKING_EH_SAM").is_ok()
}

// ---------------------------------------------------------------------------
// SAM internals
// ---------------------------------------------------------------------------

/// One state in the suffix automaton.
#[derive(Debug, Clone)]
struct SamState {
    /// Length of the longest substring in the equivalence class.
    len: usize,
    /// Suffix link (index into the `states` vec).  Root's link is 0 (self).
    link: usize,
    /// Transition map: token_id → successor state index.
    next: HashMap<u32, usize>,
    /// Index in the original `stream` where this state's longest match ENDS
    /// (1-based; 0 means "not yet set").  Used to resume the continuation
    /// after a match.
    end_pos: usize,
}

impl SamState {
    fn new(len: usize, link: usize) -> Self {
        SamState {
            len,
            link,
            next: HashMap::new(),
            end_pos: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Public struct
// ---------------------------------------------------------------------------

/// Online suffix-automaton proposer.
///
/// Maintains a SAM over the full emitted token stream (up to `window` tokens).
/// At each step, walks the automaton with up to `H=8` tail tokens to find the
/// longest suffix of the context that has a continuation in the SAM, then
/// follows the transition chain for up to `budget.k` tokens.
#[derive(Debug, Clone)]
pub struct SuffixAutomaton {
    /// All SAM states.  Index 0 is the initial (root) state.
    states: Vec<SamState>,
    /// Index of the "last" state after the most recent `extend_sam` call.
    last: usize,
    /// Mirror of the raw token stream (oldest-first).  The SAM's `end_pos`
    /// values are indices into this vec (0-based, pointing to the token right
    /// AFTER the match, so `stream[end_pos]` is the first continuation token).
    stream: Vec<u32>,
    /// Maximum emitted history to retain (default 50 000).
    window: usize,
    /// Context window for matching (default 8 tokens, as per spec).
    h: usize,
}

impl Default for SuffixAutomaton {
    fn default() -> Self {
        // Pre-allocate root state (index 0, len=0, link=0 = self).
        let root = SamState::new(0, 0);
        SuffixAutomaton {
            states: vec![root],
            last: 0,
            stream: Vec::new(),
            window: 50_000,
            h: 8,
        }
    }
}

impl SuffixAutomaton {
    pub fn new() -> Self {
        Self::default()
    }

    // -----------------------------------------------------------------------
    // Online SAM construction
    // -----------------------------------------------------------------------

    /// Append one token to the SAM (online Ukkonen/DAWG construction).
    /// O(1) amortised.
    fn extend_sam(&mut self, token: u32) {
        // Record position in stream BEFORE pushing (0-based index of this token).
        let pos = self.stream.len(); // index this token will occupy
        self.stream.push(token);

        let cur = self.states.len();
        let mut new_state = SamState::new(
            self.states[self.last].len + 1,
            0, // link set below
        );
        // end_pos is 1-based index past the end of this token = pos + 1.
        new_state.end_pos = pos + 1;
        self.states.push(new_state);

        let mut p = self.last;
        // Walk suffix-link chain adding transitions to `cur`.
        loop {
            if self.states[p].next.contains_key(&token) {
                break;
            }
            self.states[p].next.insert(token, cur);
            let link = self.states[p].link;
            if p == 0 {
                // Reached root and it had no transition for `token`.
                // Root's link is itself; break after inserting.
                // (We already inserted above before checking p==0.)
                break;
            }
            p = link;
        }

        // `q` is the state we stopped at (it has a transition for `token`).
        if !self.states[p].next.contains_key(&token) {
            // Only happens at root when we break before the `contains_key` check.
            // Root link = 0, suffix link of `cur` = root = 0.
            self.states[cur].link = 0;
        } else {
            let q = self.states[p].next[&token];
            if self.states[q].len == self.states[p].len + 1 {
                // `q` is already a proper suffix-link target.
                self.states[cur].link = q;
            } else {
                // Clone `q` as `clone_q`.
                let clone_q = self.states.len();
                let q_state = self.states[q].clone();
                let mut cloned = SamState::new(self.states[p].len + 1, q_state.link);
                cloned.next = q_state.next.clone();
                cloned.end_pos = q_state.end_pos;
                self.states.push(cloned);

                // Retarget `p`'s suffix-link chain to `clone_q`.
                let mut pp = p;
                loop {
                    let trans = self.states[pp].next.get(&token).copied();
                    if trans == Some(q) {
                        self.states[pp].next.insert(token, clone_q);
                    } else {
                        break;
                    }
                    if pp == 0 {
                        break;
                    }
                    pp = self.states[pp].link;
                }
                self.states[q].link = clone_q;
                self.states[cur].link = clone_q;
            }
        }

        self.last = cur;
    }

    // -----------------------------------------------------------------------
    // Proposal helpers
    // -----------------------------------------------------------------------

    /// Walk the SAM from root following `query` tokens.
    /// Returns `Some(state_index)` of the state reached after consuming all
    /// tokens, or `None` if any transition is missing.
    fn walk(&self, query: &[u32]) -> Option<usize> {
        let mut state = 0usize; // root
        for &tok in query {
            match self.states[state].next.get(&tok) {
                Some(&next) => state = next,
                None => return None,
            }
        }
        Some(state)
    }

    /// Given a SAM `state` reached after matching `matched_len` tokens, follow
    /// transitions greedily for up to `k` more tokens using `end_pos` as the
    /// continuation anchor.  Returns the continuation token sequence.
    ///
    /// Strategy: the state's `end_pos` tells us where that equivalence class'
    /// longest match ends in the raw stream.  The continuation is the `k`
    /// tokens starting at `stream[end_pos]`.
    fn continuation(&self, state: usize, k: usize) -> Vec<u32> {
        let ep = self.states[state].end_pos;
        if ep == 0 || ep >= self.stream.len() {
            return Vec::new();
        }
        let end = (ep + k).min(self.stream.len());
        self.stream[ep..end].to_vec()
    }
}

// ---------------------------------------------------------------------------
// Proposer impl
// ---------------------------------------------------------------------------

impl Proposer for SuffixAutomaton {
    fn name(&self) -> &'static str {
        "suffix_automaton"
    }

    fn cost_estimate(&self, _ctx: &Ctx<'_>, _budget: Budget) -> crate::proposal::CostNs {
        0
    }

    fn propose(&mut self, ctx: &Ctx<'_>, budget: Budget, _tel: &Telemetry) -> Proposal {
        if !sam_enabled() {
            return Proposal::TokenLine(Vec::new());
        }
        let k = budget.k;
        if k == 0 || self.stream.len() < 2 {
            return Proposal::TokenLine(Vec::new());
        }

        // Try progressively shorter suffixes of ctx.tokens (up to H=h).
        let tokens = ctx.tokens;
        let query_max = self.h.min(tokens.len());

        for query_len in (1..=query_max).rev() {
            let query = &tokens[tokens.len() - query_len..];
            if let Some(state) = self.walk(query) {
                // Only use a match if the state has a continuation (end_pos set
                // and there are tokens after it).
                let cont = self.continuation(state, k);
                if !cont.is_empty() {
                    return Proposal::TokenLine(cont);
                }
            }
        }

        Proposal::TokenLine(Vec::new())
    }

    fn observe(&mut self, emitted: &[u32]) {
        for &tok in emitted {
            self.extend_sam(tok);
        }
        // Window enforcement: if the stream has grown beyond `window`, we need
        // to rebuild the SAM from scratch over the retained suffix.  This is
        // O(window) but happens at most once every `window` tokens — amortised
        // O(1) per token.
        if self.stream.len() > self.window {
            let keep_from = self.stream.len() - self.window;
            let retained: Vec<u32> = self.stream[keep_from..].to_vec();
            // Reset SAM.
            let root = SamState::new(0, 0);
            self.states = vec![root];
            self.last = 0;
            self.stream.clear();
            for tok in retained {
                self.extend_sam(tok);
            }
        }
    }

    fn warm(&mut self, history: &[u32]) {
        for &tok in history {
            self.extend_sam(tok);
        }
        if self.stream.len() > self.window {
            let keep_from = self.stream.len() - self.window;
            let retained: Vec<u32> = self.stream[keep_from..].to_vec();
            let root = SamState::new(0, 0);
            self.states = vec![root];
            self.last = 0;
            self.stream.clear();
            for tok in retained {
                self.extend_sam(tok);
            }
        }
    }

    fn reset(&mut self) {
        // The SAM index persists across turns (same policy as SuffixArrayDraft).
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proposal::{Budget, Ctx, Telemetry};
    fn make_ctx(tokens: &[u32]) -> Ctx<'_> {
        Ctx {
            tokens,
            pos: tokens.len(),
            hidden: None,
        }
    }
    fn propose_tokens(sam: &mut SuffixAutomaton, ctx_toks: &[u32], k: usize) -> Vec<u32> {
        let ctx = make_ctx(ctx_toks);
        match sam.propose(&ctx, Budget::line(k), &Telemetry::default()) {
            Proposal::TokenLine(v) => v,
            _ => panic!("expected TokenLine"),
        }
    }
    #[test]
    fn sam_disabled_by_default() {
        if sam_enabled() {
            return;
        }
        let mut sam = SuffixAutomaton::new();
        sam.observe(&[1u32, 2, 3, 1, 2]);
        let result = propose_tokens(&mut sam, &[1u32, 2], 4);
        if sam_enabled() {
            return;
        }
        assert!(result.is_empty(), "SAM disabled → must propose nothing");
    }
    #[test]
    fn sam_extends_to_one_token() {
        // SAFETY: env mutation is intentional; tests may race in parallel,
        unsafe { std::env::set_var("HAWKING_EH_SAM", "1") };
        let mut sam = SuffixAutomaton::new();
        sam.observe(&[1u32, 2, 3, 1, 2]);
        let result = propose_tokens(&mut sam, &[1u32, 2], 4);
 assert!( !result.is_empty(), "expected at least one token after [1,2] in stream [1,2,3,1,2]" );
        assert_eq!(result[0], 3u32, "first continuation of [1,2] should be 3");
    }
    #[test]
    fn sam_long_recurring_span() {
        unsafe { std::env::set_var("HAWKING_EH_SAM", "1") };
        let stream: Vec<u32> = (0u32..7).chain(0u32..7).collect();
        let mut sam = SuffixAutomaton::new();
        sam.observe(&stream);
        let result = propose_tokens(&mut sam, &[6u32, 0], 6);
 assert!( result.len() >= 6, "expected 6 continuation tokens [1,2,3,4,5,6], got {:?}", result );
        assert_eq!(&result[..6], &[1u32, 2, 3, 4, 5, 6]);
    }
    #[test]
    fn sam_collision_safe() {
        unsafe { std::env::set_var("HAWKING_EH_SAM", "1") };
        let mut sam = SuffixAutomaton::new();
        sam.observe(&[1u32, 2, 3]);
        sam.observe(&[1u32, 2, 4]);
        let result = propose_tokens(&mut sam, &[1u32, 2], 4);
 assert!( !result.is_empty(), "should produce a continuation for [1,2]" );
        let result2 = propose_tokens(&mut sam, &[1u32, 2], 4);
 assert_eq!( result, result2, "SAM must be deterministic across identical calls" );
    }
    #[test]
    fn sam_respects_budget() {
        unsafe { std::env::set_var("HAWKING_EH_SAM", "1") };
        let mut sam = SuffixAutomaton::new();
        let stream: Vec<u32> = (0u32..10).chain(0u32..10).collect();
        sam.observe(&stream);
        for k in 1..=Budget::MAX_DRAFT_LEN {
            let ctx_toks: Vec<u32> = vec![0u32, 1];
            let ctx = make_ctx(&ctx_toks);
            let result = match sam.propose(&ctx, Budget::line(k), &Telemetry::default()) {
                Proposal::TokenLine(v) => v,
                _ => panic!("expected TokenLine"),
            };
 assert!( result.len() <= k, "budget k={k}: got {} tokens (> k)", result.len() );
        }
    }
}
