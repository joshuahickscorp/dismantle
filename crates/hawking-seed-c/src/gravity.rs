//! Gravity as executable law: **capability is the constraint, BPW is what gets minimized.**
//!
//! The previous law made a number the criterion -- stay under one complete bit and you were
//! legal, and capability lived somewhere else. That is precisely how Math-Preserve was sealed:
//! 0.9774 BPW, byte-perfect, six green infrastructure gates, and it cannot complete "2 + 2 =".
//! Under the old `decide()` it was allowed with no receipt and no capability evidence, because
//! `Evidence` had no capability field at all. **Being small was being legal.**
//!
//! The law is now inverted. A rate is admissible only if the artifact AT THAT RATE is proven
//! usable, and the search is for the lowest rate that still is. Two tiers:
//!
//!   Tier::Condense  no hard ceiling, prefers <= 3/2 BPW, must be usable with capability
//!   Tier::Gravity   the lowest usable rate, whatever it turns out to be
//!
//! What did not change, because it is what kept the campaign honest: representation escalation
//! precedes BPW escalation; Doctor bytes count inside the same physical budget; exact rational
//! rates, never floats, because a float is not a scientific identity. And F1 weight
//! reconstruction can never promote anything -- it now denies at EVERY rate rather than only on
//! escape, since reconstruction error was never the thing that was wrong with Math-Preserve.

use crate::{Error, Result};

/// Exact rational rate (num/den). Floats are never scientific identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rate {
    pub num: u32,
    pub den: u32,
}
impl Rate {
    pub fn new(num: u32, den: u32) -> Self {
        assert!(den != 0);
        Rate { num, den }
    }
    /// Whole-artifact BPW below one complete physical bit.
    ///
    /// Retained as a description, no longer a permission. Sub-bit is where we would like to
    /// land; it is not what makes an artifact lawful.
    pub fn is_subbit(&self) -> bool {
        (self.num as u64) < (self.den as u64)
    }
    pub fn value(&self) -> f64 {
        self.num as f64 / self.den as f64
    }
    pub fn label(&self) -> String {
        format!("{}/{}", self.num, self.den)
    }
    /// Compare exactly, by cross-multiplication. Comparing `value()` would decide law with
    /// floating point, which is the one thing exact rationals exist to prevent.
    pub fn le(&self, other: &Rate) -> bool {
        (self.num as u64) * (other.den as u64) <= (other.num as u64) * (self.den as u64)
    }
    pub fn lt(&self, other: &Rate) -> bool {
        (self.num as u64) * (other.den as u64) < (other.num as u64) * (self.den as u64)
    }
}

/// Which ladder is being walked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Lowest usable rate that keeps capability. No hard ceiling; prefers 3/2 BPW or below.
    Condense,
    /// The lowest usable rate, full stop.
    Gravity,
}
impl Tier {
    /// A *preference*, not a wall. Exceeding it is allowed and must be receipted, so that a
    /// high-BPW artifact is a recorded decision rather than a drift nobody noticed.
    pub fn preferred_ceiling(&self) -> Option<Rate> {
        match self {
            Tier::Condense => Some(Rate::new(3, 2)),
            Tier::Gravity => None,
        }
    }
}

/// Proof that a specific artifact, at a specific rate, actually works.
///
/// Bound to the artifact hash AND the rate on purpose. A proof taken at 6/5 BPW says nothing
/// about 9/10 BPW: every rung of the ladder must be proven at that rung. Without this binding,
/// "usable" silently inherits downward and the ladder proves nothing.
#[derive(Debug, Clone, PartialEq)]
pub struct CapabilityProof {
    pub artifact_index_sha256: String,
    pub rate: Rate,
    /// One forward pass on "2 + 2 =". Math-Preserve fails this.
    pub g_math: bool,
    /// Two unrelated prompts must not produce the same output.
    pub g_live: bool,
}
impl CapabilityProof {
    pub fn usable(&self) -> bool {
        self.g_math && self.g_live
    }
}

/// What Gravity is being asked to authorize.
#[derive(Debug, Clone)]
pub enum Ask {
    RepresentationEscalation,
    /// Admit an artifact at a rate. This replaces the old BpwEscalation/EscapeAboveSubbit pair:
    /// there is no longer a magic boundary at one bit, so there is nothing to "escape".
    AdmitRate {
        to: Rate,
        tier: Tier,
        artifact_index_sha256: String,
        /// Required only to exceed the tier's preferred ceiling.
        sealed_receipt: bool,
    },
}

/// Evidence Gravity weighs. `f1_only` and `scheduler_deferred` are traps: they can never admit.
#[derive(Debug, Clone, Default)]
pub struct Evidence {
    pub representation_families_tried: u32,
    pub doctor_bytes_in_budget: bool,
    pub f1_only: bool,
    pub scheduler_deferred: bool,
    /// Absent means refused. There is no rate low enough to be lawful without this.
    pub capability: Option<CapabilityProof>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub allow: bool,
    pub reason: String,
    pub requires_receipt: bool,
}

/// The whole Gravity policy: a small pure function.
pub fn decide(_current: Rate, ask: &Ask, ev: &Evidence) -> Decision {
    match ask {
        Ask::RepresentationEscalation => Decision {
            allow: true,
            reason: "representation escalation precedes BPW escalation".into(),
            requires_receipt: false,
        },
        Ask::AdmitRate { to, tier, artifact_index_sha256, sealed_receipt } => {
            if ev.representation_families_tried < 1 {
                return Decision {
                    allow: false,
                    reason: "representation-before-BPW: try a stronger family first".into(),
                    requires_receipt: false,
                };
            }
            // Reconstruction error cannot promote, at ANY rate. Math-Preserve reconstructed
            // its weights byte-perfectly and still could not add two and two.
            if ev.f1_only {
                return Decision {
                    allow: false,
                    reason: "F1 weight reconstruction is NOT capability proof; admission denied".into(),
                    requires_receipt: true,
                };
            }
            if ev.scheduler_deferred {
                return Decision {
                    allow: false,
                    reason: "scheduler deferral is not scientific collapse; admission denied".into(),
                    requires_receipt: true,
                };
            }
            let Some(proof) = ev.capability.as_ref() else {
                return Decision {
                    allow: false,
                    reason: format!(
                        "no capability proof for {}: BPW alone cannot admit an artifact. \
                         Being small is not being usable.",
                        to.label()
                    ),
                    requires_receipt: true,
                };
            };
            if proof.artifact_index_sha256 != *artifact_index_sha256 {
                return Decision {
                    allow: false,
                    reason: "capability proof belongs to a different artifact".into(),
                    requires_receipt: true,
                };
            }
            if proof.rate != *to {
                return Decision {
                    allow: false,
                    reason: format!(
                        "capability proof was measured at {}, not {}; each rate must be proven at that rate",
                        proof.rate.label(),
                        to.label()
                    ),
                    requires_receipt: true,
                };
            }
            if !proof.usable() {
                let which = match (proof.g_math, proof.g_live) {
                    (false, false) => "G_math and G_live",
                    (false, true) => "G_math",
                    _ => "G_live",
                };
                return Decision {
                    allow: false,
                    reason: format!("artifact is not usable at {}: {which} failed", to.label()),
                    requires_receipt: true,
                };
            }
            match tier.preferred_ceiling() {
                Some(ceiling) if !to.le(&ceiling) && !*sealed_receipt => Decision {
                    allow: false,
                    reason: format!(
                        "{} exceeds the {:?} preferred ceiling of {}; allowed, but only against a sealed receipt",
                        to.label(),
                        tier,
                        ceiling.label()
                    ),
                    requires_receipt: true,
                },
                Some(ceiling) if !to.le(&ceiling) => Decision {
                    allow: true,
                    reason: format!(
                        "usable at {}, above the preferred {} and receipted",
                        to.label(),
                        ceiling.label()
                    ),
                    requires_receipt: true,
                },
                _ => Decision {
                    allow: true,
                    reason: format!("usable at {}", to.label()),
                    requires_receipt: false,
                },
            }
        }
    }
}

/// A rung of the search: a rate, and whether the artifact built at it was usable.
#[derive(Debug, Clone)]
pub struct Rung {
    pub rate: Rate,
    pub usable: bool,
}

/// The ladder. "Lowest usable" is a measured result, not a target chosen in advance.
#[derive(Debug, Clone, Default)]
pub struct Ladder {
    pub rungs: Vec<Rung>,
}
impl Ladder {
    pub fn record(&mut self, rate: Rate, usable: bool) {
        self.rungs.push(Rung { rate, usable });
    }
    /// The lowest rate actually proven usable. `None` means no rung passed -- which is a real
    /// answer, and the one that must not be rounded up into a shipped artifact.
    pub fn lowest_usable(&self) -> Option<Rate> {
        self.rungs
            .iter()
            .filter(|r| r.usable)
            .map(|r| r.rate)
            .reduce(|a, b| if b.lt(&a) { b } else { a })
    }
    /// The lowest usable rate within a tier's preference, if the tier has one.
    pub fn lowest_usable_for(&self, tier: Tier) -> Option<Rate> {
        let lowest = self.lowest_usable()?;
        match tier.preferred_ceiling() {
            Some(c) if !lowest.le(&c) => None,
            _ => Some(lowest),
        }
    }
}

/// Physical-byte conservation guard: Doctor bytes count inside the total budget.
pub fn total_bpw(base_bits: u64, doctor_bits: u64, overhead_bits: u64, n_weights: u64) -> f64 {
    (base_bits + doctor_bits + overhead_bits) as f64 / n_weights.max(1) as f64
}

/// Assert Doctor spending stays within the declared physical budget (same-rate treatment law).
pub fn doctor_within_budget(base_bits: u64, doctor_bits: u64, overhead_bits: u64, budget_bpw: f64, n_weights: u64) -> Result<()> {
    let whole = total_bpw(base_bits, doctor_bits, overhead_bits, n_weights);
    if whole <= budget_bpw + 1e-9 {
        Ok(())
    } else {
        Err(Error::Gravity(format!("Doctor bytes exceed budget: {whole:.4} > {budget_bpw:.4} BPW")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: &str = "abc123";

    fn proof(rate: Rate, g_math: bool, g_live: bool) -> CapabilityProof {
        CapabilityProof { artifact_index_sha256: H.into(), rate, g_math, g_live }
    }
    fn ev(cap: Option<CapabilityProof>) -> Evidence {
        Evidence { representation_families_tried: 2, capability: cap, ..Default::default() }
    }
    fn admit(to: Rate, tier: Tier, sealed_receipt: bool) -> Ask {
        Ask::AdmitRate { to, tier, artifact_index_sha256: H.into(), sealed_receipt }
    }

    /// The Math-Preserve case, which the old law allowed: deeply sub-bit, no capability.
    #[test]
    fn small_is_not_legal_without_capability() {
        let rate = Rate::new(9774, 10000);
        assert!(rate.is_subbit());
        let d = decide(rate, &admit(rate, Tier::Gravity, false), &ev(None));
        assert!(!d.allow, "sub-bit with no capability proof must be refused");
        assert!(d.reason.contains("Being small is not being usable"));
    }

    #[test]
    fn failing_g_math_is_refused_at_any_rate() {
        for rate in [Rate::new(1, 2), Rate::new(9774, 10000), Rate::new(3, 2)] {
            let d = decide(rate, &admit(rate, Tier::Condense, true), &ev(Some(proof(rate, false, true))));
            assert!(!d.allow);
            assert!(d.reason.contains("G_math"));
        }
    }

    /// A proof is bound to one artifact and one rate; it cannot be carried downward.
    #[test]
    fn proof_does_not_inherit_to_another_rate_or_artifact() {
        let proven = Rate::new(6, 5);
        let cheaper = Rate::new(9, 10);
        let d = decide(proven, &admit(cheaper, Tier::Gravity, false), &ev(Some(proof(proven, true, true))));
        assert!(!d.allow);
        assert!(d.reason.contains("must be proven at that rate"));

        let other = Evidence {
            representation_families_tried: 2,
            capability: Some(CapabilityProof {
                artifact_index_sha256: "different".into(),
                rate: proven,
                g_math: true,
                g_live: true,
            }),
            ..Default::default()
        };
        assert!(!decide(proven, &admit(proven, Tier::Gravity, false), &other).allow);
    }

    #[test]
    fn reconstruction_can_never_admit() {
        let rate = Rate::new(1, 2);
        let mut e = ev(Some(proof(rate, true, true)));
        e.f1_only = true;
        assert!(!decide(rate, &admit(rate, Tier::Gravity, true), &e).allow);
        let mut e = ev(Some(proof(rate, true, true)));
        e.scheduler_deferred = true;
        assert!(!decide(rate, &admit(rate, Tier::Gravity, true), &e).allow);
    }

    #[test]
    fn condense_prefers_but_does_not_forbid_above_one_point_five() {
        let high = Rate::new(2, 1); // 2.0 BPW
        let e = ev(Some(proof(high, true, true)));
        let d = decide(high, &admit(high, Tier::Condense, false), &e);
        assert!(!d.allow, "above the preferred ceiling without a receipt");
        assert!(d.requires_receipt);
        let d = decide(high, &admit(high, Tier::Condense, true), &e);
        assert!(d.allow, "a preference is not a wall: receipted, it is admitted");
        assert!(d.requires_receipt);
        // Gravity has no ceiling at all -- it takes the lowest usable, whatever it is.
        assert!(decide(high, &admit(high, Tier::Gravity, false), &e).allow);
    }

    #[test]
    fn usable_sub_bit_is_admitted_without_a_receipt() {
        let rate = Rate::new(167, 1000);
        let d = decide(rate, &admit(rate, Tier::Condense, false), &ev(Some(proof(rate, true, true))));
        assert!(d.allow);
        assert!(!d.requires_receipt);
    }

    #[test]
    fn ladder_reports_the_lowest_rate_actually_proven() {
        let mut l = Ladder::default();
        l.record(Rate::new(3, 2), true);
        l.record(Rate::new(1, 1), true);
        l.record(Rate::new(1, 2), false); // built, measured, not usable
        assert_eq!(l.lowest_usable(), Some(Rate::new(1, 1)));
        assert_eq!(l.lowest_usable_for(Tier::Condense), Some(Rate::new(1, 1)));

        // Nothing usable is a real answer and must stay None rather than round up.
        let mut none = Ladder::default();
        none.record(Rate::new(1, 2), false);
        assert_eq!(none.lowest_usable(), None);

        // Lowest usable above the Condense preference is not silently returned as within it.
        let mut high = Ladder::default();
        high.record(Rate::new(2, 1), true);
        assert_eq!(high.lowest_usable(), Some(Rate::new(2, 1)));
        assert_eq!(high.lowest_usable_for(Tier::Condense), None);
    }

    #[test]
    fn representation_precedes_bpw_and_doctor_budget() {
        let rate = Rate::new(9, 10);
        let e = Evidence { capability: Some(proof(rate, true, true)), ..Default::default() };
        assert!(!decide(rate, &admit(rate, Tier::Gravity, false), &e).allow, "no family tried yet");
        assert!(doctor_within_budget(500, 300, 0, 0.8, 1000).is_ok());
        assert!(doctor_within_budget(500, 400, 0, 0.8, 1000).is_err());
    }

    #[test]
    fn exact_rational_comparison_is_exact_and_does_not_overflow() {
        // Cross-multiplication at the u32 ceiling: 4294967295^2 = 18446744065119617025,
        // against a u64 max of 18446744073709551615. It fits with 8.6e9 to spare, which is
        // close enough that widening to u64 is load-bearing rather than defensive -- in u32
        // this wraps and the law silently orders rates backwards.
        let big = Rate::new(u32::MAX, u32::MAX);
        assert!(big.le(&Rate::new(u32::MAX, u32::MAX)));
        assert!(Rate::new(u32::MAX - 1, u32::MAX).lt(&big));

        // Adjacent fractions an f64 round-trip would blur, ordered exactly.
        let a = Rate::new(100_000_000, 300_000_000);
        let b = Rate::new(100_000_000, 300_000_001);
        assert!(b.lt(&a));
        assert!(!a.lt(&b));
    }
}
