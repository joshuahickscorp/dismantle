//! hide-browser: the browser evidence and visual verification model.
//!
//! HIDE treats "did the change work in the browser?" as a first-class,
//! evidence-backed question (Bible Book VIII sec 27). This crate defines the
//! shapes that answer it and the deterministic machinery that grades them:
//!
//! - [`evidence`]: a [`BrowserStep`] evidence record (url, navigation cause, DOM
//!   snapshot, accessibility tree, screenshot ref, console + network events,
//!   the selected element, the action, the resulting state, timing) and a
//!   [`BrowserSession`] -- an ordered, replayable list of steps.
//! - [`driver`]: a [`BrowserDriver`] trait (navigate/click/fill + screenshot/
//!   dom/accessibility/console/network) and a [`ReplayDriver`] that plays a
//!   recorded session back deterministically.
//! - [`acceptance`]: a [`VisualAcceptance`] oracle with a deterministic
//!   functional evaluator that reads a recorded [`ResultingState`] and returns a
//!   typed [`Verdict`], plus a deterministic a11y evaluator over the recorded
//!   accessibility tree.
//! - [`design_mode`]: [`DesignAnnotation`] mapping a selected element to its DOM
//!   node, source symbol, CSS rule, layout box, and accessibility node.
//!
//! # Model-free
//!
//! This crate never opens a network connection, never drives a real browser,
//! and runs no model (RIP doctrine). Everything is proven with deterministic
//! tests over recorded fixtures. Heavy artifacts (screenshots, raw HTML,
//! network bodies) are carried by content-addressed [`ArtifactRef`], never
//! inlined.
//!
//! The legs that inherently need a real renderer or a model are marked
//! `DEFERRED_MODEL_REQUIRED` at their definitions and are NOT implemented or
//! claimed here:
//!
//! - a live chromium/CDP backend that produces evidence from a real page
//!   (a future implementor of [`BrowserDriver`]; see [`driver`]);
//! - pixel/appearance grading of a screenshot against a target within
//!   tolerances, and judging `semantic_requirements`
//!   (see [`VisualAcceptance::evaluate_visual`]).
//!
//! Wire shapes here are HIDE-native. Where a concept is borrowed from an open
//! spec it is noted at the definition as spec-derived (for example the ARIA
//! interactive-role set in [`a11y`]); no proprietary source is copied.
//!
//! ```
//! use crate::browser::{ElementSelector, StateCheck, VisualAcceptance, ResultingState};
//!
//! let acc = VisualAcceptance {
//!     id: "va".into(),
//!     target: None,
//!     responsive_states: vec![],
//!     semantic_requirements: vec![],
//!     functional_interactions: vec![crate::browser::FunctionalRequirement {
//!         id: "cart".into(),
//!         description: "cart shows one item".into(),
//!         check: StateCheck::SignalEquals { key: "cart.count".into(), value: "1".into() },
//!     }],
//!     a11y_requirements: vec![],
//!     tolerances: Default::default(),
//!     before_after: None,
//! };
//! let good = ResultingState::at("https://x/").signal("cart.count", "1");
//! assert!(acc.evaluate_functional(&good).is_pass());
//! let _ = ElementSelector::css("#add");
//! ```

pub use a11y::{is_interactive_role, AccessibilityNode, AccessibilityTree, INTERACTIVE_ROLES};
pub use acceptance::{
    A11yCheck, A11yRequirement, BeforeAfter, FailureKind, FailureReason, FunctionalRequirement,
    ResponsiveTarget, SemanticRequirement, StateCheck, Tolerances, Verdict, VisualAcceptance,
};
pub use design_mode::{annotate, DesignAnnotation};
pub use dom::{BoxModel, CssRule, DomNode, DomSnapshot, EdgeSizes, SelectStrategy, SourceSymbol};
pub use driver::{BrowserDriver, ReplayDriver};
pub use error::{BrowserError, Result};
pub use evidence::{
    BrowserAction, BrowserSession, BrowserStep, ConsoleEvent, ConsoleLevel, ElementSelector,
    NavigationCause, NetworkEvent, ResponsiveState, ResultingState, Viewport,
};
pub use ids::{AccessibilityNodeId, ArtifactRef, BrowserSessionId, DomNodeId};

/// Generate the JSON Schema for a type as a [`serde_json::Value`]. The schema is
/// derived from the Rust definition, never hand-maintained, so it cannot drift
/// from the shape the code serializes.
pub fn json_schema<T: schemars::JsonSchema>() -> serde_json::Value {
    let root = schemars::gen::SchemaGenerator::default().into_root_schema_for::<T>();
    serde_json::to_value(root).expect("a schemars RootSchema always serializes to JSON")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    const PRODUCT_URL: &str = "https://shop.test/product/42";
    const CHECKOUT_URL: &str = "https://shop.test/checkout";
    fn page_dom() -> DomSnapshot {
        let mut btn = DomNode::leaf("n-btn", "button");
        btn.attributes.insert("id".into(), "add-to-cart".into());
        btn.attributes.insert("data-testid".into(), "add-cart".into());
        btn.text = Some("Add to cart".into());
        btn.box_model = Some(BoxModel {
            x: 20.0,
            y: 100.0,
            width: 140.0,
            height: 44.0,
            ..Default::default()
        });
        btn.a11y_node = Some(AccessibilityNodeId::from("ax-btn"));
        btn.source_symbol = Some(SourceSymbol {
            file: "app/ProductPage.tsx".into(),
            symbol: "AddToCartButton".into(),
            line: Some(42),
            column: None,
        });
        let mut decls = BTreeMap::new();
        decls.insert("background".into(), "#070707".into());
        btn.css_rules = vec![CssRule {
            selector: "#add-to-cart".into(),
            source: Some("app/product.css".into()),
            declarations: decls,
        }];
        let mut coupon = DomNode::leaf("n-coupon", "input");
        coupon.attributes.insert("id".into(), "coupon".into());
        coupon.a11y_node = Some(AccessibilityNodeId::from("ax-coupon"));
        let mut root = DomNode::leaf("n-root", "main");
        root.children.push(btn);
        root.children.push(coupon);
        DomSnapshot::new(root)
    }
    fn page_a11y(button_named: bool) -> AccessibilityTree {
        let button = if button_named {
            AccessibilityNode::new("ax-btn", "button").with_name("Add to cart")
        } else {
            AccessibilityNode::new("ax-btn", "button")
        };
        AccessibilityTree::new(
            AccessibilityNode::new("ax-root", "document")
                .with_child(button)
                .with_child(AccessibilityNode::new("ax-coupon", "textbox").with_name("Coupon code")),
        )
    }
    fn network_get() -> NetworkEvent {
        NetworkEvent {
            request_id: "req-1".into(),
            method: "GET".into(),
            url: PRODUCT_URL.into(),
            resource_type: Some("document".into()),
            status: Some(200),
            request_ref: None,
            response_ref: Some(ArtifactRef::content_addressed(
                b"<html>product</html>",
                Some("text/html"),
            )),
            timing_ms: Some(80),
        }
    }
    fn session() -> BrowserSession {
        let step0 = BrowserStep {
            index: 0,
            url: PRODUCT_URL.into(),
            navigation_cause: NavigationCause::UserNavigate,
            action: BrowserAction::Navigate,
            selected_element: None,
            dom_snapshot: page_dom(),
            accessibility_tree: page_a11y(true),
            screenshot_ref: Some(ArtifactRef::content_addressed(b"png-step-0", Some("image/png"))),
            console_events: vec![],
            network_events: vec![network_get()],
            resulting_state: ResultingState {
                http_status: Some(200),
                ..ResultingState::at(PRODUCT_URL)
                    .signal("cart.count", "0")
                    .present("#add-to-cart")
                    .present("#coupon")
                    .text("Product 42")
            },
            timing_ms: 120,
        };
        let step1 = BrowserStep {
            index: 1,
            url: PRODUCT_URL.into(),
            navigation_cause: NavigationCause::None,
            action: BrowserAction::Click,
            selected_element: Some(ElementSelector::css("#add-to-cart").with_node("n-btn")),
            dom_snapshot: page_dom(),
            accessibility_tree: page_a11y(true),
            screenshot_ref: Some(ArtifactRef::content_addressed(b"png-step-1", Some("image/png"))),
            console_events: vec![],
            network_events: vec![],
            resulting_state: ResultingState {
                http_status: Some(200),
                ..ResultingState::at(PRODUCT_URL)
                    .signal("cart.count", "1")
                    .present("#add-to-cart")
                    .present("#coupon")
                    .present("#cart-toast")
                    .text("Added to cart")
            },
            timing_ms: 40,
        };
        let step2 = BrowserStep {
            index: 2,
            url: PRODUCT_URL.into(),
            navigation_cause: NavigationCause::None,
            action: BrowserAction::Fill {
                value: "SAVE10".into(),
            },
            selected_element: Some(ElementSelector::css("#coupon").with_node("n-coupon")),
            dom_snapshot: page_dom(),
            accessibility_tree: page_a11y(true),
            screenshot_ref: Some(ArtifactRef::content_addressed(b"png-step-2", Some("image/png"))),
            console_events: vec![],
            network_events: vec![],
            resulting_state: ResultingState {
                http_status: Some(200),
                ..ResultingState::at(PRODUCT_URL)
                    .signal("cart.count", "1")
                    .signal("coupon", "SAVE10")
            },
            timing_ms: 30,
        };
        let step3 = BrowserStep {
            index: 3,
            url: CHECKOUT_URL.into(),
            navigation_cause: NavigationCause::FormSubmit,
            action: BrowserAction::Navigate,
            selected_element: None,
            dom_snapshot: page_dom(),
            accessibility_tree: page_a11y(true),
            screenshot_ref: Some(ArtifactRef::content_addressed(b"png-step-3", Some("image/png"))),
            console_events: vec![],
            network_events: vec![],
            resulting_state: ResultingState {
                http_status: Some(200),
                ..ResultingState::at(CHECKOUT_URL)
            },
            timing_ms: 200,
        };
        BrowserSession::new("bs-cart", vec![step0, step1, step2, step3])
    }
    #[test]
    fn replay_plays_recorded_session_step_by_step_in_order() {
        let recorded = session();
        let wire = serde_json::to_string(&recorded).unwrap();
        let restored: BrowserSession = serde_json::from_str(&wire).unwrap();
        assert_eq!(restored, recorded);
        assert!(restored.indices_are_sequential());
        let mut d = ReplayDriver::new(restored);
        assert_eq!(d.len(), 4);
        let mut played = Vec::new();
        played.push(d.navigate(PRODUCT_URL).unwrap());
 assert_eq!( d.screenshot().unwrap(), recorded.steps[0].screenshot_ref.clone().unwrap() );
        assert_eq!(d.dom().unwrap(), recorded.steps[0].dom_snapshot);
        assert_eq!(d.accessibility().unwrap(), recorded.steps[0].accessibility_tree);
        assert_eq!(d.network().unwrap(), recorded.steps[0].network_events);
        assert!(d.console().unwrap().is_empty());
        played.push(
            d.click(&ElementSelector::css("#add-to-cart"))
                .unwrap(),
        );
        played.push(d.fill(&ElementSelector::css("#coupon"), "SAVE10").unwrap());
        played.push(d.navigate(CHECKOUT_URL).unwrap());
        assert_eq!(played, recorded.steps, "each step replayed in recorded order");
        assert!(d.is_exhausted());
 assert_eq!( d.navigate("anywhere"), Err(BrowserError::ReplayExhausted { requested: "navigate" }) );
    }
    #[test]
    fn replay_is_a_strict_contract_and_reports_mismatch() {
        let mut d = ReplayDriver::new(session());
        let err = d.click(&ElementSelector::css("#add-to-cart")).unwrap_err();
        match err {
            BrowserError::ReplayMismatch {
                index,
                requested,
                expected,
                ..
            } => {
                assert_eq!(index, 0);
                assert_eq!(requested, "click");
                assert_eq!(expected, "navigate");
            }
            other => panic!("expected a replay mismatch, got {other:?}"),
        }
        assert!(d.navigate(PRODUCT_URL).is_ok());
    }
    #[test]
    fn observers_error_before_any_step_is_played() {
        let d = ReplayDriver::new(session());
        assert_eq!(d.dom(), Err(BrowserError::NoCurrentStep));
        assert_eq!(d.screenshot(), Err(BrowserError::NoCurrentStep));
    }
    fn acceptance() -> VisualAcceptance {
        VisualAcceptance {
            id: "va-add-to-cart".into(),
            target: Some(ArtifactRef::new("design/add-to-cart.png")),
            responsive_states: vec![ResponsiveTarget {
                name: "mobile".into(),
                viewport: Viewport {
                    width: 375,
                    height: 812,
                },
                screenshot_ref: Some(ArtifactRef::new("design/add-to-cart.mobile.png")),
            }],
            semantic_requirements: vec![SemanticRequirement {
                id: "cta-prominent".into(),
                description: "the add-to-cart CTA is the visual focus".into(),
            }],
            functional_interactions: vec![
                FunctionalRequirement {
                    id: "cart-incremented".into(),
                    description: "cart shows one item after adding".into(),
                    check: StateCheck::SignalEquals {
                        key: "cart.count".into(),
                        value: "1".into(),
                    },
                },
                FunctionalRequirement {
                    id: "toast-shown".into(),
                    description: "a confirmation toast appears".into(),
                    check: StateCheck::ElementPresent {
                        selector: "#cart-toast".into(),
                    },
                },
                FunctionalRequirement {
                    id: "no-console-errors".into(),
                    description: "no console errors during the interaction".into(),
                    check: StateCheck::NoConsoleErrors,
                },
                FunctionalRequirement {
                    id: "stays-on-product".into(),
                    description: "still on the product page".into(),
                    check: StateCheck::UrlContains {
                        fragment: "/product/42".into(),
                    },
                },
            ],
            a11y_requirements: vec![
                A11yRequirement {
                    id: "button-has-name".into(),
                    description: "the add button has an accessible name".into(),
                    check: A11yCheck::RoleNamed {
                        role: "button".into(),
                    },
                },
                A11yRequirement {
                    id: "no-unnamed-controls".into(),
                    description: "every interactive control is named".into(),
                    check: A11yCheck::NoUnnamedInteractive,
                },
            ],
            tolerances: Tolerances::default(),
            before_after: Some(BeforeAfter {
                before: Some(ArtifactRef::new("art/before.png")),
                after: Some(ArtifactRef::new("art/after.png")),
            }),
        }
    }
    #[test]
    fn functional_oracle_passes_on_the_good_recorded_state() {
        let s = session();
        let good = &s.steps[1].resulting_state; // after the click
        let verdict = acceptance().evaluate_functional(good);
        assert!(verdict.is_pass(), "good state passes: {verdict:?}");
    }
    #[test]
    fn functional_oracle_fails_on_a_bad_state_with_typed_reasons() {
        let s = session();
        let mut bad = s.steps[1].resulting_state.clone();
        bad.signals.insert("cart.count".into(), "0".into());
        bad.present_selectors.remove("#cart-toast");
        bad.console_error_count = 2;
        let verdict = acceptance().evaluate_functional(&bad);
        assert!(!verdict.is_pass());
        let reasons = verdict.reasons();
        assert_eq!(reasons.len(), 3, "three requirements failed: {reasons:?}");
        assert!(reasons.iter().any(|r| matches!(
            &r.kind,
            FailureKind::SignalMismatch { key, expected, actual }
                if key == "cart.count" && expected == "1" && actual == "0"
        )));
        assert!(reasons.iter().any(|r| matches!(
            &r.kind,
            FailureKind::ElementMissing { selector } if selector == "#cart-toast"
        )));
        assert!(reasons.iter().any(|r| matches!(
            &r.kind,
            FailureKind::ConsoleErrors { count, allowed } if *count == 2 && *allowed == 0
        )));
        assert!(reasons.iter().all(|r| !r.requirement_id.is_empty()));
    }
    #[test]
    fn a11y_oracle_passes_on_named_controls_and_fails_when_unnamed() {
        let acc = acceptance();
        assert!(acc.evaluate_a11y(&page_a11y(true)).is_pass());
        let verdict = acc.evaluate_a11y(&page_a11y(false));
        assert!(!verdict.is_pass());
        assert!(verdict
            .reasons()
            .iter()
            .any(|r| matches!(&r.kind, FailureKind::A11yUnnamed { role } if role == "button")));
    }
    #[test]
    fn annotation_maps_a_selection_to_its_dom_node() {
        let dom = session().steps[0].dom_snapshot.clone();
        let sel = ElementSelector::css("#add-to-cart").with_node("n-btn");
        let ann = annotate(&sel, &dom).unwrap();
        assert_eq!(ann.dom_node, DomNodeId::from("n-btn"));
        assert_eq!(ann.a11y_node, Some(AccessibilityNodeId::from("ax-btn")));
 assert_eq!( ann.source_symbol.as_ref().map(|s| s.symbol.as_str()), Some("AddToCartButton") );
 assert_eq!( ann.css_rule.as_ref().map(|c| c.selector.as_str()), Some("#add-to-cart") );
        assert!(ann.box_model.width > 0.0);
        let sel2 = ElementSelector::test_id("add-cart");
        let ann2 = annotate(&sel2, &dom).unwrap();
        assert_eq!(ann2.dom_node, DomNodeId::from("n-btn"));
        let ghost = ElementSelector::css("#ghost");
 assert!(matches!( annotate(&ghost, &dom), Err(BrowserError::UnresolvedSelection { .. }) ));
    }
    #[test]
    fn screenshots_and_network_bodies_are_referenced_not_inlined() {
        let s = session();
        let step = &s.steps[0];
        let sref = step.screenshot_ref.as_ref().unwrap();
        assert!(sref.is_content_addressed());
        let net = &step.network_events[0];
        assert!(net.response_ref.as_ref().unwrap().is_content_addressed());
        let json = serde_json::to_string(step).unwrap();
        assert!(json.contains(&sref.id), "the screenshot ref id is on the wire");
        assert!(!json.contains("png-step-0"), "screenshot bytes are not inlined");
        assert!(!json.contains("<html>product</html>"), "response body not inlined");
    }
    #[test]
    fn top_types_round_trip_through_serde_json() {
        let s = session();
        let back: BrowserSession =
            serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(back, s);
        let acc = acceptance();
        let back_acc: VisualAcceptance =
            serde_json::from_str(&serde_json::to_string(&acc).unwrap()).unwrap();
        assert_eq!(back_acc, acc);
        let ann = annotate(
            &ElementSelector::css("#add-to-cart").with_node("n-btn"),
            &s.steps[0].dom_snapshot,
        )
        .unwrap();
        let back_ann: DesignAnnotation =
            serde_json::from_str(&serde_json::to_string(&ann).unwrap()).unwrap();
        assert_eq!(back_ann, ann);
        let verdict = acc.evaluate_functional(&s.steps[1].resulting_state);
        let back_v: Verdict =
            serde_json::from_str(&serde_json::to_string(&verdict).unwrap()).unwrap();
        assert_eq!(back_v, verdict);
    }
    #[test]
    fn json_schema_generates_for_the_core_types() {
        for schema in [
            json_schema::<BrowserStep>(),
            json_schema::<BrowserSession>(),
            json_schema::<VisualAcceptance>(),
            json_schema::<DesignAnnotation>(),
            json_schema::<Verdict>(),
        ] {
            assert!(schema.is_object(), "each schema is a JSON object");
            assert!(schema.get("$schema").is_some() || schema.get("title").is_some());
        }
    }
    #[test]
    fn browser_action_serializes_internally_tagged() {
        let value = serde_json::to_value(BrowserAction::Fill {
            value: "hi".into(),
        })
        .unwrap();
        assert_eq!(value.get("kind").unwrap(), "fill");
        assert_eq!(value.get("value").unwrap(), "hi");
    }
}

// --- inlined browser/a11y.rs ---
pub mod a11y {
//! Captured accessibility tree: the semantic view the platform exposes.
//!
//! This is the tree an assistive technology would see. It is captured alongside
//! the DOM so a11y requirements can be graded deterministically (does a control
//! have an accessible name? is the expected role present?) without a renderer.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::browser::ids::AccessibilityNodeId;

/// One node in the accessibility tree. `role` is an ARIA role string
/// (`button`, `link`, `textbox`, ...); `name` is the computed accessible name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AccessibilityNode {
    pub id: AccessibilityNodeId,
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    /// ARIA states, ordered for determinism (for example `disabled`,
    /// `checked`, `expanded`).
    #[serde(default)]
    pub states: Vec<String>,
    #[serde(default)]
    pub children: Vec<AccessibilityNode>,
}

impl AccessibilityNode {
    pub fn new(id: impl Into<String>, role: impl Into<String>) -> Self {
        Self {
            id: AccessibilityNodeId::new(id),
            role: role.into(),
            name: None,
            value: None,
            states: Vec::new(),
            children: Vec::new(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_child(mut self, child: AccessibilityNode) -> Self {
        self.children.push(child);
        self
    }

    /// Visit this node and every descendant, in depth-first order.
    pub fn walk<'a>(&'a self, visit: &mut dyn FnMut(&'a AccessibilityNode)) {
        visit(self);
        for child in &self.children {
            child.walk(visit);
        }
    }
}

/// The captured accessibility tree at one step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AccessibilityTree {
    pub root: AccessibilityNode,
}

impl AccessibilityTree {
    pub fn new(root: AccessibilityNode) -> Self {
        Self { root }
    }

    /// Every node with the given role.
    pub fn nodes_with_role<'a>(&'a self, role: &str) -> Vec<&'a AccessibilityNode> {
        let mut out = Vec::new();
        self.root.walk(&mut |n| {
            if n.role == role {
                out.push(n);
            }
        });
        out
    }

    /// True if any node carries the given role.
    pub fn has_role(&self, role: &str) -> bool {
        !self.nodes_with_role(role).is_empty()
    }

    /// Find a node by id.
    pub fn find(&self, id: &AccessibilityNodeId) -> Option<&AccessibilityNode> {
        let mut found = None;
        self.root.walk(&mut |n| {
            if &n.id == id {
                found = Some(n);
            }
        });
        found
    }
}

/// The set of roles considered interactive for the "no unnamed interactive
/// control" a11y check. Spec-derived from the ARIA roles model (an open W3C
/// spec); no proprietary source is copied.
pub const INTERACTIVE_ROLES: &[&str] = &[
    "button",
    "link",
    "textbox",
    "checkbox",
    "radio",
    "combobox",
    "listbox",
    "menuitem",
    "switch",
    "slider",
    "tab",
    "searchbox",
];

pub fn is_interactive_role(role: &str) -> bool {
    INTERACTIVE_ROLES.contains(&role)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn walk_and_role_queries() {
        let tree = AccessibilityTree::new(
            AccessibilityNode::new("ax-root", "document")
                .with_child(AccessibilityNode::new("ax-btn", "button").with_name("Add to cart"))
                .with_child(AccessibilityNode::new("ax-link", "link").with_name("Home")),
        );
        assert!(tree.has_role("button"));
        assert!(!tree.has_role("slider"));
        assert_eq!(tree.nodes_with_role("button").len(), 1);
        assert_eq!(tree.find(&AccessibilityNodeId::from("ax-btn")) .and_then(|n| n.name.clone()), Some("Add to cart".into()));
    }
    #[test]
    fn interactive_role_classification() {
        assert!(is_interactive_role("button"));
        assert!(!is_interactive_role("document"));
    }
}
}


// --- inlined browser/acceptance.rs ---
pub mod acceptance {
//! The visual acceptance oracle (Bible Book VIII sec 27).
//!
//! A [`VisualAcceptance`] is the spec a change is graded against: a target
//! screenshot/annotation, responsive states, semantic requirements, functional
//! interactions, a11y requirements, tolerances, and before/after artifacts. It
//! is a superset of what any single evaluator reads.
//!
//! This crate ships the DETERMINISTIC halves of the oracle:
//!
//! - [`VisualAcceptance::evaluate_functional`] grades the `functional_interactions`
//!   against a recorded [`ResultingState`] and returns a typed [`Verdict`].
//! - [`VisualAcceptance::evaluate_a11y`] grades the `a11y_requirements` against a
//!   recorded [`AccessibilityTree`].
//!
//! The pixel and layout halves -- comparing a captured screenshot to the target
//! within `tolerances`, and judging the `semantic_requirements` (does it *look*
//! right, does the responsive layout match) -- need a real renderer and/or a
//! vision model. Those are marked DEFERRED_MODEL_REQUIRED at
//! [`VisualAcceptance::evaluate_visual`] and are NOT implemented here.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::browser::a11y::{is_interactive_role, AccessibilityTree};
use crate::browser::evidence::{ResultingState, Viewport};
use crate::browser::ids::ArtifactRef;

/// A typed predicate over a [`ResultingState`]. Internally tagged on `type`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StateCheck {
    UrlEquals { url: String },
    UrlContains { fragment: String },
    HttpStatus { expected: u16 },
    SignalEquals { key: String, value: String },
    SignalPresent { key: String },
    ElementPresent { selector: String },
    ElementAbsent { selector: String },
    TextPresent { text: String },
    NoConsoleErrors,
    ConsoleErrorsAtMost { max: u32 },
}

impl StateCheck {
    /// Grade this check against a state. `Ok(())` on pass; the typed failure
    /// kind on fail.
    fn grade(&self, state: &ResultingState) -> std::result::Result<(), FailureKind> {
        match self {
            StateCheck::UrlEquals { url } => {
                if &state.url == url {
                    Ok(())
                } else {
                    Err(FailureKind::UrlMismatch {
                        expected: url.clone(),
                        actual: state.url.clone(),
                    })
                }
            }
            StateCheck::UrlContains { fragment } => {
                if state.url.contains(fragment) {
                    Ok(())
                } else {
                    Err(FailureKind::UrlFragmentMissing {
                        fragment: fragment.clone(),
                        actual: state.url.clone(),
                    })
                }
            }
            StateCheck::HttpStatus { expected } => {
                if state.http_status == Some(*expected) {
                    Ok(())
                } else {
                    Err(FailureKind::StatusMismatch {
                        expected: *expected,
                        actual: state.http_status,
                    })
                }
            }
            StateCheck::SignalEquals { key, value } => match state.signals.get(key) {
                Some(v) if v == value => Ok(()),
                Some(v) => Err(FailureKind::SignalMismatch {
                    key: key.clone(),
                    expected: value.clone(),
                    actual: v.clone(),
                }),
                None => Err(FailureKind::SignalMissing { key: key.clone() }),
            },
            StateCheck::SignalPresent { key } => {
                if state.signals.contains_key(key) {
                    Ok(())
                } else {
                    Err(FailureKind::SignalMissing { key: key.clone() })
                }
            }
            StateCheck::ElementPresent { selector } => {
                if state.present_selectors.contains(selector) {
                    Ok(())
                } else {
                    Err(FailureKind::ElementMissing {
                        selector: selector.clone(),
                    })
                }
            }
            StateCheck::ElementAbsent { selector } => {
                if state.present_selectors.contains(selector) {
                    Err(FailureKind::ElementUnexpected {
                        selector: selector.clone(),
                    })
                } else {
                    Ok(())
                }
            }
            StateCheck::TextPresent { text } => {
                if state.visible_text.iter().any(|t| t.contains(text)) {
                    Ok(())
                } else {
                    Err(FailureKind::TextMissing { text: text.clone() })
                }
            }
            StateCheck::NoConsoleErrors => {
                if state.console_error_count == 0 {
                    Ok(())
                } else {
                    Err(FailureKind::ConsoleErrors {
                        count: state.console_error_count,
                        allowed: 0,
                    })
                }
            }
            StateCheck::ConsoleErrorsAtMost { max } => {
                if state.console_error_count <= *max {
                    Ok(())
                } else {
                    Err(FailureKind::ConsoleErrors {
                        count: state.console_error_count,
                        allowed: *max,
                    })
                }
            }
        }
    }
}

/// A typed predicate over an [`AccessibilityTree`]. Internally tagged on `type`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum A11yCheck {
    /// Some node carries this role.
    RolePresent { role: String },
    /// Some node with this role has a non-empty accessible name.
    RoleNamed { role: String },
    /// Every interactive node has a non-empty accessible name.
    NoUnnamedInteractive,
    /// A node with this role has exactly this accessible name.
    NameEquals { role: String, name: String },
}

impl A11yCheck {
    fn grade(&self, tree: &AccessibilityTree) -> std::result::Result<(), FailureKind> {
        match self {
            A11yCheck::RolePresent { role } => {
                if tree.has_role(role) {
                    Ok(())
                } else {
                    Err(FailureKind::A11yRoleMissing { role: role.clone() })
                }
            }
            A11yCheck::RoleNamed { role } => {
                let named = tree
                    .nodes_with_role(role)
                    .iter()
                    .any(|n| n.name.as_deref().map(|s| !s.is_empty()).unwrap_or(false));
                if named {
                    Ok(())
                } else if tree.has_role(role) {
                    Err(FailureKind::A11yUnnamed { role: role.clone() })
                } else {
                    Err(FailureKind::A11yRoleMissing { role: role.clone() })
                }
            }
            A11yCheck::NoUnnamedInteractive => {
                let mut offender = None;
                tree.root.walk(&mut |n| {
                    if offender.is_none()
                        && is_interactive_role(&n.role)
                        && n.name.as_deref().map(|s| s.is_empty()).unwrap_or(true)
                    {
                        offender = Some(n.role.clone());
                    }
                });
                match offender {
                    Some(role) => Err(FailureKind::A11yUnnamed { role }),
                    None => Ok(()),
                }
            }
            A11yCheck::NameEquals { role, name } => {
                let matches = tree
                    .nodes_with_role(role)
                    .iter()
                    .any(|n| n.name.as_deref() == Some(name.as_str()));
                if matches {
                    Ok(())
                } else {
                    let actual = tree
                        .nodes_with_role(role)
                        .first()
                        .and_then(|n| n.name.clone());
                    Err(FailureKind::A11yNameMismatch {
                        role: role.clone(),
                        expected: name.clone(),
                        actual,
                    })
                }
            }
        }
    }
}

/// A functional interaction requirement: an id + prose + the typed check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FunctionalRequirement {
    pub id: String,
    pub description: String,
    pub check: StateCheck,
}

/// An accessibility requirement: an id + prose + the typed check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct A11yRequirement {
    pub id: String,
    pub description: String,
    pub check: A11yCheck,
}

/// A semantic (appearance) requirement. Graded by a renderer/vision comparator,
/// which is DEFERRED_MODEL_REQUIRED; carried here as the acceptance spec so a
/// later evaluator has it. The functional structure it implies should be
/// expressed as [`FunctionalRequirement`]s, which ARE graded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SemanticRequirement {
    pub id: String,
    pub description: String,
}

/// A named responsive target the change must satisfy, with the reference
/// screenshot to compare against (comparison is DEFERRED_MODEL_REQUIRED).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResponsiveTarget {
    pub name: String,
    pub viewport: Viewport,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screenshot_ref: Option<ArtifactRef>,
}

/// Comparison tolerances for the DEFERRED visual comparator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Tolerances {
    /// Maximum fraction of differing pixels allowed (0.0 = exact).
    pub pixel_diff_ratio: f64,
    /// Maximum cumulative layout shift allowed.
    pub layout_shift: f64,
    /// Per-channel color delta allowed (0-255).
    pub color_delta: u8,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            pixel_diff_ratio: 0.0,
            layout_shift: 0.0,
            color_delta: 0,
        }
    }
}

/// The before/after artifact pair for a change (both referenced, not inlined).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BeforeAfter {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before: Option<ArtifactRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<ArtifactRef>,
}

/// The full acceptance spec a change is graded against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct VisualAcceptance {
    pub id: String,
    /// The target screenshot or annotated design the change must match,
    /// referenced (not inlined). Graded by the DEFERRED visual comparator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<ArtifactRef>,
    #[serde(default)]
    pub responsive_states: Vec<ResponsiveTarget>,
    #[serde(default)]
    pub semantic_requirements: Vec<SemanticRequirement>,
    #[serde(default)]
    pub functional_interactions: Vec<FunctionalRequirement>,
    #[serde(default)]
    pub a11y_requirements: Vec<A11yRequirement>,
    #[serde(default)]
    pub tolerances: Tolerances,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_after: Option<BeforeAfter>,
}

impl VisualAcceptance {
    /// Grade the functional interactions against a recorded resulting state.
    /// Deterministic: no renderer, no model.
    pub fn evaluate_functional(&self, state: &ResultingState) -> Verdict {
        let mut reasons = Vec::new();
        for req in &self.functional_interactions {
            if let Err(kind) = req.check.grade(state) {
                reasons.push(FailureReason {
                    requirement_id: req.id.clone(),
                    detail: req.description.clone(),
                    kind,
                });
            }
        }
        Verdict::from_reasons(reasons)
    }

    /// Grade the a11y requirements against a recorded accessibility tree.
    /// Deterministic: no renderer, no model.
    pub fn evaluate_a11y(&self, tree: &AccessibilityTree) -> Verdict {
        let mut reasons = Vec::new();
        for req in &self.a11y_requirements {
            if let Err(kind) = req.check.grade(tree) {
                reasons.push(FailureReason {
                    requirement_id: req.id.clone(),
                    detail: req.description.clone(),
                    kind,
                });
            }
        }
        Verdict::from_reasons(reasons)
    }

    /// DEFERRED_MODEL_REQUIRED: compare a captured screenshot to `target` (and
    /// each `responsive_states` reference) within `tolerances`, and judge the
    /// `semantic_requirements`. This needs a real renderer and/or a vision
    /// model; it is intentionally NOT implemented here and must not be claimed.
    /// Callers that need pixel/appearance grading route to a model-bearing
    /// service outside this crate.
    pub fn evaluate_visual(&self) -> ! {
        unimplemented!(
            "DEFERRED_MODEL_REQUIRED: pixel/appearance grading needs a renderer or vision model"
        )
    }
}

/// A typed reason a requirement failed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FailureKind {
    UrlMismatch { expected: String, actual: String },
    UrlFragmentMissing { fragment: String, actual: String },
    StatusMismatch { expected: u16, actual: Option<u16> },
    SignalMismatch { key: String, expected: String, actual: String },
    SignalMissing { key: String },
    ElementMissing { selector: String },
    ElementUnexpected { selector: String },
    TextMissing { text: String },
    ConsoleErrors { count: u32, allowed: u32 },
    A11yRoleMissing { role: String },
    A11yUnnamed { role: String },
    A11yNameMismatch { role: String, expected: String, actual: Option<String> },
}

/// A single failed requirement: which one, why (typed), and its prose.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FailureReason {
    pub requirement_id: String,
    pub kind: FailureKind,
    pub detail: String,
}

/// The outcome of a deterministic acceptance evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Verdict {
    Pass,
    Fail { reasons: Vec<FailureReason> },
}

impl Verdict {
    fn from_reasons(reasons: Vec<FailureReason>) -> Self {
        if reasons.is_empty() {
            Verdict::Pass
        } else {
            Verdict::Fail { reasons }
        }
    }

    pub fn is_pass(&self) -> bool {
        matches!(self, Verdict::Pass)
    }

    pub fn reasons(&self) -> &[FailureReason] {
        match self {
            Verdict::Pass => &[],
            Verdict::Fail { reasons } => reasons,
        }
    }
}
}


// --- inlined browser/design_mode.rs ---
pub mod design_mode {
//! Design Mode annotations (Bible Book VIII sec 27).
//!
//! Design Mode is the "click an element, see everything about it" surface: a
//! selection on the page maps to its DOM node, the source symbol that emitted
//! it, the CSS rule that styled it, its layout box, and its accessibility node.
//! [`annotate`] builds that mapping deterministically from a captured
//! [`DomSnapshot`]: everything it needs was recorded, so no live browser is
//! involved.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::browser::dom::{BoxModel, CssRule, DomSnapshot, SourceSymbol};
use crate::browser::error::{BrowserError, Result};
use crate::browser::evidence::ElementSelector;
use crate::browser::ids::{AccessibilityNodeId, DomNodeId};

/// The full mapping from a selected element to what Design Mode shows about it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DesignAnnotation {
    /// The selection that produced this annotation.
    pub selected: ElementSelector,
    /// The DOM node the selection resolved to.
    pub dom_node: DomNodeId,
    /// The source symbol that emitted the node, when a source map was captured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_symbol: Option<SourceSymbol>,
    /// The most-specific CSS rule that applied, when captured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub css_rule: Option<CssRule>,
    /// The element's layout box.
    pub box_model: BoxModel,
    /// The accessibility node the element maps to, when captured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub a11y_node: Option<AccessibilityNodeId>,
}

/// Build a [`DesignAnnotation`] for a selection against a captured DOM snapshot.
///
/// Resolution order: if the selector already carries a resolved `dom_node`, that
/// node is used; otherwise the selector is resolved structurally against the
/// snapshot (`#id`, `.class`, test id, tag, or text). Fails with
/// [`BrowserError::UnresolvedSelection`] if no node matches.
pub fn annotate(selector: &ElementSelector, dom: &DomSnapshot) -> Result<DesignAnnotation> {
    let node = match &selector.dom_node {
        Some(id) => dom
            .find(id)
            .ok_or_else(|| BrowserError::UnresolvedSelection {
                selector: id.to_string(),
            })?,
        None => dom.resolve(selector.strategy, &selector.query).ok_or_else(|| {
            BrowserError::UnresolvedSelection {
                selector: selector.query.clone(),
            }
        })?,
    };

    Ok(DesignAnnotation {
        selected: selector.clone(),
        dom_node: node.id.clone(),
        source_symbol: node.source_symbol.clone(),
        css_rule: node.css_rules.first().cloned(),
        box_model: node.box_model.clone().unwrap_or_default(),
        a11y_node: node.a11y_node.clone(),
    })
}
}


// --- inlined browser/dom.rs ---
pub mod dom {
//! Captured DOM snapshot: a small, queryable tree of the page at one step.
//!
//! The snapshot is intentionally a structured tree (not raw HTML text) so it is
//! deterministically queryable: a selection resolves to a [`DomNode`], and a
//! Design Mode annotation reads that node's box, source symbol, and CSS. The
//! full raw HTML, when kept, is referenced by [`DomSnapshot::html_ref`] as an
//! artifact rather than inlined.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::browser::ids::{AccessibilityNodeId, ArtifactRef, DomNodeId};

/// The layout box of an element, in CSS pixels, relative to the viewport. The
/// content box plus the surrounding edges the browser exposes in Design Mode.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BoxModel {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    #[serde(default)]
    pub padding: EdgeSizes,
    #[serde(default)]
    pub border: EdgeSizes,
    #[serde(default)]
    pub margin: EdgeSizes,
}

/// Top/right/bottom/left edge sizes for padding, border, or margin.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EdgeSizes {
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
    pub left: f64,
}

/// A CSS rule that applies to a node, carried for Design Mode. `source` names
/// the stylesheet origin (a file path or `<style>` marker) when known.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CssRule {
    pub selector: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Declared properties, ordered for determinism.
    #[serde(default)]
    pub declarations: BTreeMap<String, String>,
}

/// A dev-time source mapping from a rendered node back to the code that emitted
/// it (for example a JSX component and line). Populated only by tooling that has
/// a source map; absent in production captures.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SourceSymbol {
    /// The source file that declared the element (repository-relative).
    pub file: String,
    /// The symbol name (component, function, or selector) that emitted it.
    pub symbol: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub column: Option<u32>,
}

/// One node in the captured DOM tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DomNode {
    pub id: DomNodeId,
    /// Lower-case tag name, for example `button`.
    pub tag: String,
    /// Attributes present on the element, ordered for determinism. `id`,
    /// `class`, and `data-testid` participate in selector resolution.
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
    /// The element's own text (not its descendants').
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// The layout box, when captured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub box_model: Option<BoxModel>,
    /// The accessibility node this element maps to, when captured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub a11y_node: Option<AccessibilityNodeId>,
    /// The source symbol that emitted this node, for Design Mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_symbol: Option<SourceSymbol>,
    /// CSS rules that apply, most-specific first, for Design Mode.
    #[serde(default)]
    pub css_rules: Vec<CssRule>,
    #[serde(default)]
    pub children: Vec<DomNode>,
}

impl DomNode {
    /// A leaf node with just an id and tag; other fields default. A convenience
    /// for building fixtures.
    pub fn leaf(id: impl Into<String>, tag: impl Into<String>) -> Self {
        Self {
            id: DomNodeId::new(id),
            tag: tag.into(),
            attributes: BTreeMap::new(),
            text: None,
            box_model: None,
            a11y_node: None,
            source_symbol: None,
            css_rules: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Depth-first search for a descendant (or self) with the given id.
    pub fn find(&self, id: &DomNodeId) -> Option<&DomNode> {
        if &self.id == id {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.find(id) {
                return Some(found);
            }
        }
        None
    }

    fn matches_selector(&self, strategy: SelectorMatch<'_>) -> bool {
        match strategy {
            SelectorMatch::Id(v) => self.attributes.get("id").map(|s| s.as_str()) == Some(v),
            SelectorMatch::TestId(v) => {
                self.attributes.get("data-testid").map(|s| s.as_str()) == Some(v)
            }
            SelectorMatch::Tag(v) => self.tag == v,
            SelectorMatch::Class(v) => self
                .attributes
                .get("class")
                .map(|s| s.split_whitespace().any(|c| c == v))
                .unwrap_or(false),
            SelectorMatch::Text(v) => self.text.as_deref().map(|t| t.contains(v)).unwrap_or(false),
        }
    }

    fn find_matching(&self, strategy: SelectorMatch<'_>) -> Option<&DomNode> {
        if self.matches_selector(strategy) {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.find_matching(strategy) {
                return Some(found);
            }
        }
        None
    }
}

/// A parsed selector kind used for structural resolution against the snapshot.
#[derive(Debug, Clone, Copy)]
enum SelectorMatch<'a> {
    Id(&'a str),
    Class(&'a str),
    TestId(&'a str),
    Tag(&'a str),
    Text(&'a str),
}

/// The captured DOM at one step: a structured tree plus an optional reference to
/// the raw HTML blob.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DomSnapshot {
    pub root: DomNode,
    /// The full serialized HTML, referenced (never inlined).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub html_ref: Option<ArtifactRef>,
    /// Total node count, for a cheap size signal without walking the tree.
    #[serde(default)]
    pub node_count: u32,
}

impl DomSnapshot {
    pub fn new(root: DomNode) -> Self {
        let node_count = count_nodes(&root);
        Self {
            root,
            html_ref: None,
            node_count,
        }
    }

    /// Find a node by id anywhere in the tree.
    pub fn find(&self, id: &DomNodeId) -> Option<&DomNode> {
        self.root.find(id)
    }

    /// Resolve a CSS-ish or role/text selector to a node. Supports `#id`,
    /// `.class`, `[data-testid=...]` (also the bare test-id string), a bare tag,
    /// and a `text=` prefix. This is a deliberately small resolver: real
    /// full-CSS matching against a live renderer is DEFERRED_MODEL_REQUIRED.
    pub fn resolve(&self, strategy: SelectStrategy, query: &str) -> Option<&DomNode> {
        let m = match strategy {
            SelectStrategy::TestId => SelectorMatch::TestId(query),
            SelectStrategy::Text => SelectorMatch::Text(query),
            SelectStrategy::Role => SelectorMatch::Tag(query),
            SelectStrategy::XPath => return None, // out of scope for the small resolver
            SelectStrategy::Css => {
                if let Some(rest) = query.strip_prefix('#') {
                    SelectorMatch::Id(rest)
                } else if let Some(rest) = query.strip_prefix('.') {
                    SelectorMatch::Class(rest)
                } else if let Some(rest) = query.strip_prefix("[data-testid=") {
                    let v = rest.trim_end_matches(']').trim_matches(|c| c == '"' || c == '\'');
                    SelectorMatch::TestId(v)
                } else {
                    SelectorMatch::Tag(query)
                }
            }
        };
        self.root.find_matching(m)
    }
}

fn count_nodes(node: &DomNode) -> u32 {
    1 + node.children.iter().map(count_nodes).sum::<u32>()
}

/// How a selector string is interpreted. Kept here (rather than in `evidence`)
/// because both the evidence records and the DOM resolver need it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SelectStrategy {
    Css,
    XPath,
    Role,
    Text,
    TestId,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tree() -> DomSnapshot {
        let mut button = DomNode::leaf("n-btn", "button");
        button
            .attributes
            .insert("id".into(), "add-to-cart".into());
        button
            .attributes
            .insert("data-testid".into(), "add-cart".into());
        button.text = Some("Add to cart".into());
        let mut root = DomNode::leaf("n-root", "div");
        root.children.push(button);
        DomSnapshot::new(root)
    }
    #[test]
    fn find_by_id_walks_the_tree() {
        let snap = tree();
        let n = snap.find(&DomNodeId::from("n-btn")).unwrap();
        assert_eq!(n.tag, "button");
        assert!(snap.find(&DomNodeId::from("n-missing")).is_none());
    }
    #[test]
    fn resolve_supports_id_testid_and_text() {
        let snap = tree();
        assert_eq!(snap.resolve(SelectStrategy::Css, "#add-to-cart").unwrap().id, DomNodeId::from("n-btn"));
        assert_eq!(snap.resolve(SelectStrategy::TestId, "add-cart").unwrap().id, DomNodeId::from("n-btn"));
 assert_eq!( snap.resolve(SelectStrategy::Text, "Add to").unwrap().id, DomNodeId::from("n-btn") );
        assert!(snap.resolve(SelectStrategy::Css, "#nope").is_none());
    }
    #[test]
    fn node_count_is_captured() {
        assert_eq!(tree().node_count, 2);
    }
}
}


// --- inlined browser/driver.rs ---
pub mod driver {
//! The browser driver trait and a deterministic replay driver.
//!
//! [`BrowserDriver`] is the seam every browser backend implements: the action
//! verbs (navigate, click, fill) and the observer verbs (screenshot, dom,
//! accessibility, console, network). Backends return owned evidence so the trait
//! stays object-safe and implementable by both a replay driver and (later) a
//! live one.
//!
//! [`ReplayDriver`] is the only backend in this crate. It plays a recorded
//! [`BrowserSession`] back, step by step, verifying that each driver call
//! matches the next recorded step. The action verbs advance a cursor and return
//! the played step; the observer verbs read the evidence captured at the current
//! step. This makes a recorded trace a strict, deterministic contract that a
//! test can drive without a browser.
//!
//! DEFERRED_MODEL_REQUIRED-adjacent: a real chromium/CDP backend that produces
//! these records from a live page is out of scope for this crate and is NOT
//! implemented or claimed here. It would implement this same trait, most likely
//! async, wrapping the transport; nothing about the evidence model depends on
//! it.

use crate::browser::a11y::AccessibilityTree;
use crate::browser::dom::DomSnapshot;
use crate::browser::error::{BrowserError, Result};
use crate::browser::evidence::{
    BrowserAction, BrowserSession, BrowserStep, ConsoleEvent, ElementSelector, NetworkEvent,
};
use crate::browser::ids::ArtifactRef;

/// The seam a browser backend implements. Action verbs mutate the page and
/// return the resulting evidence step; observer verbs read evidence about the
/// current step without changing it.
pub trait BrowserDriver {
    /// Navigate to a URL. Returns the resulting evidence step.
    fn navigate(&mut self, url: &str) -> Result<BrowserStep>;

    /// Click the element the selector locates. Returns the resulting step.
    fn click(&mut self, selector: &ElementSelector) -> Result<BrowserStep>;

    /// Fill the located element with `value`. Returns the resulting step.
    fn fill(&mut self, selector: &ElementSelector, value: &str) -> Result<BrowserStep>;

    /// The screenshot reference for the current step.
    fn screenshot(&self) -> Result<ArtifactRef>;

    /// The DOM snapshot captured at the current step.
    fn dom(&self) -> Result<DomSnapshot>;

    /// The accessibility tree captured at the current step.
    fn accessibility(&self) -> Result<AccessibilityTree>;

    /// The console events captured at the current step.
    fn console(&self) -> Result<Vec<ConsoleEvent>>;

    /// The network events captured at the current step.
    fn network(&self) -> Result<Vec<NetworkEvent>>;
}

/// A deterministic driver that replays a recorded [`BrowserSession`].
///
/// The driver holds the recorded steps and a cursor at the last-played step
/// (`None` before the first call). Each action verb consumes the next recorded
/// step, checks that the call matches what was recorded, advances the cursor,
/// and returns the played step. Observer verbs read the current step.
#[derive(Debug, Clone)]
pub struct ReplayDriver {
    steps: Vec<BrowserStep>,
    /// Index of the last-played step; `None` before anything is played.
    cursor: Option<usize>,
}

impl ReplayDriver {
    pub fn new(session: BrowserSession) -> Self {
        Self {
            steps: session.steps,
            cursor: None,
        }
    }

    /// The index of the step a subsequent action verb would play.
    fn next_index(&self) -> usize {
        match self.cursor {
            None => 0,
            Some(i) => i + 1,
        }
    }

    /// The step most recently played, for the observer verbs.
    pub fn current_step(&self) -> Option<&BrowserStep> {
        self.cursor.and_then(|i| self.steps.get(i))
    }

    fn require_current(&self) -> Result<&BrowserStep> {
        self.current_step().ok_or(BrowserError::NoCurrentStep)
    }

    /// Whether every recorded step has been played.
    pub fn is_exhausted(&self) -> bool {
        self.next_index() >= self.steps.len()
    }

    /// The number of recorded steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Take the next step, enforcing that `requested` is the verb the recorded
    /// action expects. `matches` decides whether the recorded action is the one
    /// the caller asked for; `describe_expected` renders the recorded action for
    /// a mismatch error.
    fn play_next(
        &mut self,
        requested: &'static str,
        matches: impl FnOnce(&BrowserStep) -> bool,
    ) -> Result<BrowserStep> {
        let idx = self.next_index();
        let step = self
            .steps
            .get(idx)
            .ok_or(BrowserError::ReplayExhausted { requested })?;
        if !matches(step) {
            return Err(BrowserError::ReplayMismatch {
                index: idx,
                requested,
                expected: step.action.label().to_string(),
                detail: format!("recorded url={:?} action={}", step.url, step.action.label()),
            });
        }
        self.cursor = Some(idx);
        Ok(self.steps[idx].clone())
    }
}

impl BrowserDriver for ReplayDriver {
    fn navigate(&mut self, url: &str) -> Result<BrowserStep> {
        self.play_next("navigate", |step| {
            matches!(step.action, BrowserAction::Navigate) && step.url == url
        })
    }

    fn click(&mut self, selector: &ElementSelector) -> Result<BrowserStep> {
        self.play_next("click", |step| {
            matches!(step.action, BrowserAction::Click)
                && step
                    .selected_element
                    .as_ref()
                    .map(|e| e.same_target(selector))
                    .unwrap_or(false)
        })
    }

    fn fill(&mut self, selector: &ElementSelector, value: &str) -> Result<BrowserStep> {
        self.play_next("fill", |step| {
            matches!(&step.action, BrowserAction::Fill { value: v } if v == value)
                && step
                    .selected_element
                    .as_ref()
                    .map(|e| e.same_target(selector))
                    .unwrap_or(false)
        })
    }

    fn screenshot(&self) -> Result<ArtifactRef> {
        let step = self.require_current()?;
        step.screenshot_ref
            .clone()
            .ok_or(BrowserError::MissingEvidence { what: "screenshot" })
    }

    fn dom(&self) -> Result<DomSnapshot> {
        Ok(self.require_current()?.dom_snapshot.clone())
    }

    fn accessibility(&self) -> Result<AccessibilityTree> {
        Ok(self.require_current()?.accessibility_tree.clone())
    }

    fn console(&self) -> Result<Vec<ConsoleEvent>> {
        Ok(self.require_current()?.console_events.clone())
    }

    fn network(&self) -> Result<Vec<NetworkEvent>> {
        Ok(self.require_current()?.network_events.clone())
    }
}
}


// --- inlined browser/error.rs ---
pub mod error {
//! Errors surfaced by the browser evidence and verification layer.

use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum BrowserError {
    /// The replay ran off the end of the recorded session: a driver call was
    /// made with no further recorded step to satisfy it.
    #[error("replay exhausted: no recorded step remains for the requested {requested} call")]
    ReplayExhausted { requested: &'static str },

    /// A driver call did not match the next recorded step. The replay driver is
    /// a strict contract: the caller must drive the exact recorded sequence.
    #[error("replay mismatch at step {index}: expected {expected}, got a {requested} call ({detail})")]
    ReplayMismatch {
        index: usize,
        requested: &'static str,
        expected: String,
        detail: String,
    },

    /// An observer method was called before any step had been played.
    #[error("no current step: play a navigate/click/fill step before reading evidence")]
    NoCurrentStep,

    /// The current step did not capture the requested piece of evidence.
    #[error("the current step captured no {what}")]
    MissingEvidence { what: &'static str },

    /// A selection could not be resolved to a node in the given snapshot.
    #[error("could not resolve selection {selector:?} to a DOM node")]
    UnresolvedSelection { selector: String },

    #[error("serialization error: {0}")]
    Serde(String),
}

impl From<serde_json::Error> for BrowserError {
    fn from(e: serde_json::Error) -> Self {
        BrowserError::Serde(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, BrowserError>;
}


// --- inlined browser/evidence.rs ---
pub mod evidence {
//! The browser evidence model (Bible Book VIII sec 27).
//!
//! A [`BrowserStep`] is the full evidence record for one action the agent (or a
//! recorder) took against a page: what happened, why, and everything observable
//! afterward. A [`BrowserSession`] is an ordered list of steps -- the replayable
//! trace. Heavy payloads (screenshots, raw HTML, network bodies) are referenced
//! by [`ArtifactRef`], never inlined, so a step stays small and a trace stays
//! cheap to store and diff.
//!
//! This is a schema layer: it captures and structures evidence. It runs
//! nothing. A real driver that produces these records from a live browser is
//! out of scope here (see [`crate::browser::driver`]).

use std::collections::{BTreeMap, BTreeSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::browser::a11y::AccessibilityTree;
use crate::browser::dom::{DomSnapshot, SelectStrategy};
use crate::browser::ids::{ArtifactRef, BrowserSessionId};

/// Why a navigation happened. Captured on every step; steps that did not
/// navigate carry [`NavigationCause::None`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NavigationCause {
    /// The user (or agent) navigated directly to a URL.
    UserNavigate,
    /// A link click caused the navigation.
    LinkClick,
    /// A form submission caused the navigation.
    FormSubmit,
    /// A server or meta redirect.
    Redirect,
    /// Script (`location.assign`, history API) caused it.
    ScriptNavigation,
    /// Back in history.
    HistoryBack,
    /// Forward in history.
    HistoryForward,
    /// A reload.
    Reload,
    /// No navigation occurred on this step.
    None,
}

/// How an element was located.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ElementSelector {
    pub strategy: SelectStrategy,
    /// The selector query text (a CSS selector, role name, visible text, or
    /// test id) interpreted per `strategy`.
    pub query: String,
    /// The DOM node the selector resolved to, when the recorder captured it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dom_node: Option<crate::browser::ids::DomNodeId>,
}

impl ElementSelector {
    pub fn css(query: impl Into<String>) -> Self {
        Self {
            strategy: SelectStrategy::Css,
            query: query.into(),
            dom_node: None,
        }
    }

    pub fn test_id(query: impl Into<String>) -> Self {
        Self {
            strategy: SelectStrategy::TestId,
            query: query.into(),
            dom_node: None,
        }
    }

    pub fn with_node(mut self, node: impl Into<String>) -> Self {
        self.dom_node = Some(crate::browser::ids::DomNodeId::new(node));
        self
    }

    /// Whether this selector targets the same element as `other`, matched on
    /// strategy and query. The resolved `dom_node` is not required to match (a
    /// request need not know it up front).
    pub fn same_target(&self, other: &ElementSelector) -> bool {
        self.strategy == other.strategy && self.query == other.query
    }
}

/// The operation a step performed. Internally tagged on `kind` so it reads as
/// `{ "kind": "click" }` on the wire. The target of a navigate is the step's
/// `url`; the target of a click/fill is the step's `selected_element`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BrowserAction {
    Navigate,
    Click,
    Fill { value: String },
    Screenshot,
    ReadDom,
    ReadAccessibility,
    ReadConsole,
    ReadNetwork,
    Wait { ms: u64 },
    Custom { name: String },
}

impl BrowserAction {
    /// A short human label used in replay-mismatch diagnostics.
    pub fn label(&self) -> &'static str {
        match self {
            BrowserAction::Navigate => "navigate",
            BrowserAction::Click => "click",
            BrowserAction::Fill { .. } => "fill",
            BrowserAction::Screenshot => "screenshot",
            BrowserAction::ReadDom => "read_dom",
            BrowserAction::ReadAccessibility => "read_accessibility",
            BrowserAction::ReadConsole => "read_console",
            BrowserAction::ReadNetwork => "read_network",
            BrowserAction::Wait { .. } => "wait",
            BrowserAction::Custom { .. } => "custom",
        }
    }
}

/// Severity of a console message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleLevel {
    Log,
    Info,
    Warn,
    Error,
    Debug,
}

/// One console message captured during a step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ConsoleEvent {
    pub level: ConsoleLevel,
    pub text: String,
    pub timestamp_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// One network exchange captured during a step. Request and response bodies are
/// referenced by artifact id, never inlined.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NetworkEvent {
    pub request_id: String,
    pub method: String,
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    /// The request headers/body blob, referenced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_ref: Option<ArtifactRef>,
    /// The response body blob, referenced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_ref: Option<ArtifactRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing_ms: Option<u64>,
}

/// A named viewport size the page was observed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
}

/// The responsive state a step was captured in: which named breakpoint and
/// viewport were active.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResponsiveState {
    /// The breakpoint name (for example `mobile`, `tablet`, `desktop`).
    pub name: String,
    pub viewport: Viewport,
}

/// The observable post-action state a functional oracle grades against. This is
/// the deterministic, structured summary of "what is true now" -- distinct from
/// the raw DOM: it names the signals a check cares about.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResultingState {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub http_status: Option<u16>,
    /// Named application signals a functional check reads (for example
    /// `cart.count -> "1"`). Ordered for determinism.
    #[serde(default)]
    pub signals: BTreeMap<String, String>,
    /// Selectors observed present in the page after the action. A set for
    /// deterministic membership tests.
    #[serde(default)]
    pub present_selectors: BTreeSet<String>,
    /// Visible text fragments observed after the action.
    #[serde(default)]
    pub visible_text: Vec<String>,
    /// How many console errors were seen up to and including this step.
    #[serde(default)]
    pub console_error_count: u32,
    /// The responsive state this was captured in, when relevant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub responsive: Option<ResponsiveState>,
}

impl ResultingState {
    /// A minimal state at a URL.
    pub fn at(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            title: None,
            http_status: None,
            signals: BTreeMap::new(),
            present_selectors: BTreeSet::new(),
            visible_text: Vec::new(),
            console_error_count: 0,
            responsive: None,
        }
    }

    pub fn signal(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.signals.insert(key.into(), value.into());
        self
    }

    pub fn present(mut self, selector: impl Into<String>) -> Self {
        self.present_selectors.insert(selector.into());
        self
    }

    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.visible_text.push(text.into());
        self
    }
}

/// The full evidence record for one browser step (Bible Book VIII sec 27).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BrowserStep {
    /// Zero-based position within the session, for stable ordering.
    pub index: u32,
    /// The page URL after the step.
    pub url: String,
    pub navigation_cause: NavigationCause,
    pub action: BrowserAction,
    /// The element a click/fill targeted, when any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_element: Option<ElementSelector>,
    pub dom_snapshot: DomSnapshot,
    pub accessibility_tree: AccessibilityTree,
    /// The screenshot for this step, referenced (never inlined).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screenshot_ref: Option<ArtifactRef>,
    #[serde(default)]
    pub console_events: Vec<ConsoleEvent>,
    #[serde(default)]
    pub network_events: Vec<NetworkEvent>,
    pub resulting_state: ResultingState,
    /// Wall time this step took, in milliseconds.
    pub timing_ms: u64,
}

/// A recorded, replayable browser session: an ordered list of steps plus a
/// header. This is the fixture a [`crate::browser::driver::ReplayDriver`] plays back.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BrowserSession {
    pub id: BrowserSessionId,
    pub steps: Vec<BrowserStep>,
    pub created_ms: u64,
}

impl BrowserSession {
    pub fn new(id: impl Into<String>, steps: Vec<BrowserStep>) -> Self {
        Self {
            id: BrowserSessionId::new(id),
            steps,
            created_ms: 0,
        }
    }

    /// Whether the step indices are 0..len in order.
    pub fn indices_are_sequential(&self) -> bool {
        self.steps
            .iter()
            .enumerate()
            .all(|(i, s)| s.index as usize == i)
    }
}
}


// --- inlined browser/ids.rs ---
pub mod ids {
//! Typed identifiers and artifact references for the browser evidence model.
//!
//! Ids are transparent string newtypes: they serialize as bare strings and
//! generate a plain-string JSON Schema. This crate never mints an id or a live
//! artifact; a caller (a recorder, a fixture, or a real driver built later)
//! owns the values. That keeps the crate deterministic and model-free.
//!
//! [`ArtifactRef`] is the one non-newtype here: it is how heavy evidence
//! (screenshots, raw DOM HTML, network bodies) is carried by reference instead
//! of inlined. Bytes never live in the evidence records; only a content
//! addressed reference to a stored blob does.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

macro_rules! id_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(
            Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash,
            Serialize, Deserialize, JsonSchema,
        )]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            /// Wrap an existing id value. This crate never generates ids.
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_string())
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

id_newtype!(
    /// Identifies one recorded browser session (a `Vec<BrowserStep>` with a
    /// header).
    BrowserSessionId
);
id_newtype!(
    /// A stable id for a node inside a captured DOM snapshot. The recorder
    /// assigns it; a selection and a Design Mode annotation both resolve to it.
    DomNodeId
);
id_newtype!(
    /// A stable id for a node inside a captured accessibility tree.
    AccessibilityNodeId
);

/// A reference to a stored binary artifact: a screenshot, the raw DOM HTML, a
/// network request or response body. The evidence model carries these instead
/// of the bytes so a step record stays small and the heavy payload lives in a
/// blob store addressed by [`ArtifactRef::id`].
///
/// The id is content addressed when built with [`ArtifactRef::content_addressed`]
/// (a `blake3:` digest of the bytes), so the same bytes always produce the same
/// reference -- deterministic and de-duplicating. There is deliberately no field
/// on this type that can hold the bytes: inlining is impossible by construction.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct ArtifactRef {
    /// The blob address. `blake3:<hex>` when content addressed.
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
}

impl ArtifactRef {
    /// Reference a blob by an already-known id (for example one a recorder
    /// assigned). No bytes are involved.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            media_type: None,
            size_bytes: None,
        }
    }

    /// Build a content-addressed reference from the artifact bytes. The bytes
    /// are hashed and then dropped; only the digest, size, and media type are
    /// retained. Deterministic: identical bytes yield an identical reference.
    pub fn content_addressed(bytes: &[u8], media_type: Option<&str>) -> Self {
        let hex = blake3::hash(bytes).to_hex();
        Self {
            id: format!("blake3:{hex}"),
            media_type: media_type.map(|s| s.to_string()),
            size_bytes: Some(bytes.len() as u64),
        }
    }

    /// Whether this reference is a content-addressed `blake3:` digest.
    pub fn is_content_addressed(&self) -> bool {
        self.id.starts_with("blake3:")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ids_serialize_transparently_as_bare_strings() {
        let id = DomNodeId::from("n-42");
        assert_eq!(serde_json::to_string(&id).unwrap(), "\"n-42\"");
        let back: DomNodeId = serde_json::from_str("\"n-42\"").unwrap();
        assert_eq!(back, id);
    }
    #[test]
    fn content_addressed_ref_is_deterministic_and_holds_no_bytes() {
        let bytes = b"fake-png-bytes";
        let a = ArtifactRef::content_addressed(bytes, Some("image/png"));
        let b = ArtifactRef::content_addressed(bytes, Some("image/png"));
        assert_eq!(a, b, "same bytes -> same reference");
        assert!(a.is_content_addressed());
        assert_eq!(a.size_bytes, Some(bytes.len() as u64));
        let json = serde_json::to_string(&a).unwrap();
        assert!(json.contains("blake3:"));
        assert!(!json.contains("fake-png-bytes"));
    }
}
}

