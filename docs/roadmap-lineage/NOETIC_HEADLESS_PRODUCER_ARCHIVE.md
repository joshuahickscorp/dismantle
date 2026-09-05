# Noetic headless producer disposition

Three standalone headless producers were retired after their live authority
collapsed into the current Rust/HCLI path:

- `bandwidth_ascent.py` — historical bandwidth sweep;
- `noetic_dispatch_fusion.py` — historical dispatch-count and fusion sweep;
- `noetic_fused_subbit.py` — historical fused-subbit producer.

Their sealed receipts remain under `receipts/headless/`, including bandwidth,
dispatch-fusion, and fused-subbit findings. `frontier_adversary.py` retains
the small dispatch accounting predicate it still needs, while its live source
and parent-weight checks now use the Rust fused-subbit source plus the surviving
parent observer. No measured result or negative control is promoted by this
retirement; the producers were only parallel execution surfaces.
