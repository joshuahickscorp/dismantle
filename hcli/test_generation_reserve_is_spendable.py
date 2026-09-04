"""The window must not be reserved for generation the engine cannot request.

`context_budget` withholds DEFAULT_GENERATION_RESERVE (4096) from every window,
but `engine._MAX_TOKENS_CEILING` caps every completion request at 2048. On the
resident's 8192-token window that stranded 2048 tokens -- 25% of the whole
window -- reserved for tokens nothing will ever ask for. Combined with the 512
framing reserve it left usable_input 3584 against a measured packet floor of
6501, so no mission of any length could be admitted.

Reserving headroom the requester is structurally incapable of using is not
safety, it is waste. The reserve is now clamped to what the engine can actually
spend.
"""

from __future__ import annotations

import unittest

from hcli.context_budget import DEFAULT_GENERATION_RESERVE, resolve
from hcli.engine import _MAX_TOKENS_CEILING


class TestGenerationReserveIsSpendable(unittest.TestCase):
    def test_reserve_never_exceeds_what_the_engine_can_request(self) -> None:
        budget = resolve(model_path=None, n_parallel=1)
        self.assertLessEqual(
            budget.generation_reserve,
            _MAX_TOKENS_CEILING,
            f"reserved {budget.generation_reserve} for generation but the engine "
            f"can never request more than {_MAX_TOKENS_CEILING}",
        )

    def test_the_stranded_tokens_are_returned_to_input(self) -> None:
        """The whole point: usable input must actually grow."""
        budget = resolve(model_path=None, n_parallel=1)
        stranded = DEFAULT_GENERATION_RESERVE - _MAX_TOKENS_CEILING
        self.assertGreater(stranded, 0, "fixture assumes the default exceeds the ceiling")
        self.assertEqual(
            budget.usable_input_tokens,
            budget.per_request_ctx - _MAX_TOKENS_CEILING - budget.framing_reserve,
            "the reclaimed tokens did not reach usable input",
        )

    def test_an_explicit_smaller_reserve_is_still_honoured(self) -> None:
        """Negative control: clamping must not raise a caller's smaller request."""
        budget = resolve(model_path=None, n_parallel=1, generation_reserve=256)
        self.assertEqual(
            budget.generation_reserve, 256, "an explicit smaller reserve was overridden"
        )

    def test_an_explicit_larger_reserve_is_still_clamped(self) -> None:
        """A caller asking for more than the engine can spend is still waste."""
        budget = resolve(model_path=None, n_parallel=1, generation_reserve=8192)
        self.assertLessEqual(budget.generation_reserve, _MAX_TOKENS_CEILING)


if __name__ == "__main__":
    unittest.main()
