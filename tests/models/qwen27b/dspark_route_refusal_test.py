#!/usr/bin/env python3
"""The Qwen DSpark route: one refusal, placed before construction.

Ruling A replaced step 4's `conv_group_size == 0` test with a structural
completeness test, so the refusal now lives in
src/models/qwen/dflash_weights.cpp and nowhere else. A refusal sitting in front
of a working path is its own defect, so:

  * every construction funnel still runs the predicate BEFORE the `new`;
  * exactly ONE file holds a drafter refusal sentence. A second copy at a CLI
    site drifts from the first, which is what step 4 shipped;
  * the conv launcher carries the host guard for group_size == 0, because the
    device division at dflash_kernels.hip:78-79 is undefined behaviour, not a
    clean error.

Ruling 3 lifted the Qwen-benchmark refusal in bench.cpp so both CLI surfaces
route a DSpark target through the same drafter factory; the DeepSeek-arm
refusal at bench.cpp:391 is a different check and must survive.

Each case fails against a deliberately broken implementation:
  * gate moved below the `new`                       -> OrderTest fails
  * a second copy of the refusal in the CLI           -> SingleCopyTest fails
  * the host guard removed from the launcher          -> GuardTest fails
  * the executor sizing conv_group_size by division    -> OrderTest fails
  * bench refusing Qwen dspark again, or dropping the DeepSeek refusal
                                                     -> SingleCopyTest fails
  * the executor re-deriving `rows` from block_count  -> DraftCapTest fails
"""
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
EXECUTOR = ROOT / "src/models/qwen/hip/dflash/executor.cpp"
MODEL = ROOT / "src/models/qwen/hip/dflash/model.cpp"
BACKEND = ROOT / "src/models/qwen/hip/dflash/backend.cpp"
PROMPT = ROOT / "src/cli/prompt/prompt.cpp"
KERNELS = ROOT / "src/models/qwen/hip/kernels/dflash_kernels.hip"
PREDICATE = ROOT / "src/models/qwen/dflash_weights.cpp"

GATE = "DFlashGpuExecutorSupportRefusal"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def body_of(source: str, signature: str) -> str:
    """Source of the function whose definition line contains `signature`."""
    index = source.index(signature)
    brace = source.index("{", index)
    depth = 0
    for position in range(brace, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return source[index:position + 1]
    raise AssertionError(f"unbalanced braces after {signature!r}")


class OrderTest(unittest.TestCase):
    def test_executor_gate_precedes_construction(self):
        source = read(EXECUTOR)
        create = body_of(source, "QwenDFlashGpuExecutor::Create(")
        gate = create.index(GATE)
        construction = create.index("new QwenDFlashGpuExecutor")
        self.assertLess(
            gate, construction,
            "the refusal must run before `new QwenDFlashGpuExecutor`; a gate "
            "after it has already sized a drafter that cannot draft")

    def test_executor_gate_returns_before_the_new(self):
        source = read(EXECUTOR)
        create = body_of(source, "QwenDFlashGpuExecutor::Create(")
        gate = create.index(GATE)
        construction = create.index("new QwenDFlashGpuExecutor")
        self.assertIn(
            "return nullptr", create[gate:construction],
            "the refusal must return before constructing, not just be called")

    def test_estimate_gate_precedes_the_conv_sizing(self):
        """ResourceClaim() sizes a runner before any executor exists."""
        source = read(EXECUTOR)
        estimate = body_of(source, "QwenDFlashGpuExecutor::EstimateMemoryUsage(")
        self.assertLess(estimate.index(GATE),
                        estimate.index("DFlashDraftScratchElements"),
                        "the estimate must refuse before it sizes the scratch "
                        "it cannot divide for")
        self.assertIn(
            "DFlashDraftScratchElements", estimate,
            "the estimate must size through the shared, division-free helper")

    def test_model_gate_precedes_weight_packing(self):
        source = read(MODEL)
        create = body_of(source, "QwenDFlashGpuModel::Create(")
        self.assertIn(GATE, create,
                      "QwenDFlashGpuModel::Create is the funnel shared by "
                      "bench, prompt and serve")
        self.assertLess(create.index(GATE), create.index("allocations"),
                        "the refusal must precede weight packing")

    def test_backend_constructor_path_constructs(self):
        source = read(BACKEND)
        create = body_of(source, "QwenDFlashGpuDraftBackend::Create(")
        self.assertIn("QwenDFlashGpuExecutor::Create", create,
                      "a DFlash-2 drafter must still reach construction")
        self.assertNotIn(GATE, create,
                         "the backend gate belongs in the model and executor "
                         "factories, not here, or DFlash-2 would stop working")


class SingleCopyTest(unittest.TestCase):
    def test_one_file_holds_the_refusal(self):
        """The predicate is the only place a drafter refusal sentence lives."""
        owners = []
        for path in (PREDICATE, PROMPT, EXECUTOR, MODEL, BACKEND, KERNELS):
            text = read(path)
            if "drafting is refused" in text:
                owners.append(path.name)
        self.assertEqual(
            owners, ["dflash_weights.cpp"],
            "a drafter refusal sentence must exist in exactly one file; found "
            f"{owners}")

    def test_cli_does_not_refuse_dspark(self):
        prompt = read(PROMPT)
        self.assertNotIn("Qwen DSpark drafting is refused", prompt,
                         "the CLI must propagate the predicate message, not "
                         "restate it")
        self.assertIn("QwenDFlashGpuDraftBackend::CreateFromGguf", prompt,
                      "the Qwen DSpark route must reach the shared backend")

    def test_bench_no_longer_refuses_dspark_on_a_qwen_target(self):
        """Ruling 3 lifted the Qwen-benchmark refusal. What must survive is the
        DeepSeek-only refusal at bench.cpp:391, which is a different check on a
        different surface and was not in scope."""
        bench = read(ROOT / "src/cli/bench/bench.cpp")
        self.assertNotIn(
            "--speculative dspark benchmarks run DeepSeek only", bench,
            "the Qwen benchmark arm must no longer refuse --speculative dspark")
        self.assertIn(
            "DeepSeek supports only --speculative dspark or off", bench,
            "the DeepSeek-arm refusal must survive the lift")


class DraftCapTest(unittest.TestCase):
    """Ruling 2 wiring: the executor takes the cap from the tested helper."""

    def test_executor_takes_its_chain_geometry_from_the_capped_helper(self):
        source = read(EXECUTOR)
        forward = body_of(source, "QwenDFlashGpuExecutor::ForwardBlock(")
        self.assertIn("DSparkChainRangeFor", forward,
                      "the DSpark chain geometry must come from the helper")
        self.assertNotRegex(
            forward, r"anchor_predicted\s*\?\s*block_count",
            "rows must not be the raw block width: the caller asked for "
            "draft_count")

    def test_helper_is_cpu_visible_and_cap_documented(self):
        header = read(ROOT / "src/models/qwen/dflash_weights.hpp")
        self.assertIn("DSparkChainRangeFor", header,
                      "the cap must be callable from a CPU test")
        self.assertIn("draft_count", header,
                      "the helper's contract is the caller's draft_count")


class GuardTest(unittest.TestCase):
    def test_conv_launcher_refuses_zero_group_size_on_the_host(self):
        launcher = body_of(
            read(KERNELS), "void LaunchDFlashGroupedDynamicConv(")
        self.assertIn("DFlashGroupedConvLaunchRefusal", launcher,
                      "the launcher must refuse group_size == 0 before it "
                      "launches: the device divides by it at :78-79")
        refusal = launcher.index("DFlashGroupedConvLaunchRefusal")
        launch = launcher.index("dflash_grouped_dynamic_conv_kernel<<<")
        self.assertLess(refusal, launch,
                        "the host refusal must precede the kernel launch")
        self.assertRegex(launcher, r"throw",
                         "a refused launch must throw, not return silently")

    def test_predicate_names_the_zero_divisor(self):
        body = body_of(read(PREDICATE), "std::string DFlashGroupedConvLaunchRefusal(")
        self.assertIn("conv_group_size", body,
                      "the refusal must name the zero divisor")
        self.assertIn("undefined", body,
                      "the refusal must say why a zero divisor cannot be "
                      "caught on the device")


class BenchSurfaceTest(unittest.TestCase):
    """Ruling 3: bench and prompt are the same surface with different speed
    vehicles. Each check fails if either drifts back."""

    def test_bench_accepts_dspark_on_a_qwen_target(self):
        """The gate that builds the drafter must name BOTH backends. Testing
        the bare literal would pass on a build that names dspark only inside
        an unreachable dflash2 branch."""
        bench = read(ROOT / "src/cli/bench/bench.cpp")
        create = body_of(bench, "int RunBench(")
        gates = re.findall(
            r'if \(([^)]*speculative_backend[^)]*)\) \{', create)
        gated = [g for g in gates if '"dflash2"' in g]
        self.assertTrue(gated, "bench must gate the drafter on the backend flag")
        self.assertTrue(
            any('"dspark"' in g for g in gated),
            "the drafter construction gate must admit dspark, not just "
            f"dflash2; gates found: {gated}")
        self.assertIn("QwenDFlashGpuDraftBackend::CreateFromGguf", create,
                      "bench must build the DSpark drafter it accepts")

    def test_both_surfaces_name_the_model_the_same_way(self):
        """The one difference between the surfaces is the flag that carries the
        path, so the empty-path refusal is worded identically in both."""
        bench = read(ROOT / "src/cli/bench/bench.cpp")
        prompt = read(PROMPT)
        for surface, text in (("bench", bench), ("prompt", prompt)):
            self.assertIn("DSpark requires --dspark-model", text,
                          f"{surface} must name the missing-model requirement")
            self.assertIn("opt.dspark_model_path", text,
                          f"{surface} must read the path from --dspark-model")

    def test_neither_surface_calls_the_drafter_refusal_itself(self):
        """A prose mention in a comment is fine; a call means the CLI decided
        to re-run the drafter's own predicate and could diverge from it."""
        for surface, text in (("bench", read(ROOT / "src/cli/bench/bench.cpp")),
                              ("prompt", read(PROMPT))):
            self.assertNotRegex(
                text, r"DFlashGpuExecutorSupportRefusal\s*\(",
                f"{surface} propagates the drafter's own refusal message "
                "rather than re-running its predicate")


class MessageTest(unittest.TestCase):
    def test_predicate_refusal_is_never_empty(self):
        header = read(ROOT / "src/models/qwen/dflash_weights.hpp")
        self.assertIn(GATE, header,
                      "the predicate must be declared in the CPU-visible "
                      "header")
        body = body_of(read(PREDICATE), f"std::string {GATE}(")
        returns = [r for r in re.findall(r'return\s+([^;]+);', body)
                   if r.strip() != "{}"]
        self.assertTrue(returns, "expected explicit returns")
        for expression in returns:
            self.assertNotEqual(expression.strip(), '""',
                                "a refusal must never be an empty string")
            self.assertTrue(
                expression.strip().startswith(('"', "std::to_string", "return",
                                               "config.", "draft.")),
                f"unexpected return shape: {expression!r}")
        # The sentence is assembled from adjacent string literals, so compare
        # with the line breaks and indentation collapsed away.
        joined = re.sub(r'"\s*"', "", body)
        self.assertIn("neither a complete DFlash-2 nor a complete DSpark",
                      joined,
                      "the mixed-artifact refusal must name the invariant")


if __name__ == "__main__":
    unittest.main(verbosity=2)
    sys.exit(0)