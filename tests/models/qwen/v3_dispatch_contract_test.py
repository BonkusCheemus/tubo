#!/usr/bin/env python3
"""Source contract for the small-batch v3 dispatcher arms (no build, no GPU).

Extracts the real predicates from two sources and evaluates them against
each other. Neither side is a hardcoded copy of the other:

  oracle  src/models/qwen/hip/kernels/small_batch_dot4.hpp
          SmallBatchDot4KQuantV3Kernel static_asserts (WType set, RowsPerLane
          legality, ActK legality)
  subject src/models/qwen/hip/kernels/prefill_quant_gemm.hip
          TryLaunchKQuantV3 by_geometry/by_batch guards and align_mask

The contract, over nine types x Batch 1..8 x each arm's actual switch
geometries x RowsPerLane {1,2,4} x ActK {0,1}, is set equality within that
geometry footprint: a tuple emits iff the kernel static_assert permits it.
Legality alone would pass a dispatcher that under-covers; equality also
requires every kernel-legal arm to be reachable. Three source mutations
(drop a required x2 type, allow low-batch x4, allow unsupported ActK) must
each break the check, so a silent regression cannot stay green.

Run: python3 tests/models/qwen/v3_dispatch_contract_test.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
KERNEL = ROOT / "src/models/qwen/hip/kernels/small_batch_dot4.hpp"
DISPATCH = ROOT / "src/models/qwen/hip/kernels/prefill_quant_gemm.hip"

TYPES = [
    "kQ4_K", "kQ5_K", "kQ6_K", "kIQ4_XS", "kIQ4_NL",
    "kQ3_K", "kIQ3_S", "kIQ3_XXS", "kIQ2_S",
]
EXCLUDED = "kIQ2_XS"
GEOMETRY_CASES = 5
BATCH_CASES = 7  # cases 1..7 plus an explicit default for 8


def paren_inner(text, i):
    assert text[i] == "(", "expected '(' at %d" % i
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    raise ValueError("unbalanced parentheses")


def guards_after(text, needle):
    idx = text.index(needle)
    open_idx = text.index("(", text.rindex("if constexpr (", 0, idx))
    return paren_inner(text, open_idx)


def to_py(cond):
    cond = " ".join(cond.split())
    cond = re.sub(r"(?:WType|Type)\s*==\s*core::GgmlType::(k\w+)", r" T_\1 ", cond)
    cond = re.sub(r"(\d+)U", r"\1", cond)
    cond = cond.replace("&&", " and ").replace("||", " or ")
    cond = re.sub(r"!(?!=)", " not ", cond)
    return cond


def ev(cond, tname, batch=1, rpl=1, actk=0, rpw=2, waves=4):
    toks = set(re.findall(r"core::GgmlType::(k\w+)", cond))
    ns = {"T_" + t: (t == tname) for t in toks}
    ns.update(Batch=batch, RowsPerLane=rpl, ActK=actk,
              RowsPerWave=rpw, Waves=waves)
    return bool(eval(to_py(cond), {"__builtins__": {}}, ns))


def parse_kernel():
    text = KERNEL.read_text()
    start = text.index("void SmallBatchDot4KQuantV3Kernel(")
    region = text[start:text.index("constexpr bool kQh", start)]
    asserts = [paren_inner(region, m.end() - 1)
               for m in re.finditer(r"static_assert\(", region)]
    assert len(asserts) == 5, "kernel assertion shape changed; review extractor"
    return asserts


def parse_dispatcher():
    text = DISPATCH.read_text()
    body = text[text.index("TryLaunchKQuantV3("):
                text.index("\nvoid LaunchKQuantSmallBatchByType(")]
    x2 = guards_after(body, "cell->rows_per_lane == 2U")
    x4 = guards_after(body, "cell->rows_per_lane == 4U")
    actk = guards_after(body, ", true>(")
    # The fallback has 4x2; x2/x4 do NOT. Bind switch keys to emitted args.
    starts = [body.index("if (cell->rows_per_lane == 2U)"),
              body.index("if (cell->rows_per_lane == 4U)"),
              body.index("if (cell->rows_per_lane != 1U)")]
    ends = starts[1:] + [body.index("const auto by_batch")]
    geoms = {}
    for lane, lo, hi in zip((2, 4, 1), starts, ends):
        launches = re.findall(
            r"case\s+(\d+)U\s*\*\s*16U\s*\+\s*(\d+)U:\s*"
            r"return go\.template operator\(\)<(\d+),\s*(\d+)(?:,\s*(\d+))?>\(\)",
            body[lo:hi])
        assert launches, "missing launch cases"
        geoms[lane] = set()
        for a, b, r, w, l in launches:
            assert (int(a), int(b), lane) == (int(r), int(w), int(l or 1)), "geometry argument mismatch"
            geoms[lane].add((int(r), int(w)))
    bindings = re.findall(
        r"if\s*\(type == core::GgmlType::(k\w+)\)\s*"
        r"return by_batch\.template operator\(\)<core::GgmlType::(k\w+)>\(\)", body)
    assert len(bindings) == len(TYPES) and all(a == b for a, b in bindings), "type argument mismatch"
    dispatch = [b for a, b in bindings]
    batches = re.findall(r"case\s+(\d+):\s*return by_geometry\.template operator\(\)<Type,\s*(\d+)>\(\)", body)
    assert all(a == b for a, b in batches), "batch argument mismatch"
    batch_cases = {int(a) for a, b in batches}
    has_default = bool(re.search(r"default:\s*return by_geometry\.template operator\(\)<Type,\s*8>\(\)", body))
    align = body[body.index("const std::uintptr_t align_mask ="):
                 body.index(";", body.index("const std::uintptr_t align_mask ="))]
    align = align.split("=", 1)[1]
    return dict(x2=x2, x4=x4, actk=actk, geoms=geoms, dispatch=dispatch,
                batch_cases=batch_cases, has_default=has_default, align=align)


def parse_align_mask(expr):
    s = " ".join(expr.split())
    segs = s.split("?")
    conds, vals, tail = [segs[0]], [], None
    for i in range(1, len(segs)):
        if ":" in segs[i]:
            v, rest = segs[i].split(":", 1)
            vals.append(v.strip())
            if i == len(segs) - 1:
                tail = rest.strip()
            else:
                conds.append(rest.strip())
    return conds, vals, tail


def mask_of(conds, vals, tail, tname):
    for cond, val in zip(conds, vals):
        py = re.sub(r"type\s*==\s*core::GgmlType::(k\w+)",
                    lambda m: "True" if m.group(1) == tname else "False", cond)
        py = py.replace("||", " or ").replace("&&", " and ")
        if eval(py, {"__builtins__": {}}):
            return val.rstrip("U")
    return tail.rstrip("U")


def emitted(subj, tname, batch, rpw, waves, rpl, actk):
    if rpl == 2:
        arm = ev(subj["x2"], tname, batch, rpl, actk)
    elif rpl == 4:
        arm = ev(subj["x4"], tname, batch, rpl, actk)
    else:
        arm = rpl == 1
    if not arm:
        return False
    if actk and not ev(subj["actk"], tname, batch, rpl, actk):
        return False
    return True


def allowed(kern, tname, batch, rpw, waves, rpl, actk):
    return all(ev(cond, tname, batch, rpl, actk, rpw, waves) for cond in kern)


def contract(subj, kern):
    illegal, missing, emitted_count = [], [], 0
    for tname in TYPES:
        for batch in range(1, 9):
            for rpl in (1, 2, 4):
                for (rpw, waves) in sorted(subj["geoms"][rpl]):
                    for actk in (0, 1):
                        cell = (tname, batch, rpw, waves, rpl, actk)
                        em = emitted(subj, *cell)
                        al = allowed(kern, *cell)
                        emitted_count += em
                        if em and not al:
                            illegal.append(cell)
                        if al and not em:
                            missing.append(cell)
    return illegal, missing, emitted_count


def main():
    for path in (KERNEL, DISPATCH):
        if not path.exists():
            print("MISSING source: %s" % path)
            return 2

    kern = parse_kernel()
    subj = parse_dispatcher()
    failures = []

    kern_types = set(re.findall(r"core::GgmlType::(k\w+)", kern[0]))
    if kern_types != set(TYPES):
        failures.append("kernel WType set != nine types: %s" % sorted(kern_types))
    if EXCLUDED in kern_types:
        failures.append("kernel WType set contains %s" % EXCLUDED)

    disp_types = set(subj["dispatch"])
    if disp_types != set(TYPES):
        failures.append("by_batch dispatch types != nine: %s" % sorted(disp_types))
    if EXCLUDED in disp_types:
        failures.append("by_batch dispatch contains %s" % EXCLUDED)
    for name, cond in (("x2", subj["x2"]), ("x4", subj["x4"]), ("actk", subj["actk"])):
        if EXCLUDED in cond:
            failures.append("%s guard mentions %s" % (name, EXCLUDED))

    if len(subj["geoms"][1]) != GEOMETRY_CASES or any(
            subj["geoms"][lane] != subj["geoms"][1] - {(4, 2)} for lane in (2, 4)):
        failures.append("geometry footprint changed: %s" % subj["geoms"])
    if subj["batch_cases"] != set(range(1, BATCH_CASES + 1)) or not subj["has_default"]:
        failures.append("by_batch cases=%s default=%s"
                        % (sorted(subj["batch_cases"]), subj["has_default"]))

    conds, vals, tail = parse_align_mask(subj["align"])
    masks = {t: mask_of(conds, vals, tail, t) for t in TYPES}

    illegal, missing, count = contract(subj, kern)
    if illegal:
        failures.append("illegal (emit but kernel rejects) %d, e.g. %s"
                        % (len(illegal), illegal[:4]))
    if missing:
        failures.append("missing (kernel allows, dispatcher drops) %d, e.g. %s"
                        % (len(missing), missing[:4]))

    # Negative mutation controls: each must break the contract.
    def mutated(name, repl_from, repl_to, key):
        text = subj[key]
        if repl_from not in text:
            return None
        m = dict(subj)
        m[key] = text.replace(repl_from, repl_to)
        if m[key] == text:
            return None
        ig, mi, _ = contract(m, kern)
        return name, bool(ig or mi)

    controls = [
        mutated("drop x2 kQ6_K", "Type == core::GgmlType::kQ6_K ||", "", "x2"),
        mutated("low-batch x4", "Batch >= 6", "Batch >= 2", "x4"),
        mutated("unsupported ActK kQ3_K",
                "Type == core::GgmlType::kIQ4_XS",
                "Type == core::GgmlType::kIQ4_XS || Type == core::GgmlType::kQ3_K",
                "actk"),
    ]
    for c in controls:
        if c is None:
            failures.append("mutation control could not be applied (source drift)")
        elif not c[1]:
            failures.append("mutation control not detected: %s" % c[0])

    print("kernel WType: %s" % " ".join(sorted(kern_types)))
    print("dispatch    : %s" % " ".join(sorted(disp_types)))
    print("geometries  : %s" % " ".join(
        "x%d=%s" % (lane, sorted(subj["geoms"][lane])) for lane in (1, 2, 4)))
    print("emitted tuples: %d" % count)
    print("align_mask  : %s" % " ".join("%s=%s" % (t, masks[t]) for t in TYPES))
    print("controls    : %s"
          % " ".join("n/a" if c is None else "%s->%s" % (c[0], "breaks" if c[1] else "GREEN")
                     for c in controls))

    if failures:
        print("\nFAIL")
        for f in failures:
            print("  - " + f)
        return 1
    print("\nPASS: dispatch == all kernel assertions within actual geometry footprint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
