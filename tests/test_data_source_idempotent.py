#!/usr/bin/env python3
"""The data_source actuator must never copy a shard it already holds (T3-0 4s.32).

`cf_fz_loop_n32_r1c` copied the same shard FIVE times per node, 160 copies of 422 to 731 s, and
finished no epoch in ninety minutes. One plan was published per agent; the repeat was entirely on
this side. `time.time()` raised because line 21 rebinds `time` with `from time import time`, and it
raised AFTER the copy and BEFORE the state assignment, so the file list stayed on the PFS and
`self._data_source` was never set, and the next `make_loader` resolved again.

THE REPLAY CANNOT CATCH THIS. `replay_cell` drives the planner and synthesises the ack, so it never
runs the resolver; both planners show one stage per node on r1c's record. The defect is only visible
here, which is why these assertions are structural rather than behavioural: `main.py` needs the full
DLIO import graph, so this reads the source and pins the three properties that failed.

    python3 tests/test_data_source_idempotent.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "..", "dlio_benchmark", "main.py")
_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" -- " + detail) if detail else ""))


def main():
    src = open(MAIN).read()

    # 1. THE SHADOWING. `time` is the function here; `time.time()` raises.
    check("no time.time() call survives, because time is the function",
          not re.search(r"^\s*[^#]*\btime\.time\(\)", src, re.M),
          "matches: %s" % re.findall(r".*time\.time\(\).*", src)[:1])
    check("and the module still binds time as a function",
          "from time import time" in src)

    # 2. THE GUARD, IN FRONT OF THE RESOLVER. Idempotence must not depend on state written after it.
    i_guard = src.find("if data_source == self._data_source:")
    i_resolve = src.find("_new = _resolve(")
    check("a data_source already applied short-circuits", i_guard != -1)
    check("and it does so BEFORE the resolver runs, not after",
          i_guard != -1 and i_resolve != -1 and i_guard < i_resolve,
          "guard@%d resolve@%d" % (i_guard, i_resolve))
    check("the short-circuit reports no_move_needed", "reason=no_move_needed" in src)

    # 3. STATE BEFORE LOGGING. The copy is irreversible; nothing after it may lose the fact.
    # Search from the resolver onward: the FIRST ack line in the file belongs to the guard's
    # short-circuit, which legitimately precedes the success path's state assignment.
    i_state = src.find("self._data_source = data_source", i_resolve)
    i_log = src.find("data_source ack: {self._data_source}", i_state)
    check("the applied state is assigned before the ack is logged",
          i_state != -1 and i_log != -1 and i_state < i_log,
          "state@%d log@%d" % (i_state, i_log))
    check("and the ack carries the reason, the applied flag and the completion",
          all(k in src for k in ("reason={_reason}", "applied={_reason ==", "completed_at=")))

    print("\n%d/%d checks passed" % (sum(_results), len(_results)))
    return 0 if all(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
