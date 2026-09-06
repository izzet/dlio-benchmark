#!/usr/bin/env python3
"""The actuator must report when the relocation TAKES EFFECT, not only when the copy ends.

T3-0 4s.37. `data_source` resolves into a file list; DLIO reads that list only when the loader is
REBUILT, which happens at an epoch boundary. On `cf_fz_loop_n32_r1f` the copy returned at 08:13:34
and the loader was rebuilt at 08:21:54, so for eight minutes the run read the PFS through a knob
that said `local`. The planner credited those windows to the moved arm, the arms tied, and all 32
nodes reverted -- one window before the rebuild that would have shown 5.5 s windows against 25.

So the actuator sends a SECOND ack, from the rebuild, carrying `effective_at`. Three properties
have to hold and none of them is visible to the replay, which never runs this code:

  1. the effective mark is taken where the loader is BUILT, not in the cached or reconfigured
     branches, which return a loader built on the previous file list,
  2. it fires once per change of source, so a rebuild that changes nothing sends nothing,
  3. the plan is held from the first ack so the second one can be attributed to it.

Structural, like `test_data_source_idempotent.py`: `main.py` needs the full DLIO import graph.

    python3 tests/test_effective_ack.py
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
    print("\n1. the effect is reported from the loader build")

    check("_note_data_source_effective exists", "def _note_data_source_effective" in src)
    # The build branch is the one that calls get_loader; the mark must be inside it.
    build = re.search(r"loader = self\.framework\.get_loader\(.*?\n(.*?)\n\s*self\._cached_loader = loader",
                      src, re.S)
    check("the mark is taken in the branch that builds the loader",
          bool(build) and "_note_data_source_effective" in build.group(1),
          (build.group(1).strip()[:70] if build else "no build branch found"))
    # ... and NOT in the two branches that reuse one built on the old list.
    cached = re.search(r"\(cached\)\"\s*\n\s*\)\n\s*loader = self\._cached_loader", src)
    check("the cached branch reports nothing", bool(cached))
    recfg = src.split("(reconfigure)")[0].rsplit("elif", 1)[-1] if "(reconfigure)" in src else ""
    check("the reconfigured branch reports nothing",
          "_note_data_source_effective" not in recfg)

    print("\n2. it fires once per change of source")
    fn = src.split("def _note_data_source_effective", 1)[-1].split("\n    def ", 1)[0]
    check("it compares the source the loader was last built for",
          "_data_source_loader_built_for" in fn, fn.strip()[:60])
    check("and returns without acking when that is unchanged",
          re.search(r"_data_source_loader_built_for\", \"pfs\"\) == _ds:\s*\n\s*return", fn)
          is not None)
    check("the ack it sends carries effective_at", "effective_at=_eff" in fn)
    check("and it carries the completion too, so the record keeps both instants",
          "completed_at=getattr(self, \"_data_source_completed_at\", None)" in fn)
    check("a failure to ack cannot fail the training run",
          "except Exception" in fn and "raise" not in fn)

    print("\n3. the plan is held from the first ack")
    hook = src.split("def _liveflow_ack_pending", 1)[-1].split("\n    def ", 1)[0] \
        if "_liveflow_ack_pending" in src else ""
    check("_liveflow_ack_pending exists", bool(hook))
    check("it holds only data_source", 'knob_id.split(".")[-1] != "data_source"' in hook)
    check("and only when the copy actually happened",
          '_data_source_reason", None) != "relocated"' in hook)
    check("the held plan is consumed exactly once",
          "_data_source_pending_ack = None" in fn, "cleared before the ack is sent")

    print("\n%d/%d checks passed" % (sum(_results), len(_results)))
    return 0 if all(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
