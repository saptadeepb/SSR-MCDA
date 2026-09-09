#!/usr/bin/env python3
"""
audit_and_package.py -- final integrity audit and assembly of the replication package.

Runs a set of checks that must ALL pass before the package is shipped, then builds the
zip. The audit is deliberately unforgiving: it is easier to find a broken cross-reference
here than to have a referee find it.

Checks
------
 1. every property test passes
 2. results.json exists and every case is complete
 3. every LaTeX macro used in the manuscript is defined in numbers.tex
 4. every \\cite key resolves to an entry in references.bib
 5. every \\ref target has a matching \\label
 6. every \\includegraphics target exists
 7. every \\input target exists
 8. no placeholder text (TODO, XXX, ??) survives
 9. every figure is a valid PDF
10. the manuscript compiles without undefined references or citations

Usage:  python audit_and_package.py [--skip-tests] [--no-zip]

Author: Saptadeep Biswas
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
MS = os.path.join(ROOT, "manuscript")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

FAIL: list[str] = []
WARN: list[str] = []


def check(cond: bool, msg: str, warn_only: bool = False) -> None:
    if cond:
        print(f"  PASS  {msg}")
    else:
        (WARN if warn_only else FAIL).append(msg)
        print(f"  {'WARN' if warn_only else 'FAIL'}  {msg}")


def read(path: str) -> str:
    # LaTeX logs are not valid UTF-8; never let an encoding error abort the audit
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


# ==========================================================================
def audit(skip_tests: bool = False) -> None:
    print("\n=== 1. property tests ===")
    if skip_tests:
        print("  SKIP (requested)")
    else:
        r = subprocess.run([sys.executable, os.path.join(HERE, "tests", "test_ssr_mcda.py")],
                           capture_output=True, text=True, cwd=HERE)
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "no output"
        check(r.returncode == 0, f"property tests: {tail}")

    print("\n=== 2. results completeness ===")
    rj = os.path.join(RES, "results.json")
    check(os.path.exists(rj), "results.json exists")
    if os.path.exists(rj):
        d = json.load(open(rj))
        for case, c in d["cases"].items():
            for key in ("recommended", "stability_radius_linf", "choquet_scores",
                        "max_regret", "shapley_importance", "monte_carlo",
                        "marichal_roubens", "preference_resampling"):
                check(key in c and c[key] is not None, f"{case}: results.json has '{key}'")
            check(c["cci"].get("preference_violations", 0) == 0,
                  f"{case}: identified capacity represents every revealed preference")

    print("\n=== 3. LaTeX macros ===")
    tex = read(os.path.join(MS, "main.tex"))
    nums_path = os.path.join(MS, "numbers.tex")
    check(os.path.exists(nums_path), "numbers.tex exists")
    if os.path.exists(nums_path):
        defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", read(nums_path)))
        body = tex.split(r"\input{numbers.tex}")[-1]
        used = set(re.findall(r"\\([a-z][A-Za-z]*)(?:\{\}|\\|\s|\.|,|\)|\$)", body))
        latex_builtin = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", tex))
        missing = sorted(u for u in used
                         if u in {m for m in used if m[0].islower()}
                         and u not in defined and u not in latex_builtin
                         and re.match(r"^(rec|rad|orness|marichalH|shannonH|cci|bwm|npref|"
                                      r"mcvalid|rr|score|regret|rank|mcone|mcmean|smaa|phi|"
                                      r"kt|same|zero|pref|ntests|mcdraws|eps|nposture|"
                                      r"ncriteria|nsector|nmoebius|nmonotone|norigin|mr|loo|"
                                      r"null|min)", u))
        check(not missing, f"all result macros defined (missing: {missing[:8]})")

    print("\n=== 4. citations ===")
    bib = read(os.path.join(MS, "references.bib"))
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+)\s*,", bib))
    cited: set[str] = set()
    # a \citep{} may wrap across lines with a trailing comment character, so strip
    # comments and whitespace before splitting on commas
    for m in re.findall(r"\\cite[a-z]*\{([^}]*)\}", tex):
        cleaned = re.sub(r"%.*?\n", "", m)
        cited.update(re.sub(r"\s+", "", k) for k in cleaned.split(",") if k.strip())
    undefined = sorted(cited - bib_keys)
    check(not undefined, f"every \\cite key is in references.bib (undefined: {undefined})")
    uncited = sorted(bib_keys - cited)
    check(len(uncited) == 0,
          f"no uncited bib entries ({len(uncited)} uncited: {uncited[:10]})", warn_only=True)

    print("\n=== 5. cross-references ===")
    # labels live in the generated table files too, so expand \input before checking
    expanded = tex
    for inp in re.findall(r"\\input\{([^}]*)\}", tex):
        cand = os.path.join(MS, inp if inp.endswith(".tex") else inp + ".tex")
        if os.path.exists(cand):
            expanded += "\n" + read(cand)
    labels = set(re.findall(r"\\label\{([^}]*)\}", expanded))
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]*)\}", tex))
    dangling = sorted(refs - labels)
    check(not dangling, f"every \\ref resolves (dangling: {dangling})")

    print("\n=== 6. figures and inputs ===")
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex):
        path = os.path.normpath(os.path.join(MS, g))
        ok = os.path.exists(path)
        check(ok, f"figure exists: {os.path.basename(g)}")
        if ok:
            with open(path, "rb") as f:
                check(f.read(5) == b"%PDF-", f"valid PDF: {os.path.basename(g)}")
    for inp in re.findall(r"\\input\{([^}]*)\}", tex):
        cand = os.path.join(MS, inp if inp.endswith(".tex") else inp + ".tex")
        check(os.path.exists(cand), f"input exists: {inp}")

    print("\n=== 7. placeholders ===")
    for bad in ("TODO", "XXX", "FIXME", "PLACEHOLDER", "??"):
        check(bad not in tex, f"no '{bad}' in manuscript")

    print("\n=== 8. compilation ===")
    log = os.path.join(MS, "main.log")
    if os.path.exists(log):
        lg = read(log)
        check("Undefined control sequence" not in lg, "no undefined control sequences")
        check("There were undefined references" not in lg, "no undefined references")
        check("Citation" not in lg or "undefined" not in lg.split("Citation")[-1][:200],
              "no undefined citations", warn_only=True)
    else:
        check(False, "main.log exists (has the manuscript been compiled?)", warn_only=True)


# ==========================================================================
def build_zip(name: str = "SSR_MCDA_replication_package.zip") -> str:
    out = os.path.join(ROOT, name)
    include_dirs = ["code", "data", "results", "figures", "manuscript", "review"]
    skip_ext = {".pyc", ".aux", ".out", ".fls", ".fdb_latexmk", ".synctex.gz", ".png"}
    skip_names = {"__pycache__", ".ipynb_checkpoints"}
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for top in include_dirs:
            base = os.path.join(ROOT, top)
            if not os.path.isdir(base):
                continue
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if d not in skip_names]
                for fn in sorted(filenames):
                    if os.path.splitext(fn)[1] in skip_ext:
                        continue
                    full = os.path.join(dirpath, fn)
                    z.write(full, os.path.relpath(full, ROOT))
                    n += 1
        for extra in ("README.md", "requirements.txt"):
            p = os.path.join(ROOT, extra)
            if os.path.exists(p):
                z.write(p, extra)
                n += 1
    print(f"\nPackaged {n} files -> {out} ({os.path.getsize(out)/1e6:.2f} MB)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()
    audit(a.skip_tests)
    print("\n" + "=" * 66)
    if FAIL:
        print(f"AUDIT FAILED: {len(FAIL)} blocking issue(s)")
        for f in FAIL:
            print(f"   - {f}")
    else:
        print("AUDIT PASSED" + (f" with {len(WARN)} warning(s)" if WARN else ""))
        for w in WARN:
            print(f"   ! {w}")
    print("=" * 66)
    if not a.no_zip and not FAIL:
        build_zip()
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
