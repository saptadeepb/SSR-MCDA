#!/usr/bin/env python3
"""
run_case_study.py -- reproduces every numerical result reported in the paper.

Usage
-----
    python run_case_study.py [--draws 2000] [--out ../results]

Outputs (written to --out)
--------------------------
    performance_raw_{CASE}.csv          Layer-I equilibrium performance matrix
    performance_norm_{CASE}.csv         normalised, benefit-oriented matrix
    capacity_{CASE}.csv                 Moebius coefficients, Shapley, interaction
    interaction_{CASE}.csv              Shapley interaction index matrix
    mcr_{CASE}.csv                      max regret, rank, adversary, stability radius
    benchmarks_{CASE}.csv               comparator method scores and ranks
    rank_agreement_{CASE}.csv           Kendall tau / Spearman rho between methods
    montecarlo_{CASE}.csv               rank-frequency distribution over parameter draws
    sensitivity_oat_{CASE}.csv          one-at-a-time parameter sensitivity
    rank_reversal_{CASE}.csv            rank-reversal probe results
    sourcing_{CASE}.csv                 equilibrium sourcing shares by sector and posture
    regret_attribution_{CASE}.csv       criterion-level decomposition of worst-case regret
    provenance_{CASE}.csv               source of every model input
    results.json                        machine-readable digest for the manuscript
    ssr_mcda_results.xlsx               all tables in one workbook

Author: Saptadeep Biswas
License: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ssr_mcda as S  # noqa: E402
from ssr_mcda import benchmarks as B  # noqa: E402
from ssr_mcda.data import (  # noqa: E402
    CASES,
    REVEALED_PREFERENCES,
    SECTOR_DEF,
    build_case,
    labour_intensities,
    preference_indices,
)
from ssr_mcda.regret import (  # noqa: E402
    UncertaintySet,
    mcr_ranking,
    preference_rows,
    regret_attribution,
    stability_radius,
)

CRIT = S.CRITERION_CODES
CRIT_NAME = [f"{c} {n}" for c, n, _ in S.CRITERIA]
POST = S.POSTURE_CODES
POST_NAME = [f"{c}: {d}" for c, d in S.POSTURES]
EPS_DEFAULT = 0.05

# Declared a-priori Best-Worst reference profile (see table_benchmarks). Best = C3
# domestic value added, worst = C9 transport carbon. These are a STATED hypothetical
# used to give the elicited-weight comparator a concrete, documented input; they are
# not derived from any output of the proposed method, and they are reported in the
# paper so a reader can substitute their own.
#              C1  C2  C3  C4  C5  C6  C7  C8  C9
BWM_BEST_TO_OTHERS  = [3,  4,  1,  3,  2,  5,  4,  2,  7]   # C3 relative to each
BWM_OTHERS_TO_WORST = [3,  2,  7,  3,  4,  2,  2,  5,  1]   # each relative to C9


# ==========================================================================
def solve_case(case_id: str, par: S.GameParams, epsilon: float = EPS_DEFAULT) -> dict:
    """Run the three layers once for a given parameterisation."""
    case = build_case(case_id, par)
    P, costs, detail = S.build_performance_matrix(
        case.sectors, case.env, par, case.rival, labour_intensities()
    )
    Z = S.normalise(P)
    prefs = preference_indices(case_id, POST)
    cap, info = S.cci(Z, prefs)
    G = preference_rows(Z, prefs)
    U = UncertaintySet(n=len(CRIT), centre=cap.vector, epsilon=epsilon, preference_rows=G)
    mcr = mcr_ranking(Z, U)
    return {"case": case, "P": P, "Z": Z, "cap": cap, "info": info, "prefs": prefs,
            "U": U, "mcr": mcr, "detail": detail, "costs": costs, "par": par}


# ==========================================================================
def table_performance(r: dict, case_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(r["P"], index=POST, columns=CRIT_NAME)
    nrm = pd.DataFrame(r["Z"], index=POST, columns=CRIT_NAME)
    raw.index.name = nrm.index.name = "posture"
    return raw.round(6), nrm.round(6)


def table_capacity(r: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cap = r["cap"]
    phi = cap.shapley()
    phi_game = r["info"]["shapley_of_game"]
    rows = []
    for i, c in enumerate(CRIT):
        rows.append({
            "criterion": c, "name": S.CRITERIA[i][1], "sense": S.CRITERIA[i][2],
            "moebius_singleton": cap.singles[i],
            "shapley_of_game": phi_game[i],
            "shapley_importance": phi[i],
        })
    caps = pd.DataFrame(rows).set_index("criterion")
    I = cap.interaction_matrix()
    inter = pd.DataFrame(I, index=CRIT, columns=CRIT)
    return caps.round(6), inter.round(6)


def table_mcr(r: dict) -> pd.DataFrame:
    mcr, cap, Z = r["mcr"], r["cap"], r["Z"]
    G = r["U"].preference_rows
    scores = np.asarray(cap.choquet(Z), float).ravel()
    rad, chal, _ = stability_radius(Z, cap.vector, mcr["recommended"], norm="l1",
                                    preference_rows=G)
    rad_inf, chal_inf, _ = stability_radius(Z, cap.vector, mcr["recommended"], norm="linf",
                                            preference_rows=G)
    rows = []
    for i, c in enumerate(POST):
        rows.append({
            "posture": c, "description": S.POSTURES[i][1],
            "choquet_score": scores[i],
            "score_rank": int(np.argsort(np.argsort(-scores))[i] + 1),
            "max_regret": mcr["regret"][i],
            "mcr_rank": int(mcr["rank"][i]),
            "binding_adversary": POST[mcr["adversary"][i]] if mcr["adversary"][i] >= 0 else "",
        })
    df = pd.DataFrame(rows).set_index("posture")
    df.attrs["stability_radius_l1"] = rad
    df.attrs["stability_radius_linf"] = rad_inf
    df.attrs["stability_challenger_l1"] = POST[chal] if chal >= 0 else ""
    df.attrs["stability_challenger_linf"] = POST[chal_inf] if chal_inf >= 0 else ""
    return df.round(6)


def table_benchmarks(r: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    Z, cap = r["Z"], r["cap"]
    phi = cap.shapley()
    w_ent, w_cri = B.entropy_weights(Z), B.critic_weights(Z)
    # BWM reference judgements are DECLARED A PRIORI and are deliberately independent
    # of anything the proposed method produces. No expert elicitation was conducted for
    # this study, so the profile below is a stated hypothetical: a policy analyst who
    # treats domestic value added (C3) as most important and transport carbon (C9) as
    # least, on the standard 1-9 ratio ladder. Deriving these judgements from the
    # identified Shapley vector would make the comparator a mirror of the method and
    # the resulting rank agreement uninformative.
    best, worst = 2, 8                       # C3 best, C9 worst
    a_bo = np.array(BWM_BEST_TO_OTHERS, dtype=float)
    a_ow = np.array(BWM_OTHERS_TO_WORST, dtype=float)
    w_bwm, xi = B.bwm_weights(best, worst, a_bo, a_ow)

    # The decisive comparator: a 2-additive capacity identified from the SAME Pi by the
    # standard maximum-separation procedure. This isolates what the discrimination game
    # contributes over textbook capacity identification.
    mr_cap, mr_info = B.marichal_roubens_capacity(Z, r["prefs"], min_shapley=0.02)

    methods = {
        "SSR-MCDA (Choquet score)": (np.asarray(cap.choquet(Z), float).ravel(), True),
        "SSR-MCDA (minimax regret)": (r["mcr"]["regret"], False),
        "Choquet (Marichal-Roubens capacity)": (np.asarray(mr_cap.choquet(Z), float).ravel(), True),
        "WSM (Shapley weights)": (B.weighted_sum(Z, phi), True),
        "WSM (entropy weights)": (B.weighted_sum(Z, w_ent), True),
        "WSM (CRITIC weights)": (B.weighted_sum(Z, w_cri), True),
        "WSM (BWM weights)": (B.weighted_sum(Z, w_bwm), True),
        "TOPSIS (Shapley weights)": (B.topsis(Z, phi), True),
        "VIKOR (Shapley weights, Q)": (B.vikor(Z, phi)["Q"], False),
        "PROMETHEE II (Shapley weights)": (B.promethee2(Z, phi), True),
        "COPRAS (Shapley weights)": (B.copras(Z, phi), True),
    }
    sm = B.smaa2(Z, n_draws=20000)
    methods["SMAA-2 (rank-1 acceptability)"] = (sm["acceptability"][:, 0], True)

    rows, ranks = [], {}
    for name, (sc, hib) in methods.items():
        rk = B.scores_to_ranks(sc, higher_is_better=hib)
        ranks[name] = rk
        for i, p in enumerate(POST):
            rows.append({"method": name, "posture": p, "score": float(sc[i]), "rank": int(rk[i])})
    bench = pd.DataFrame(rows)

    names = list(methods)
    agree = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            agree.loc[a, b] = B.rank_agreement(ranks[a], ranks[b])["kendall_tau"]
    bench.attrs["bwm_xi"] = xi
    bench.attrs["mr_separation"] = mr_info["separation"]
    bench.attrs["mr_shapley"] = mr_cap.shapley().tolist()
    bench.attrs["mr_phi_l1_distance"] = float(np.abs(cap.shapley() - mr_cap.shapley()).sum())
    bench.attrs["weights"] = {"shapley": phi.tolist(), "entropy": w_ent.tolist(),
                              "critic": w_cri.tolist(), "bwm": w_bwm.tolist()}
    bench.attrs["smaa_acceptability"] = sm["acceptability"].tolist()
    return bench, agree.round(4)


def table_sourcing(r: dict, case_id: str) -> pd.DataFrame:
    rows = []
    origins = r["case"].env.origins
    for code, det in r["detail"].items():
        for sname, d in det.items():
            for k, o in enumerate(origins):
                rows.append({"posture": code, "sector": sname, "origin": o,
                             "share": float(d["x"][k]), "tariff": float(d["tau"][k]),
                             "retaliation": float(d["r"][k])})
    return pd.DataFrame(rows)


def table_regret_attribution(r: dict) -> pd.DataFrame:
    rows = []
    mcr, Z, U = r["mcr"], r["Z"], r["U"]
    for i, p in enumerate(POST):
        b = mcr["adversary"][i]
        if b < 0:
            continue
        att = regret_attribution(Z, i, b, U, criteria=CRIT)
        row = {"posture": p, "adversary": POST[b], "max_regret": float(mcr["regret"][i])}
        row.update({k: float(v) for k, v in att.items()})
        rows.append(row)
    return pd.DataFrame(rows).set_index("posture").round(6)


# ==========================================================================
def monte_carlo(case_id: str, base: S.GameParams, draws: int, seed: int = 20260901) -> pd.DataFrame:
    """Monte-Carlo over the Layer-III uncertainty set of *game* parameters.

    Each parameter is drawn from a uniform interval centred on its default and wide
    enough to span the plausible range implied by the literature; delta_dom, the one
    calibration parameter, is drawn over its full declared range [0.15, 0.65].
    """
    rng = np.random.default_rng(seed)
    spec = {
        "rho": (0.30, 1.20), "lam": (0.10, 0.70), "beta": (0.60, 2.00),
        "gamma": (0.00, 0.06), "theta0": (0.50, 1.50), "delta_dom": (0.15, 0.65),
        "subsidy": (0.05, 0.30), "uniform_hike": (0.05, 0.20),
        "targeted_hike": (0.10, 0.45), "align_threshold": (0.60, 0.95),
        "near_threshold_km": (3000.0, 8000.0),
    }
    recs, failures = [], []
    for d in range(draws):
        kw = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in spec.items()}
        par = replace(base, **kw)
        try:
            case = build_case(case_id, par)
            P, _, _ = S.build_performance_matrix(
                case.sectors, case.env, par, case.rival, labour_intensities()
            )
            Z = S.normalise(P)
            cap, info = S.cci(Z, preference_indices(case_id, POST))
            sc = np.asarray(cap.choquet(Z), float).ravel()
        except (ValueError, RuntimeError) as e:
            failures.append({"draw": d, "error": f"{type(e).__name__}: {e}"})
            continue
        rk = B.scores_to_ranks(sc, higher_is_better=True)
        rec = {"draw": d,
               # representability is REPORTED, not discarded: a draw whose identified
               # capacity fails to reproduce a statement it was built from is still a
               # valid ranking of that capacity, but the reader must be able to see how
               # often it happens.
               "preference_violations": int(info.get("preference_violations", 0)),
               **{f"rank_{p}": int(rk[i]) for i, p in enumerate(POST)},
               **{f"score_{p}": float(sc[i]) for i, p in enumerate(POST)}, **kw}
        recs.append(rec)
    out = pd.DataFrame(recs)
    out.attrs["failures"] = failures
    return out


def sensitivity_oat(case_id: str, base: S.GameParams, n_grid: int = 11) -> pd.DataFrame:
    """One-at-a-time sensitivity: sweep each parameter across its range holding the
    rest at their defaults, and record the resulting ranking."""
    spec = {
        "rho": (0.30, 1.20), "lam": (0.10, 0.70), "beta": (0.60, 2.00),
        "gamma": (0.00, 0.06), "theta0": (0.50, 1.50), "delta_dom": (0.15, 0.65),
        "subsidy": (0.05, 0.30), "uniform_hike": (0.05, 0.20),
        "targeted_hike": (0.10, 0.45), "align_threshold": (0.60, 0.95),
        "near_threshold_km": (3000.0, 8000.0),
    }
    rows = []
    for name, (lo, hi) in spec.items():
        for val in np.linspace(lo, hi, n_grid):
            par = replace(base, **{name: float(val)})
            try:
                r = solve_case(case_id, par)
            except Exception:
                continue
            sc = np.asarray(r["cap"].choquet(r["Z"]), float).ravel()
            rk = B.scores_to_ranks(sc)
            rows.append({"parameter": name, "value": float(val),
                         "top_choquet": POST[int(np.argmax(sc))],
                         "top_mcr": POST[r["mcr"]["recommended"]],
                         **{f"rank_{p}": int(rk[i]) for i, p in enumerate(POST)}})
    return pd.DataFrame(rows)


def preference_resampling(case_id: str, par: S.GameParams, n_boot: int = 300,
                          seed: int = 20260901) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify how much of the identified capacity is actually pinned down by Pi.

    Two experiments, both essential when 45 Moebius coefficients are identified from a
    handful of statements:

    * leave-one-out: drop each statement in turn and re-identify, so the influence of
      any single statement on the importance vector and on the ranking is visible;
    * bootstrap: resample Pi with replacement and report the spread of the importance
      and interaction indices, together with a permutation null in which the pairs are
      randomised, so the reader can judge whether the reported indices are
      distinguishable from what a preference set of the same size would give by chance.
    """
    rng = np.random.default_rng(seed)
    r0 = solve_case(case_id, par)
    Z, prefs = r0["Z"], r0["prefs"]
    phi0 = r0["cap"].shapley()
    I0 = r0["cap"].interaction_matrix()
    base_rank = B.scores_to_ranks(np.asarray(r0["cap"].choquet(Z), float).ravel())

    loo = []
    for k in range(len(prefs)):
        sub = [q for j, q in enumerate(prefs) if j != k]
        try:
            cap, _ = S.cci(Z, sub)
        except (ValueError, RuntimeError):
            continue
        rk = B.scores_to_ranks(np.asarray(cap.choquet(Z), float).ravel())
        loo.append({"dropped_statement": k,
                    "dropped_pair": f"{POST[prefs[k][0]]}>{POST[prefs[k][1]]}",
                    "kendall_tau_vs_base": B.rank_agreement(base_rank, rk)["kendall_tau"],
                    "top_alternative": POST[int(np.argmin(rk))],
                    "phi_l1_shift": float(np.abs(cap.shapley() - phi0).sum()),
                    **{f"phi_{c}": float(cap.shapley()[i]) for i, c in enumerate(CRIT)}})

    boot, null = [], []
    m_alt = Z.shape[0]
    for b in range(n_boot):
        idx = rng.integers(0, len(prefs), size=len(prefs))
        sub = [prefs[i] for i in idx]
        try:
            cap, _ = S.cci(Z, sub)
            boot.append(cap.shapley())
        except (ValueError, RuntimeError):
            pass
        # permutation null: a random preference set of the same size
        rp = []
        while len(rp) < len(prefs):
            a, c = rng.integers(0, m_alt, size=2)
            if a != c:
                rp.append((int(a), int(c)))
        try:
            capn, _ = S.cci(Z, rp)
            null.append(capn.shapley())
        except (ValueError, RuntimeError):
            pass

    boot = np.array(boot) if boot else np.zeros((0, len(CRIT)))
    null = np.array(null) if null else np.zeros((0, len(CRIT)))
    rows = []
    for i, c in enumerate(CRIT):
        rows.append({
            "criterion": c,
            "phi_point": float(phi0[i]),
            "phi_boot_mean": float(boot[:, i].mean()) if len(boot) else np.nan,
            "phi_boot_sd": float(boot[:, i].std(ddof=1)) if len(boot) > 1 else np.nan,
            "phi_boot_q05": float(np.quantile(boot[:, i], 0.05)) if len(boot) else np.nan,
            "phi_boot_q95": float(np.quantile(boot[:, i], 0.95)) if len(boot) else np.nan,
            "phi_null_mean": float(null[:, i].mean()) if len(null) else np.nan,
            "phi_null_q95": float(np.quantile(null[:, i], 0.95)) if len(null) else np.nan,
            "exceeds_null_q95": bool(phi0[i] > np.quantile(null[:, i], 0.95)) if len(null) else False,
        })
    summ = pd.DataFrame(rows)
    summ.attrs["n_boot"] = int(len(boot))
    summ.attrs["n_null"] = int(len(null))
    summ.attrs["max_abs_interaction"] = float(np.abs(I0).max())
    return pd.DataFrame(loo), summ


def rank_reversal(case_id: str, par: S.GameParams) -> pd.DataFrame:
    """Delete one non-recommended posture at a time and re-run the whole pipeline,
    checking whether the relative order of the survivors is preserved."""
    full = solve_case(case_id, par)
    rec = full["mcr"]["recommended"]
    base_scores = np.asarray(full["cap"].choquet(full["Z"]), float).ravel()
    rows = []
    for drop in range(len(POST)):
        if drop == rec:
            continue
        keep = [i for i in range(len(POST)) if i != drop]
        Zs = full["Z"][keep]
        remap = {old: new for new, old in enumerate(keep)}
        prefs = [(remap[a], remap[b]) for (a, b) in full["prefs"]
                 if a in remap and b in remap]
        if not prefs:
            continue
        cap2, _ = S.cci(Zs, prefs)
        U2 = UncertaintySet(n=len(CRIT), centre=cap2.vector, epsilon=EPS_DEFAULT,
                            preference_rows=preference_rows(Zs, prefs))
        mcr2 = mcr_ranking(Zs, U2)
        from scipy.stats import rankdata
        before = rankdata(-base_scores[keep])
        after = rankdata(-np.asarray(cap2.choquet(Zs), float).ravel())
        rows.append({
            "dropped": POST[drop],
            "choquet_order_preserved": bool(np.array_equal(before, after)),
            "mcr_recommendation": POST[keep[mcr2["recommended"]]],
            "recommendation_stable": bool(keep[mcr2["recommended"]] == rec),
        })
    return pd.DataFrame(rows)


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    ap.add_argument("--epsilon", type=float, default=EPS_DEFAULT)
    args = ap.parse_args()
    out = os.path.normpath(args.out)
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    base = S.GameParams()
    digest: dict = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "package_version": S.__version__,
                    "epsilon": args.epsilon, "monte_carlo_draws": args.draws,
                    "default_parameters": asdict(base),
                    "criteria": [{"code": c, "name": n, "sense": s} for c, n, s in S.CRITERIA],
                    "postures": [{"code": c, "description": d} for c, d in S.POSTURES],
                    "sectors": {k: {"hs6": v["hs6"], "trade_group": v["group"],
                                    "labour_intensity": v["labour"]}
                                for k, v in SECTOR_DEF.items()},
                    "cases": {}}
    sheets: dict[str, pd.DataFrame] = {}

    for case_id in CASES:
        print(f"\n=== {case_id} : {CASES[case_id]['label']} ===")
        r = solve_case(case_id, base, epsilon=args.epsilon)

        raw, nrm = table_performance(r, case_id)
        caps, inter = table_capacity(r)
        mcr_df = table_mcr(r)
        bench, agree = table_benchmarks(r)
        src = table_sourcing(r, case_id)
        att = table_regret_attribution(r)
        prov = r["case"].provenance

        print(mcr_df[["choquet_score", "score_rank", "max_regret", "mcr_rank"]].to_string())
        print(f"  stability radius  L1={mcr_df.attrs['stability_radius_l1']:.4f} "
              f"Linf={mcr_df.attrs['stability_radius_linf']:.4f}")

        print("  Monte Carlo ...", end="", flush=True)
        mc = monte_carlo(case_id, base, args.draws)
        print(f" {len(mc)} valid draws")
        print("  OAT sensitivity ...", end="", flush=True)
        oat = sensitivity_oat(case_id, base)
        print(f" {len(oat)} points")
        print("  Rank reversal ...", end="", flush=True)
        rr = rank_reversal(case_id, base)
        print(f" {len(rr)} probes")
        print("  Preference resampling ...", end="", flush=True)
        loo, boot = preference_resampling(case_id, base, n_boot=200)
        print(f" {len(loo)} leave-one-out, {boot.attrs['n_boot']} bootstrap draws")

        for nm, df in [("performance_raw", raw), ("performance_norm", nrm),
                       ("capacity", caps), ("interaction", inter), ("mcr", mcr_df),
                       ("benchmarks", bench), ("rank_agreement", agree),
                       ("montecarlo", mc), ("sensitivity_oat", oat),
                       ("rank_reversal", rr), ("sourcing", src),
                       ("regret_attribution", att), ("provenance", prov),
                       ("leave_one_out", loo), ("bootstrap_importance", boot)]:
            path = os.path.join(out, f"{nm}_{case_id}.csv")
            df.to_csv(path, index=(nm not in ("benchmarks", "montecarlo", "sensitivity_oat",
                                              "rank_reversal", "sourcing", "provenance",
                                              "leave_one_out", "bootstrap_importance")))
            sheets[f"{nm}_{case_id}"[:31]] = df

        top_freq = {p: float((mc[f"rank_{p}"] == 1).mean()) for p in POST} if len(mc) else {}
        mean_rank = {p: float(mc[f"rank_{p}"].mean()) for p in POST} if len(mc) else {}
        digest["cases"][case_id] = {
            "label": CASES[case_id]["label"],
            "origins": r["case"].env.origins,
            "rival": r["case"].rival,
            "revealed_preferences": [
                {"preferred": a, "over": b, "justification": j}
                for a, b, j in REVEALED_PREFERENCES[case_id]
            ],
            "shapley_importance": dict(zip(CRIT, r["cap"].shapley().round(6).tolist())),
            "shapley_of_game": dict(zip(CRIT, np.round(r["info"]["shapley_of_game"], 6).tolist())),
            "cci": {"rounds": r["info"]["rounds"], "cuts": r["info"]["cuts"],
                    "objective": round(float(r["info"]["objective"]), 6),
                    "monotonicity_violation": float(r["info"]["monotonicity_violation"]),
                    "preference_violations": int(r["info"].get("preference_violations", 0)),
                    "preference_margins": [round(float(x), 6)
                                           for x in r["info"].get("preference_margins", [])]},
            "orness": round(r["cap"].orness(), 6),
            "marichal_entropy": round(r["cap"].marichal_entropy(), 6),
            "shapley_entropy": round(r["cap"].shapley_entropy(), 6),
            "min_shapley": r["info"].get("min_shapley"),
            "ridge": r["info"].get("ridge"),
            "choquet_scores": dict(zip(POST, np.asarray(r["cap"].choquet(r["Z"]), float).round(6).tolist())),
            "max_regret": dict(zip(POST, np.round(r["mcr"]["regret"], 6).tolist())),
            "mcr_rank": dict(zip(POST, r["mcr"]["rank"].tolist())),
            "recommended": POST[r["mcr"]["recommended"]],
            "stability_radius_l1": round(float(mcr_df.attrs["stability_radius_l1"]), 6),
            "stability_radius_linf": round(float(mcr_df.attrs["stability_radius_linf"]), 6),
            "stability_challenger": mcr_df.attrs["stability_challenger_linf"],
            "bwm_consistency_xi": round(float(bench.attrs["bwm_xi"]), 6),
            "marichal_roubens": {
                "separation": round(float(bench.attrs["mr_separation"]), 6),
                "shapley": [round(x, 6) for x in bench.attrs["mr_shapley"]],
                "phi_l1_distance_from_cci": round(float(bench.attrs["mr_phi_l1_distance"]), 6),
                "kendall_tau_vs_ssr": float(
                    agree.loc["SSR-MCDA (Choquet score)", "Choquet (Marichal-Roubens capacity)"]),
            },
            "preference_resampling": {
                "n_bootstrap": int(boot.attrs["n_boot"]),
                "leave_one_out_min_tau": float(loo["kendall_tau_vs_base"].min()) if len(loo) else None,
                "leave_one_out_recommendation_stable": bool(
                    (loo["top_alternative"] == POST[r["mcr"]["recommended"]]).all()) if len(loo) else None,
                "phi_boot_sd": {c: round(float(boot.loc[boot.criterion == c, "phi_boot_sd"].iloc[0]), 6)
                                for c in CRIT},
                "criteria_exceeding_permutation_null": [
                    c for c in CRIT
                    if bool(boot.loc[boot.criterion == c, "exceeds_null_q95"].iloc[0])],
            },
            "weights": {k: np.round(v, 6).tolist() for k, v in bench.attrs["weights"].items()},
            "smaa_rank1_acceptability": dict(
                zip(POST, [round(x[0], 4) for x in bench.attrs["smaa_acceptability"]])),
            "monte_carlo": {
                "n_valid": int(len(mc)),
                "n_failed": int(len(mc.attrs.get("failures", []))),
                "draws_with_preference_violation": int((mc["preference_violations"] > 0).sum())
                    if "preference_violations" in mc else 0,
                "rank1_frequency": top_freq, "mean_rank": mean_rank},
            "rank_reversal": {"probes": int(len(rr)),
                              "recommendation_stable_share":
                                  float(rr["recommendation_stable"].mean()) if len(rr) else None,
                              "choquet_order_preserved_share":
                                  float(rr["choquet_order_preserved"].mean()) if len(rr) else None},
            "kendall_tau_vs_benchmarks": {
                m: float(agree.loc["SSR-MCDA (minimax regret)", m])
                for m in agree.columns if m != "SSR-MCDA (minimax regret)"},
        }

    with pd.ExcelWriter(os.path.join(out, "ssr_mcda_results.xlsx"), engine="xlsxwriter") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31])
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(digest, f, indent=2)

    print(f"\nWrote {len(sheets)} tables to {out} in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
