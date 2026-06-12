#!/usr/bin/env python
"""Headless verification of app.py — run with `uv run python verify_app.py`.

Checks both WASM-portability and the app's pedagogical punchlines, so a
changed seed or numpy upgrade can't silently turn the story false:

1. app.py parses and no two cells define the same global name.
2. Model invariants: agents conserved every round, cell values stay in
   {-1, 0, 1}, identical seeds give identical histories.
3. agent_stats matches hand-computed 3x3 examples (incl. empty houses).
4. The section-2 punchline (t = 0.30): similarity rises from ~50% to
   >= 65%, everyone ends up happy, the city settles within 30 rounds.
5. The sweep punchlines (section 4): assortativity ~0 in a mixed city,
   rising with t to near-total segregation at t = 0.70, then collapsing
   past t ~ 0.75 where the city cannot settle and most agents stay
   unhappy — and the sweep stays fast enough for the browser.
6. Every always-rendered view cell runs headless without errors.
"""

import ast
import sys
import time


def check_unique_defs():
    tree = ast.parse(open("app.py", encoding="utf-8").read())
    owner = {}
    dups = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not any(
            isinstance(dec, ast.Attribute) and dec.attr == "cell"
            for dec in node.decorator_list
        ):
            continue
        defs = set()
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    for name_node in ast.walk(target):
                        if isinstance(name_node, ast.Name):
                            defs.add(name_node.id)
            elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
                if isinstance(stmt.target, ast.Name):
                    defs.add(stmt.target.id)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defs.add(stmt.name)
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                for alias in stmt.names:
                    defs.add(alias.asname or alias.name.split(".")[0])
        for name in defs:
            if name.startswith("_"):
                continue
            if name in owner:
                dups.append(f"{name} (cells {owner[name]} and {node.name})")
            owner[name] = node.name
    assert not dups, f"duplicate global definitions: {dups}"
    print(f"OK  app.py parses; {len(owner)} globals, no duplicates")


def main():
    check_unique_defs()

    import numpy as np

    import app as app_module

    # Cell.run() executes ancestor cells automatically.
    _, h = app_module.helpers.run()
    agent_stats = h["agent_stats"]
    run_schelling = h["run_schelling"]
    assortativity = h["assortativity"]
    GRID_SIZE = h["GRID_SIZE"]
    EMPTY_SHARE = h["EMPTY_SHARE"]
    N_ROUNDS = h["N_ROUNDS"]
    SIM_SEED = h["SIM_SEED"]

    # --- 2. invariants -----------------------------------------------------
    grids, sims, haps, moves = run_schelling(20, 0.25, 0.4, 30, 5)
    n_blue = int((grids[0] == 1).sum())
    n_orange = int((grids[0] == -1).sum())
    for g in grids:
        assert set(np.unique(g)).issubset({-1, 0, 1}), "unexpected cell value"
        assert int((g == 1).sum()) == n_blue, "blue agents not conserved"
        assert int((g == -1).sum()) == n_orange, "orange agents not conserved"
    assert not np.isnan(sims).any() and not np.isnan(haps).any()
    grids2, sims2, _, _ = run_schelling(20, 0.25, 0.4, 30, 5)
    assert np.array_equal(grids[-1], grids2[-1]), "same seed, different result"
    assert np.allclose(sims, sims2), "same seed, different similarity curve"
    print("OK  agents conserved, values clean, runs deterministic")

    # --- 3. hand-computed examples -----------------------------------------
    city = np.array([[1, -1, -1], [1, 1, -1], [-1, -1, -1]], dtype=np.int8)
    # centre agent: 2 of 8 neighbours similar = 0.25
    _, _, unhappy_30 = agent_stats(city, 0.30)
    _, _, unhappy_25 = agent_stats(city, 0.25)
    assert unhappy_30[1, 1], "centre (2/8 = 0.25) should be unhappy at t=0.30"
    assert not unhappy_25[1, 1], "centre (0.25) should be happy at t=0.25"
    city_e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.int8)
    mean_sim_e, _, unhappy_e = agent_stats(city_e, 0.30)
    # ratios: corner blue 1/1, centre blue 1/2, corner orange 0/1 -> mean 0.5
    assert abs(mean_sim_e - 0.5) < 1e-12, f"empty-house maths off: {mean_sim_e}"
    assert unhappy_e[2, 2] and not unhappy_e[0, 0], "empty-house happiness off"
    print("OK  similarity ratio matches hand-computed neighbourhoods")

    # --- 4. section-2 punchline ---------------------------------------------
    tg, ts, th, tm = run_schelling(GRID_SIZE, EMPTY_SHARE, 0.30, N_ROUNDS, SIM_SEED)
    assert 0.45 <= ts[0] <= 0.55, f"mixed city should start ~50%, got {ts[0]:.3f}"
    assert ts[-1] >= 0.65, f"t=0.30 should end >= 65% similar, got {ts[-1]:.3f}"
    assert ts[-1] <= 0.85, f"t=0.30 should not look total, got {ts[-1]:.3f}"
    assert th[-1] >= 0.95, f"city should settle happy, got {th[-1]:.3f}"
    assert tm[-1] == 0, f"should settle within {N_ROUNDS} rounds"
    r0, r1 = assortativity(tg[0]), assortativity(tg[-1])
    assert abs(r0) < 0.10, f"mixed city should have r~0, got {r0:.3f}"
    assert r1 > 0.35, f"settled city should be assortative, got {r1:.3f}"
    print(
        f"OK  punchline: {ts[0]:.0%} -> {ts[-1]:.0%} similar "
        f"(assortativity {r0:.2f} -> {r1:.2f}), settled after "
        f"{len(tm)} rounds (seed {SIM_SEED})"
    )

    # --- 5. sweep punchlines -------------------------------------------------
    t0 = time.perf_counter()
    _, s4 = app_module.s4_data.run()
    sweep_seconds = time.perf_counter() - t0
    levels = s4["sweep_levels"]
    a_mean = s4["sweep_assort"].mean(axis=1)
    h_mean = s4["sweep_happy"].mean(axis=1)

    def at(level):
        return int(np.argmin(np.abs(levels - level)))

    assert a_mean[at(0.2)] < a_mean[at(0.3)] < a_mean[at(0.5)], (
        "assortativity should rise with t in the moderate range"
    )
    assert a_mean[at(0.7)] > 0.9, (
        f"t=0.70 should be near-total, got {a_mean[at(0.7)]:.3f}"
    )
    assert a_mean[at(0.8)] < 0.2, (
        f"t=0.80 should collapse towards mixed, got {a_mean[at(0.8)]:.3f}"
    )
    assert h_mean[at(0.8)] < 0.5, (
        f"t=0.80 should leave most agents unhappy, got {h_mean[at(0.8)]:.3f}"
    )
    print(
        f"OK  sweep: assortativity {a_mean[at(0.2)]:.2f} (t=0.2) -> "
        f"{a_mean[at(0.7)]:.2f} (t=0.7), collapse to {a_mean[at(0.8)]:.2f} "
        f"at t=0.80 ({sweep_seconds:.1f}s)"
    )
    assert sweep_seconds < 10, f"sweep too slow for the browser: {sweep_seconds:.1f}s"

    # --- 6. all view cells run headless --------------------------------------
    for cell_name in (
        "sidebar",
        "title",
        "s1_view",
        "s2_view",
        "pg_view",
        "s4_view",
        "footer",
    ):
        getattr(app_module, cell_name).run()
    print("OK  all view cells render headless without errors")

    print("\nAll checks passed.")


if __name__ == "__main__":
    sys.exit(main())
