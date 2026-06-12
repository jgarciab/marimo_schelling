# Schelling segregation — a marimo teaching app

A self-contained [marimo](https://marimo.io) app that turns the Schelling
(1971) segregation model into a short, guided interactive lesson: mild
individual preferences plus movement produce a strongly segregated city
that no individual asked for.

## Live demo

**<https://javier.science/marimo_schelling/>**

The app is a static WebAssembly bundle that runs entirely in the browser —
no server needed. It is rebuilt and deployed to GitHub Pages on every push
to `main` (see [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)).

## What it does

Four sections, one storyline:

1. **One house, one rule** — a single agent and its 8 neighbours, shown both
   as a small network (houses = nodes, adjacency = edges) and as grid cells.
   Two sliders: similar neighbours and the similarity threshold *t*
   (higher *t* = more demanding, less tolerant agents).
2. **Watch it unfold** — scrub through the rounds of a 30×30 city while the
   mean-similarity and share-happy curves build up; with *t* = 0.30 the city
   goes from ≈ 50% to ≈ 75% similar — the famous emergence punchline.
3. **Playground** — city size, vacancy share, threshold and seed unlocked,
   with a **Run** timer that plays through the rounds, a **+1 round** step
   button, a **⏮ Reset** button, and a draggable Round slider.
4. **Demands vs segregation** — a threshold sweep (3 seeds, min–max band)
   measured as **assortativity** (0 = random mixing, 1 = complete
   segregation): a steep rise in the moderate range, near-total segregation
   by *t* = 0.7, then collapse past *t* ≈ 0.75 where the city can never
   settle.

## The model

- Grid city: half blue agents, half orange, a share of houses empty.
- Moore neighbourhood (up to 8 neighbours). An agent's **similarity ratio**
  = similar occupied neighbours ÷ occupied neighbours; agents with no
  occupied neighbours count as happy.
- An agent is unhappy if its ratio is below the threshold *t*; each round,
  **all** unhappy agents move to randomly chosen empty houses (synchronous
  rounds, which vectorise in numpy and keep everything interactive under
  Pyodide). The run stops as soon as a round moves nobody.
- Segregation is summarised two ways: the mean similarity ratio over agents,
  and Newman's assortativity coefficient computed from the blue–blue /
  orange–orange / mixed counts of adjacent agent pairs.

Everything is seeded and deterministic; there are no data files and no
network calls. Dependencies: marimo, numpy, matplotlib.

## Run locally

This folder lives on a cloud-synced drive that does not support symlinks,
so the uv venv lives outside the project (`~/.uv_envs/schelling_segregation`)
— `run.sh` and `export_wasm.sh` take care of that.

```bash
./run.sh --setup   # first time only: create venv + install deps
./run.sh           # serve the app locally (marimo run)
```

To edit the notebook instead: `UV_PROJECT_ENVIRONMENT=~/.uv_envs/schelling_segregation uv run marimo edit app.py`.

## Verify

```bash
UV_PROJECT_ENVIRONMENT=~/.uv_envs/schelling_segregation uv run python verify_app.py
```

The checks pin the *pedagogy*, not just the syntax: the ~50% → ~75%
punchline, the sweep's rise-then-collapse shape, hand-computed neighbourhood
maths, agent conservation, determinism, and that every view cell renders
headless.

## Deployment

The canonical WASM build happens in GitHub Actions, not locally:
sync deps → run `verify_app.py` → `marimo export html-wasm` into `build/`
→ deploy `build/` to GitHub Pages. `build/` (and any local export) is
gitignored; `export_wasm.sh` exists only for local previews.

Note: the WASM runtime is Pyodide, which ships its own package versions —
keep dependencies to numpy/matplotlib-level packages and check Pyodide
compatibility before adding anything new.

## Credits

- Thomas C. Schelling, *Dynamic Models of Segregation*, Journal of
  Mathematical Sociology (1971).
- Inspired by [Adil Moujahid's streamlit app](https://github.com/adilmoujahid/streamlit-schelling)
  and [Frank McCown's Nifty assignment](http://nifty.stanford.edu/2014/mccown-schelling-model-segregation/)
  (the implementation here is an independent, vectorised rewrite).
- Javier Garcia-Bernardo — ODISSEI Social Data Science team (SoDa) &
  Department of Methodology and Statistics, Utrecht University.
