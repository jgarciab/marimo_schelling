import marimo

__generated_with = "0.23.6"
app = marimo.App(
    width="medium",
    app_title="Schelling segregation model",
)


@app.cell
def imports():
    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#333333",
        "text.color": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 130,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.color": "#cccccc",
    })

    BLUE = "#0b789d"      # group A agents — UU blue, the app accent
    ORANGE = "#e07b00"    # group B agents
    EMPTY_COLOR = "#f2f2f2"
    ACCENT = BLUE
    HAPPY_GREEN = "#3F7D52"
    UNHAPPY_RED = "#d1495b"
    EDGE_COLOR = "#aaaaaa"
    # City grids store -1 (orange), 0 (empty house), 1 (blue).
    CITY_CMAP = ListedColormap([ORANGE, EMPTY_COLOR, BLUE])
    return (
        ACCENT,
        BLUE,
        CITY_CMAP,
        EDGE_COLOR,
        EMPTY_COLOR,
        HAPPY_GREEN,
        ListedColormap,
        ORANGE,
        UNHAPPY_RED,
        matplotlib,
        mo,
        np,
        plt,
    )


@app.cell
def helpers(CITY_CMAP, np):
    GRID_SIZE = 30
    EMPTY_SHARE = 0.20
    SIM_SEED = 2026
    N_ROUNDS = 30    # rounds simulated in section 2
    PG_ROUNDS = 40   # rounds simulated in the playground

    def neighbor_counts(mask):
        """How many of the (up to 8) surrounding cells of every cell are True."""
        _p = np.pad(mask.astype(np.int32), 1)
        return (
            _p[:-2, :-2] + _p[:-2, 1:-1] + _p[:-2, 2:]
            + _p[1:-1, :-2] + _p[1:-1, 2:]
            + _p[2:, :-2] + _p[2:, 1:-1] + _p[2:, 2:]
        )

    def make_city(size, empty_share, rng):
        """Random city: half blue (1), half orange (-1), rest empty (0)."""
        _n = size * size
        _cells = np.zeros(_n, dtype=np.int8)
        _n_agents = round(_n * (1.0 - empty_share))
        _cells[: _n_agents // 2] = -1
        _cells[_n_agents // 2 : _n_agents] = 1
        rng.shuffle(_cells)
        return _cells.reshape(size, size)

    def agent_stats(city, threshold):
        """(mean similarity ratio, share of happy agents, unhappy mask).

        Similarity ratio of an agent = similar occupied neighbours / occupied
        neighbours; the mean is over agents with at least one occupied
        neighbour. Agents with no occupied neighbours count as happy.
        """
        _blue_nb = neighbor_counts(city == 1)
        _orange_nb = neighbor_counts(city == -1)
        _occ = _blue_nb + _orange_nb
        _same = np.where(city == 1, _blue_nb, _orange_nb)
        _agents = city != 0
        _has_nb = _agents & (_occ > 0)
        _ratio = np.divide(
            _same, _occ, out=np.ones_like(_same, dtype=float), where=_occ > 0
        )
        _mean_sim = float(_ratio[_has_nb].mean()) if _has_nb.any() else 1.0
        _unhappy = _has_nb & (_ratio < threshold)
        _share_happy = 1.0 - _unhappy.sum() / max(int(_agents.sum()), 1)
        return _mean_sim, float(_share_happy), _unhappy

    def schelling_step(city, threshold, rng):
        """One round: every unhappy agent moves to a random empty house."""
        _, _, _unhappy = agent_stats(city, threshold)
        _movers = np.flatnonzero(_unhappy.ravel())
        if _movers.size == 0:
            return city, 0
        _flat = city.copy().ravel()
        _groups = _flat[_movers].copy()
        _flat[_movers] = 0
        _empties = np.flatnonzero(_flat == 0)
        _flat[rng.choice(_empties, size=_movers.size, replace=False)] = _groups
        return _flat.reshape(city.shape), int(_movers.size)

    def run_schelling(size, empty_share, threshold, n_rounds, seed):
        """Full history (stops early once nobody wants to move).

        Returns (grids, similarity, happiness, moves): n+1 grids, two arrays
        of length n+1, and the number of agents moved in each of n rounds.
        """
        _rng = np.random.default_rng(seed)
        _city = make_city(size, empty_share, _rng)
        _s, _h, _ = agent_stats(_city, threshold)
        _grids, _sims, _haps, _moves = [_city.copy()], [_s], [_h], []
        for _ in range(n_rounds):
            _city, _n_moved = schelling_step(_city, threshold, _rng)
            _s, _h, _ = agent_stats(_city, threshold)
            _grids.append(_city.copy())
            _sims.append(_s)
            _haps.append(_h)
            _moves.append(_n_moved)
            if _n_moved == 0:
                break
        return _grids, np.array(_sims), np.array(_haps), np.array(_moves)

    def assortativity(city):
        """Assortativity by colour of the neighbour network (-1 to 1).

        Newman's coefficient for a categorical attribute, computed from the
        blue-blue / orange-orange / mixed counts of adjacent agent pairs.
        0 means random mixing, 1 complete segregation.
        """
        _blue = city == 1
        _orange = city == -1
        _e_bb = int(neighbor_counts(_blue)[_blue].sum()) / 2
        _e_oo = int(neighbor_counts(_orange)[_orange].sum()) / 2
        _e_bo = int(neighbor_counts(_orange)[_blue].sum())
        _e = _e_bb + _e_oo + _e_bo
        if _e == 0:
            return 0.0
        _a_b = (_e_bb + _e_bo / 2.0) / _e
        _a_o = (_e_oo + _e_bo / 2.0) / _e
        _denom = 1.0 - _a_b**2 - _a_o**2
        _same = (_e_bb + _e_oo) / _e
        return float((_same - _a_b**2 - _a_o**2) / _denom) if _denom > 0 else 1.0

    def sweep_thresholds(levels, size, empty_share, n_rounds, seeds):
        """Final assortativity and share happy per threshold x seed."""
        _assort = np.zeros((len(levels), len(seeds)))
        _hap = np.zeros_like(_assort)
        for _i, _thr in enumerate(levels):
            for _j, _seed in enumerate(seeds):
                _grids, _, _h, _ = run_schelling(
                    size, empty_share, _thr, n_rounds, _seed
                )
                _assort[_i, _j] = assortativity(_grids[-1])
                _hap[_i, _j] = _h[-1]
        return _assort, _hap

    def draw_city(ax, city, title=None):
        """Render a city grid: blue/orange agents, light grey empty houses."""
        ax.pcolormesh(
            np.flipud(city),
            cmap=CITY_CMAP,
            vmin=-1,
            vmax=1,
            edgecolors="white",
            linewidth=0.3,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for _spine in ax.spines.values():
            _spine.set_visible(False)
        if title:
            ax.set_title(title, fontsize=12)

    return (
        EMPTY_SHARE,
        GRID_SIZE,
        N_ROUNDS,
        PG_ROUNDS,
        SIM_SEED,
        agent_stats,
        assortativity,
        draw_city,
        make_city,
        neighbor_counts,
        run_schelling,
        schelling_step,
        sweep_thresholds,
    )


@app.cell
def theme(ACCENT, mo):
    css = mo.Html(
        f"""
        <style>
        :root {{
          --sch-accent: {ACCENT};
        }}
        .sch-callout {{
          border-left: 4px solid var(--sch-accent);
          padding: 0.65rem 0.85rem;
          background: color-mix(in srgb, var(--sch-accent) 8%, white);
          margin: 0.4rem 0 0.8rem;
        }}
        </style>
        """
    )
    return (css,)


@app.cell
def sidebar(BLUE, EMPTY_COLOR, ORANGE, mo):
    mo.sidebar(
        [
            mo.md("### Schelling's model"),
            mo.md(
                """
                1. One house, one rule
                2. Watch it unfold
                3. Playground
                4. Demands vs segregation
                """
            ),
            mo.md("---"),
            mo.Html(
                f"""
                <div style="font-size:0.9rem; line-height:1.7;">
                <span style="background:{BLUE}; color:white;
                  padding:0.05rem 0.45rem; border-radius:0.25rem;">blue</span>
                and
                <span style="background:{ORANGE}; color:white;
                  padding:0.05rem 0.45rem; border-radius:0.25rem;">orange</span>
                agents,
                <span style="background:{EMPTY_COLOR}; color:#666666;
                  padding:0.05rem 0.45rem; border-radius:0.25rem;
                  border:1px solid #dddddd;">empty</span> houses.
                </div>
                """
            ),
            mo.md(
                """
                **The whole model:** an agent with fewer than a share *t* of
                similar neighbours moves to a random empty house. Repeat until
                everyone is content.
                """
            ),
        ]
    )
    return


@app.cell
def title(css, mo):
    mo.vstack(
        [
            css,
            mo.md(
                """
                # One rule, a divided city: the Schelling model

                *Javier Garcia-Bernardo — Utrecht University*

                **Motivating question:** census maps of cities like New York or
                Chicago show segregation street by street. Does a strongly
                segregated city require strongly intolerant residents?

                In 1971 the economist (and later Nobel laureate) Thomas
                Schelling answered with a model so simple it fits in two
                lines — and with one of the most famous punchlines in social
                science.
                """
            ),
            mo.Html(
                """
                <div class="sch-callout">
                <b>The whole model.</b> Two kinds of agents live on a grid of
                houses; some houses are empty.
                <ol style="margin:0.3rem 0 0.1rem;">
                <li>An agent is <b>happy</b> if at least a share <i>t</i> of
                its neighbours are similar to it. Higher <i>t</i> = more
                demanding (less tolerant) agents.</li>
                <li>An unhappy agent <b>moves to a random empty house</b>.</li>
                </ol>
                Repeat until everyone is happy. That's it — no prices, no
                history, no institutions.
                </div>
                """
            ),
        ]
    )
    return


@app.cell
def s1_widgets(mo):
    s1_similar = mo.ui.slider(
        start=0,
        stop=8,
        step=1,
        value=2,
        label="Similar neighbours (of 8)",
        show_value=True,
    )
    s1_threshold = mo.ui.slider(
        start=0.0,
        stop=0.8,
        step=0.05,
        value=0.30,
        label="Similarity threshold t",
        show_value=True,
    )
    return s1_similar, s1_threshold


@app.cell
def s1_view(
    BLUE,
    CITY_CMAP,
    EDGE_COLOR,
    HAPPY_GREEN,
    ORANGE,
    UNHAPPY_RED,
    mo,
    np,
    plt,
    s1_similar,
    s1_threshold,
):
    _k = int(s1_similar.value)
    _t = float(s1_threshold.value)
    _ratio = _k / 8.0
    _happy = _ratio >= _t

    # Order in which the 8 neighbours of the centre house are filled in.
    _nb_positions = [
        (1, 1), (1, 2), (1, 3),
        (2, 1),         (2, 3),
        (3, 1), (3, 2), (3, 3),
    ]

    _fig, (_ax_net, _ax_zoom) = plt.subplots(
        1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1.25, 1]},
        layout="constrained",
    )

    # Left: the city seen as a network — houses are nodes, adjacency edges.
    for _r in range(5):
        for _c in range(5):
            for _dr, _dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
                _r2, _c2 = _r + _dr, _c + _dc
                if 0 <= _r2 < 5 and 0 <= _c2 < 5:
                    _on_center = (_r, _c) == (2, 2) or (_r2, _c2) == (2, 2)
                    _ax_net.plot(
                        [_c, _c2],
                        [-_r, -_r2],
                        color="#555555" if _on_center else EDGE_COLOR,
                        linewidth=1.8 if _on_center else 0.8,
                        zorder=1,
                    )
    _nb_colors = {
        _pos: (BLUE if _i < _k else ORANGE)
        for _i, _pos in enumerate(_nb_positions)
    }
    for _r in range(5):
        for _c in range(5):
            if (_r, _c) == (2, 2):
                _color, _size, _edge = BLUE, 320, "#333333"
            elif (_r, _c) in _nb_colors:
                _color, _size, _edge = _nb_colors[(_r, _c)], 230, "white"
            else:
                _color, _size, _edge = "#cccccc", 130, "white"
            _ax_net.scatter(
                _c, -_r, s=_size, color=_color, edgecolor=_edge,
                linewidth=1.4, zorder=2,
            )
    _ax_net.set_title("Houses are nodes, adjacency is an edge")
    _ax_net.set_xlim(-0.6, 4.6)
    _ax_net.set_ylim(-4.6, 0.6)
    _ax_net.axis("off")

    # Right: the same centre agent and its 8 neighbours, as grid cells.
    _zoom = np.zeros((3, 3), dtype=np.int8)
    _zoom[1, 1] = 1
    for _i, (_r, _c) in enumerate(_nb_positions):
        _zoom[_r - 1, _c - 1] = 1 if _i < _k else -1
    _ax_zoom.pcolormesh(
        np.flipud(_zoom), cmap=CITY_CMAP, vmin=-1, vmax=1,
        edgecolors="white", linewidth=2,
    )
    _ax_zoom.add_patch(
        plt.Rectangle((1, 1), 1, 1, fill=False, edgecolor="#333333",
                      linewidth=2.5)
    )
    _ax_zoom.set_aspect("equal")
    _ax_zoom.set_xticks([])
    _ax_zoom.set_yticks([])
    _ax_zoom.grid(False)
    for _spine in _ax_zoom.spines.values():
        _spine.set_visible(False)
    _ax_zoom.set_title("…and as cells of a grid")

    _badge_style = (
        "color:white; padding:0.12rem 0.6rem; border-radius:1rem; "
        "font-weight:600;"
    )
    _verdict = (
        f'<span style="background:{HAPPY_GREEN}; {_badge_style}">'
        "HAPPY — stays put</span>"
        if _happy
        else f'<span style="background:{UNHAPPY_RED}; {_badge_style}">'
        "UNHAPPY — moves to a random empty house</span>"
    )

    mo.vstack(
        [
            mo.md(
                """
                ## 1. One house, one rule

                A city like this is also a *network*: every house is a node,
                connected to the (up to) 8 houses around it. Both panels show
                the same thing — the blue agent in the centre and its 8
                neighbours.

                The agent computes its **similarity ratio** = similar
                neighbours ÷ occupied neighbours, and is happy if that ratio
                reaches its **similarity threshold *t***. Higher *t* means a
                more demanding, less tolerant agent.

                **Try this:** keep *t* = 0.30 and move **similar neighbours**
                from 2 to 3 — the verdict flips. Note how undemanding this
                rule is: at *t* = 0.30 an agent is perfectly happy as a clear
                *minority* (3 of 8 similar).
                """
            ),
            mo.hstack([s1_similar, s1_threshold], gap=1.5, wrap=True),
            _fig,
            mo.md(
                f"Similarity ratio = {_k}/8 = **{_ratio:.0%}** &nbsp;·&nbsp; "
                f"needs at least **{_t:.0%}** &nbsp;→&nbsp; {_verdict}"
            ),
        ]
    )
    return


@app.cell
def s2_widgets(N_ROUNDS, mo):
    s2_threshold = mo.ui.slider(
        start=0.0,
        stop=0.8,
        step=0.05,
        value=0.30,
        label="Similarity threshold t",
        show_value=True,
    )
    s2_round = mo.ui.slider(
        start=0,
        stop=N_ROUNDS,
        step=1,
        value=0,
        label="Round",
        show_value=True,
    )
    return s2_round, s2_threshold


@app.cell
def s2_sim(EMPTY_SHARE, GRID_SIZE, N_ROUNDS, SIM_SEED, run_schelling, s2_threshold):
    dyn_history = run_schelling(
        GRID_SIZE, EMPTY_SHARE, float(s2_threshold.value), N_ROUNDS, SIM_SEED
    )
    return (dyn_history,)


@app.cell
def s2_view(
    BLUE,
    HAPPY_GREEN,
    N_ROUNDS,
    draw_city,
    dyn_history,
    mo,
    plt,
    s2_round,
    s2_threshold,
):
    _grids, _sims, _haps, _moves = dyn_history
    _t = min(int(s2_round.value), len(_grids) - 1)
    _settled = len(_moves) > 0 and _moves[-1] == 0

    _fig, (_ax_g, _ax_c) = plt.subplots(
        1, 2, figsize=(7.4, 3.6), layout="constrained"
    )
    draw_city(_ax_g, _grids[_t], f"City at round {_t}")
    _ax_c.plot(
        range(len(_sims)), _sims, color=BLUE, linewidth=2.2,
        label="mean similarity",
    )
    _ax_c.plot(
        range(len(_haps)), _haps, color=HAPPY_GREEN, linewidth=2.2,
        label="share happy",
    )
    _ax_c.axvline(_t, color="#888888", linestyle="--", linewidth=1)
    _ax_c.set_xlim(0, N_ROUNDS)
    _ax_c.set_ylim(0, 1.05)
    _ax_c.set_xlabel("round")
    _ax_c.set_ylabel("fraction")
    _ax_c.set_title("Segregation and happiness")
    _ax_c.legend(loc="lower right", frameon=False)
    _ax_c.set_box_aspect(1)  # match the square city panel

    _status = (
        f"settled after {len(_moves)} rounds"
        if _settled
        else f"still moving after {N_ROUNDS} rounds"
    )

    mo.vstack(
        [
            mo.md(
                """
                ---

                ## 2. Watch it unfold

                Now a whole city: **30 × 30 houses**, 20% empty, the rest split
                evenly between blue and orange agents placed at random. Each
                round, every unhappy agent moves to a random empty house. The
                grid shows the city at the selected round; the curves show the
                whole run.

                **Try this, in order:**

                1. Keep *t* = 0.30 and drag **Round** from 0 to 30 — colour
                   blobs crystallise within ~15 rounds, and average similarity
                   climbs from ≈ 50% to ≈ **75%** — far more segregation than
                   the 30% anyone asked for. Then the curves go flat: everyone
                   is happy, nobody moves again.
                2. Set *t* = 0.05 — such undemanding agents are almost never
                   unhappy, so the city stays mixed.
                3. Set *t* = 0.60 — more demanding agents push similarity near
                   96%, after a longer period of churn.
                4. Set *t* = 0.80 — demands this high can almost never be met:
                   happiness collapses and the moving never stops. Hold that
                   thought for section 4.
                """
            ),
            mo.hstack([s2_threshold, s2_round], gap=1.5, wrap=True),
            _fig,
            mo.md(
                f"Round {_t}: mean similarity **{_sims[_t]:.0%}**, "
                f"**{_haps[_t]:.0%}** happy · {_status}."
            ),
            mo.Html(
                """
                <div class="sch-callout">
                <b>This is emergence.</b> Not a single agent wants
                segregation — each one is content as a 30% minority. Yet mild
                individual preferences plus movement produce a strongly
                segregated city. The pattern is a property of the
                <b>system</b>, not of any individual.
                </div>
                """
            ),
        ]
    )
    return


@app.cell
def pg_state(mo):
    # The playground round is shared state, so the Run / +1 / Reset buttons
    # and the slider can all drive the same value.
    get_pg_round, set_pg_round = mo.state(0)
    return get_pg_round, set_pg_round


@app.cell
def pg_params(mo):
    pg_size = mo.ui.slider(
        start=16, stop=40, step=2, value=30,
        label="City size (side)", show_value=True,
    )
    pg_empty = mo.ui.slider(
        start=0.05, stop=0.50, step=0.05, value=0.20,
        label="Share of empty houses", show_value=True,
    )
    pg_threshold = mo.ui.slider(
        start=0.0, stop=0.8, step=0.05, value=0.30,
        label="Similarity threshold t", show_value=True,
    )
    pg_seed = mo.ui.slider(
        start=1, stop=20, step=1, value=7,
        label="Random seed", show_value=True,
    )
    return pg_empty, pg_seed, pg_size, pg_threshold


@app.cell
def pg_controls(PG_ROUNDS, mo, set_pg_round):
    pg_play = mo.ui.refresh(
        options=["500ms", "1s"],
        default_interval="500ms",
        label="Run",
    )
    pg_step = mo.ui.button(
        label="+1 round",
        on_click=lambda _: set_pg_round(lambda _r: min(_r + 1, PG_ROUNDS)),
    )
    pg_reset = mo.ui.button(
        label="⏮ Reset",
        on_click=lambda _: set_pg_round(0),
    )
    return pg_play, pg_reset, pg_step


@app.cell
def pg_round_widget(PG_ROUNDS, get_pg_round, mo, set_pg_round):
    pg_round = mo.ui.slider(
        start=0,
        stop=PG_ROUNDS,
        step=1,
        value=get_pg_round(),
        on_change=set_pg_round,
        label="Round",
        show_value=True,
    )
    return (pg_round,)


@app.cell
def pg_tick(PG_ROUNDS, pg_play, set_pg_round):
    # Each tick of the Run timer advances the round by one (value is empty
    # until the timer is switched on, so nothing moves on page load).
    if pg_play.value:
        set_pg_round(lambda _r: min(_r + 1, PG_ROUNDS))
    return


@app.cell
def pg_sim(PG_ROUNDS, pg_empty, pg_seed, pg_size, pg_threshold, run_schelling):
    pg_history = run_schelling(
        int(pg_size.value),
        float(pg_empty.value),
        float(pg_threshold.value),
        PG_ROUNDS,
        int(pg_seed.value),
    )
    return (pg_history,)


@app.cell
def pg_view(
    BLUE,
    HAPPY_GREEN,
    PG_ROUNDS,
    assortativity,
    draw_city,
    mo,
    pg_empty,
    pg_history,
    pg_play,
    pg_reset,
    pg_round,
    pg_seed,
    pg_size,
    pg_step,
    pg_threshold,
    plt,
):
    _grids, _sims, _haps, _moves = pg_history
    _t = min(int(pg_round.value), len(_grids) - 1)
    _settled = len(_moves) > 0 and _moves[-1] == 0
    _r_final = assortativity(_grids[-1])

    _fig, (_ax_g, _ax_c) = plt.subplots(
        1, 2, figsize=(7.4, 3.6), layout="constrained"
    )
    draw_city(_ax_g, _grids[_t], f"City at round {_t}")
    _ax_c.plot(
        range(len(_sims)), _sims, color=BLUE, linewidth=2.2,
        label="mean similarity",
    )
    _ax_c.plot(
        range(len(_haps)), _haps, color=HAPPY_GREEN, linewidth=2.2,
        label="share happy",
    )
    _ax_c.axvline(_t, color="#888888", linestyle="--", linewidth=1)
    _ax_c.set_xlim(0, PG_ROUNDS)
    _ax_c.set_ylim(0, 1.05)
    _ax_c.set_xlabel("round")
    _ax_c.set_ylabel("fraction")
    _ax_c.set_title("Segregation and happiness")
    _ax_c.legend(loc="lower right", frameon=False, fontsize=9)
    _ax_c.set_box_aspect(1)  # match the square city panel

    _status = (
        f"settled after {len(_moves)} rounds"
        if _settled
        else f"still moving after {PG_ROUNDS} rounds"
    )

    mo.vstack(
        [
            mo.md(
                """
                ---

                ## 3. Playground

                All controls unlocked. Press **Run** to play through the
                rounds (switch it off to pause), **+1 round** to step, or
                **⏮ Reset** to go back to the start — the Round slider can
                also be dragged directly. Three experiments worth running:

                1. **Vacancy changes the speed, not the destination.** At
                   *t* = 0.30, try empty share 0.05 vs 0.40: final similarity
                   stays ≈ 75% either way, but with almost no empty houses the
                   city needs several times more rounds to settle.
                2. **Size doesn't matter.** Compare 16×16 with 40×40 at the
                   same threshold: the outcome is the same — the pattern is
                   not a small-grid artefact.
                3. **The vacancy escape hatch.** Set *t* = 0.75 with empty
                   share 0.10: endless churn, roughly half the city unhappy.
                   Now raise empty share to 0.40 — suddenly it *settles*, at
                   ≈ 100% similarity: perfect enclaves padded by empty space.
                """
            ),
            mo.hstack(
                [pg_size, pg_empty, pg_threshold, pg_seed], gap=1.5, wrap=True
            ),
            mo.hstack(
                [pg_play, pg_step, pg_reset, pg_round],
                gap=1.0,
                wrap=True,
                align="center",
            ),
            _fig,
            mo.md(
                f"Round {_t}: mean similarity **{_sims[_t]:.0%}**, "
                f"**{_haps[_t]:.0%}** happy. End of run: similarity "
                f"**{_sims[-1]:.0%}**, assortativity **{_r_final:.2f}** · "
                f"{_status}."
            ),
        ]
    )
    return


@app.cell
def s4_data(np, sweep_thresholds):
    sweep_levels = np.array(
        [0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50,
         0.60, 0.65, 0.70, 0.75, 0.80]
    )
    sweep_assort, sweep_happy = sweep_thresholds(
        sweep_levels, size=24, empty_share=0.20, n_rounds=50, seeds=(1, 2, 3)
    )
    return sweep_assort, sweep_happy, sweep_levels


@app.cell
def s4_view(
    BLUE,
    HAPPY_GREEN,
    UNHAPPY_RED,
    mo,
    plt,
    sweep_assort,
    sweep_happy,
    sweep_levels,
):
    _a_mean = sweep_assort.mean(axis=1)
    _h_mean = sweep_happy.mean(axis=1)

    _fig, (_ax_a, _ax_h) = plt.subplots(
        1, 2, figsize=(7.6, 3.4), sharex=True, layout="constrained"
    )

    _ax_a.fill_between(
        sweep_levels, sweep_assort.min(axis=1), sweep_assort.max(axis=1),
        color=BLUE, alpha=0.18, linewidth=0,
    )
    _ax_a.plot(sweep_levels, _a_mean, color=BLUE, linewidth=2.4, marker="o",
               markersize=4)
    _i_third = int((abs(sweep_levels - 0.30)).argmin())
    _ax_a.annotate(
        "the ⅓ rule",
        xy=(0.30, _a_mean[_i_third]),
        xytext=(0.02, 0.88),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )
    _ax_a.axvspan(0.72, 0.82, color=UNHAPPY_RED, alpha=0.08)
    _ax_a.set_xlim(-0.02, 0.82)
    _ax_a.set_ylim(-0.05, 1.05)
    _ax_a.set_xlabel("similarity threshold t")
    _ax_a.set_ylabel("final assortativity")
    _ax_a.set_title("Segregation vs demands")

    _ax_h.fill_between(
        sweep_levels, sweep_happy.min(axis=1), sweep_happy.max(axis=1),
        color=HAPPY_GREEN, alpha=0.18, linewidth=0,
    )
    _ax_h.plot(sweep_levels, _h_mean, color=HAPPY_GREEN, linewidth=2.4,
               marker="o", markersize=4)
    _ax_h.axvspan(0.72, 0.82, color=UNHAPPY_RED, alpha=0.08)
    _ax_h.text(0.97, 0.38, "no stable\ncity", color=UNHAPPY_RED, fontsize=10,
               ha="right", transform=_ax_h.transAxes)
    _ax_h.set_ylim(0, 1.05)
    _ax_h.set_xlabel("similarity threshold t")
    _ax_h.set_ylabel("share happy after 50 rounds")
    _ax_h.set_title("…and who is actually happy")

    mo.vstack(
        [
            mo.md(
                """
                ---

                ## 4. How much demand makes how much segregation?

                Each point runs the model to the end at that threshold, for
                three different random cities (the band shows the min–max).
                Segregation is measured as **assortativity**: do similar
                agents live next to each other? It is 0 when colours mix at
                random and 1 when the city is completely segregated.

                Two things should surprise you:

                1. **The curve is steep where you'd least expect it.** Between
                   *t* = 0.20 and *t* = 0.40 — agents who are all still happy
                   as minorities — assortativity jumps from ≈ 0.15 to ≈ 0.6.
                   By *t* = 0.70 segregation is near-total.
                2. **Then it collapses.** Past *t* ≈ 0.7 (red band), demands
                   are so high that agents can almost never all be happy: the
                   city churns forever and ends up looking mixed. Stable,
                   settled segregation is the product of *moderate,
                   satisfiable* preferences — not extreme ones.
                """
            ),
            _fig,
        ]
    )
    return


@app.cell
def footer(mo):
    mo.md(
        """
        ---

        _Javier Garcia-Bernardo — ODISSEI Social Data Science team (SoDa) &
        Department of Methodology and Statistics, Utrecht University._

        Based on Thomas Schelling's *Dynamic Models of Segregation* (1971).
        Inspired by
        [Adil Moujahid's streamlit app](https://github.com/adilmoujahid/streamlit-schelling)
        and Frank McCown's
        [Nifty assignment](http://nifty.stanford.edu/2014/mccown-schelling-model-segregation/).
        Built with [marimo](https://marimo.io/).
        """
    )
    return


if __name__ == "__main__":
    app.run()
