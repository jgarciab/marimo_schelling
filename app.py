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

    VIOLET = "#3d348b"    # group A agents, and the app accent
    AMBER = "#e6af2e"     # group B agents
    EMPTY_COLOR = "#f2f2f2"
    ACCENT = VIOLET
    HAPPY_GREEN = "#3F7D52"
    UNHAPPY_RED = "#d1495b"
    EDGE_COLOR = "#aaaaaa"
    # City grids store -1 (amber), 0 (empty house), 1 (violet).
    CITY_CMAP = ListedColormap([AMBER, EMPTY_COLOR, VIOLET])
    return (
        ACCENT,
        AMBER,
        CITY_CMAP,
        EDGE_COLOR,
        EMPTY_COLOR,
        HAPPY_GREEN,
        ListedColormap,
        UNHAPPY_RED,
        VIOLET,
        matplotlib,
        mo,
        np,
        plt,
    )


@app.cell
def helpers(CITY_CMAP, np):
    PG_ROUNDS = 40  # rounds simulated for the city

    def neighbor_counts(mask):
        """How many of the (up to 8) surrounding cells of every cell are True."""
        _p = np.pad(mask.astype(np.int32), 1)
        return (
            _p[:-2, :-2] + _p[:-2, 1:-1] + _p[:-2, 2:]
            + _p[1:-1, :-2] + _p[1:-1, 2:]
            + _p[2:, :-2] + _p[2:, 1:-1] + _p[2:, 2:]
        )

    def make_city(size, empty_share, rng):
        """Random city: half violet (1), half amber (-1), rest empty (0)."""
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
        _violet_nb = neighbor_counts(city == 1)
        _amber_nb = neighbor_counts(city == -1)
        _occ = _violet_nb + _amber_nb
        _same = np.where(city == 1, _violet_nb, _amber_nb)
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
        violet-violet / amber-amber / mixed counts of adjacent agent pairs.
        0 means random mixing, 1 complete segregation.
        """
        _violet = city == 1
        _amber = city == -1
        _e_vv = int(neighbor_counts(_violet)[_violet].sum()) / 2
        _e_aa = int(neighbor_counts(_amber)[_amber].sum()) / 2
        _e_va = int(neighbor_counts(_amber)[_violet].sum())
        _e = _e_vv + _e_aa + _e_va
        if _e == 0:
            return 0.0
        _a_v = (_e_vv + _e_va / 2.0) / _e
        _a_a = (_e_aa + _e_va / 2.0) / _e
        _denom = 1.0 - _a_v**2 - _a_a**2
        _same = (_e_vv + _e_aa) / _e
        return float((_same - _a_v**2 - _a_a**2) / _denom) if _denom > 0 else 1.0

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
        """Render a city grid: violet/amber agents, light grey empty houses."""
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
        PG_ROUNDS,
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
def pg_state(mo):
    # Shared state so the sidebar slider and buttons drive the same values.
    get_pg_round, set_pg_round = mo.state(0)
    get_city_seed, set_city_seed = mo.state(1)
    return get_city_seed, get_pg_round, set_city_seed, set_pg_round


@app.cell
def pg_params(mo):
    pg_size = mo.ui.slider(
        start=16, stop=40, step=2, value=30,
        label="City size", show_value=True,
    )
    pg_empty = mo.ui.slider(
        start=0.05, stop=0.50, step=0.05, value=0.20,
        label="Empty share", show_value=True,
    )
    pg_threshold = mo.ui.slider(
        start=0.0, stop=0.8, step=0.05, value=0.30,
        label="Similarity threshold t", show_value=True,
    )
    return pg_empty, pg_size, pg_threshold


@app.cell
def pg_controls(PG_ROUNDS, mo, set_city_seed, set_pg_round):
    pg_step = mo.ui.button(
        label="+1 round",
        full_width=True,
        on_click=lambda _: set_pg_round(lambda _r: min(_r + 1, PG_ROUNDS)),
    )
    pg_newcity = mo.ui.button(
        label="New random city",
        full_width=True,
        on_click=lambda _: (
            set_city_seed(lambda _s: _s + 1),
            set_pg_round(0),
        ),
    )
    return pg_newcity, pg_step


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
def sidebar(
    AMBER,
    EMPTY_COLOR,
    VIOLET,
    mo,
    pg_empty,
    pg_newcity,
    pg_round,
    pg_size,
    pg_step,
    pg_threshold,
):
    mo.sidebar(
        [
            mo.md("### Schelling's model"),
            mo.Html(
                f"""
                <div style="font-size:0.9rem; line-height:1.7;">
                <span style="background:{VIOLET}; color:white;
                  padding:0.05rem 0.45rem; border-radius:0.25rem;">violet</span>
                and
                <span style="background:{AMBER}; color:#333333;
                  padding:0.05rem 0.45rem; border-radius:0.25rem;">amber</span>
                agents,
                <span style="background:{EMPTY_COLOR}; color:#666666;
                  padding:0.05rem 0.45rem; border-radius:0.25rem;
                  border:1px solid #dddddd;">empty</span> houses.
                </div>
                """
            ),
            mo.md(
                """
                **The rule:** an agent with fewer than a share *t* of similar
                neighbours moves to a random empty house.
                """
            ),
            mo.md("---"),
            mo.md("### Controls"),
            pg_size,
            pg_empty,
            pg_threshold,
            pg_round,
            pg_step,
            pg_newcity,
        ],
        width="17rem",
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

                Census maps of cities like New York or Chicago show segregation
                street by street. Does that mean residents are strongly
                intolerant?

                In 1971, the economist Thomas Schelling showed that the answer
                can be no. His whole model fits in two lines:
                """
            ),
            mo.Html(
                """
                <div class="sch-callout">
                Two kinds of agents live on a grid of houses; some houses are
                empty.
                <ol style="margin:0.3rem 0 0.1rem;">
                <li>An agent is <b>happy</b> if at least a share <i>t</i> of
                its neighbours are similar to it. Higher <i>t</i> = more
                demanding (less tolerant) agents.</li>
                <li>An unhappy agent <b>moves to a random empty house</b>.</li>
                </ol>
                Repeat until everyone is happy. No prices, no history, no
                institutions.
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
    AMBER,
    CITY_CMAP,
    EDGE_COLOR,
    HAPPY_GREEN,
    UNHAPPY_RED,
    VIOLET,
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
        _pos: (VIOLET if _i < _k else AMBER)
        for _i, _pos in enumerate(_nb_positions)
    }
    for _r in range(5):
        for _c in range(5):
            if (_r, _c) == (2, 2):
                _color, _size, _edge = VIOLET, 320, "#333333"
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

                Think of the city as a grid of houses — or, equivalently, as a
                network where each house is connected to the (up to) 8 houses
                around it. Both panels show the same situation: the violet
                agent in the centre and its 8 neighbours.

                The agent computes its **similarity ratio** = similar
                neighbours ÷ occupied neighbours, and is happy if that ratio
                reaches its **similarity threshold *t***. Higher *t* means a
                more demanding, less tolerant agent.

                **Try it:** keep *t* = 0.30 and move **similar neighbours**
                from 2 to 3 — the verdict flips. Note how undemanding the rule
                is: at *t* = 0.30, an agent is happy even as a clear minority
                (3 of 8 similar).
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
def pg_sim(
    PG_ROUNDS, get_city_seed, pg_empty, pg_size, pg_threshold, run_schelling
):
    pg_history = run_schelling(
        int(pg_size.value),
        float(pg_empty.value),
        float(pg_threshold.value),
        PG_ROUNDS,
        int(get_city_seed()),
    )
    return (pg_history,)


@app.cell
def pg_view(
    HAPPY_GREEN,
    PG_ROUNDS,
    VIOLET,
    assortativity,
    draw_city,
    mo,
    pg_history,
    pg_round,
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
        range(len(_sims)), _sims, color=VIOLET, linewidth=2.2,
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

                ## 2. Now a whole city

                Houses are split between violet and amber agents placed at
                random, with some left empty. Each round, every unhappy agent
                moves to a random empty house. Use the **controls in the
                sidebar**: drag **Round** (or click **+1 round**) to step
                through the simulation, and **New random city** to start over.

                Things to try:

                1. With *t* = 0.30, step through the rounds. Clusters appear
                   within ~15 rounds and average similarity climbs from ≈ 50%
                   to ≈ 75% — much more segregation than anyone asked for.
                2. Set *t* = 0.05: almost nobody is ever unhappy, and the city
                   stays mixed.
                3. Vacancy changes the speed, not the destination. At
                   *t* = 0.30, empty share 0.05 and 0.40 end at about the same
                   similarity, but with few empty houses it takes several
                   times longer to settle.
                4. Set *t* = 0.75 with empty share 0.10: endless churn, half
                   the city unhappy. Raise the empty share to 0.40 and it
                   settles at ≈ 100% — perfect enclaves padded by empty space.
                """
            ),
            _fig,
            mo.md(
                f"Round {_t}: mean similarity **{_sims[_t]:.0%}**, "
                f"**{_haps[_t]:.0%}** happy. End of run: similarity "
                f"**{_sims[-1]:.0%}**, assortativity **{_r_final:.2f}** · "
                f"{_status}."
            ),
            mo.Html(
                """
                <div class="sch-callout">
                <b>The point:</b> no agent wants a segregated city — each is
                happy as a 30% minority. Mild preferences plus movement still
                produce strong segregation. The pattern belongs to the system,
                not to any individual.
                </div>
                """
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
    HAPPY_GREEN,
    UNHAPPY_RED,
    VIOLET,
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
        color=VIOLET, alpha=0.18, linewidth=0,
    )
    _ax_a.plot(sweep_levels, _a_mean, color=VIOLET, linewidth=2.4, marker="o",
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

                ## 3. How much demand makes how much segregation?

                Each point runs the model to the end at that threshold, for
                three random cities (the band shows the min–max). Segregation
                is measured as **assortativity**: 0 means the colours mix at
                random, 1 means complete segregation.

                Two things to notice:

                1. **The curve is steep early.** Between *t* = 0.20 and
                   *t* = 0.40 — agents who are still fine as minorities —
                   assortativity jumps from ≈ 0.15 to ≈ 0.6. By *t* = 0.70
                   segregation is near-total.
                2. **Then it collapses.** Past *t* ≈ 0.7 (red band), demands
                   are so high that agents can almost never all be happy: the
                   city keeps churning and ends up looking mixed. Stable
                   segregation comes from moderate, satisfiable preferences —
                   not extreme ones.
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
