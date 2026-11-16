"""Hoca taktik parmak izi.

coach_profile : kariyer boyu stil (lig ortalamasina gore fark), dizilis tercihi, rakibe gore uyum ozeti
coach_split   : ayni ozellikler rakip gucune (guclu / benzer / zayif) ve ic saha / deplasmana gore
coach_subs    : skor durumuna gore degisiklik aliskanliklari (dakika, hucumcu/savunmaci degisiklik)
"""
import numpy as np
import pandas as pd

from .features import FEATURES, aggregate

MIN_MATCHES = 10
KEY = ["ppg", "xg_pg", "xga_pg", "shots_pg", "shots_a_pg", "ppda", "ppda_proxy", "possession",
       "deep_pg", "long_ball_share", "width_idx", "back3_share", "back5_share", "strikers_pg",
       "setpiece_goal_share", "counter_goal_share", "first_sub_min", "share_leading", "elo_diff_avg"]


def _baseline(g, league_prof):
    """Hocanin maclarinin lig-sezon dagilimina gore agirlikli lig ortalamasi."""
    lp = league_prof.set_index(["league", "season"])
    w = g[g.league != "UCL"].groupby(["league", "season"]).size()
    w = w[w.index.isin(lp.index)]
    if w.empty:
        return None
    return lp.loc[w.index, list(FEATURES)].mul(w.values, axis=0).sum() / w.sum()


def coach_profiles(t, league_prof):
    t = t[t.coach.notna()].sort_values("date")
    rows, splits = [], []
    for coach, g in t.groupby("coach"):
        if len(g) < MIN_MATCHES:
            continue
        p = aggregate(g)
        p["coach"] = coach
        p["clubs"] = ", ".join(g.team_key.drop_duplicates().tolist()[-6:])
        p["first_match"], p["last_match"] = g.date.iloc[0], g.date.iloc[-1]
        p["current_team"] = g.team_key.iloc[-1]
        base = _baseline(g, league_prof)
        if base is not None:
            for f in KEY:
                p[f"rel_{f}"] = p[f] - base[f] if pd.notna(p[f]) and pd.notna(base[f]) else np.nan
        main = p.get("main_formation")

        # Rakip gucu ve saha bolmeleri
        parts = {}
        for col, val in [("opp_class", "stronger"), ("opp_class", "similar"), ("opp_class", "weaker"),
                         ("is_home", 1), ("is_home", 0)]:
            sub = g[g[col] == val]
            if len(sub) < 3:
                continue
            s = aggregate(sub)
            s["coach"], s["split"] = coach, f"{col}={val}"
            s["main_formation_used"] = (sub.formation == main).mean() if main else np.nan
            parts[f"{col}={val}"] = s
            splits.append(s)

        # Uyum ozeti: guclu rakibe karsi zayif rakibe gore ne degisiyor
        st, wk = parts.get("opp_class=stronger"), parts.get("opp_class=weaker")
        if st is not None and wk is not None:
            p["adapt_n_stronger"], p["adapt_n_weaker"] = st["matches"], wk["matches"]
            p["adapt_formation_change"] = wk["main_formation_used"] - st["main_formation_used"]
            for f in ("back3_share", "back5_share", "strikers_pg", "ppda", "ppda_proxy",
                      "possession", "xg_pg", "xga_pg", "shots_pg", "long_ball_share",
                      "width_idx", "first_sub_min"):
                p[f"adapt_{f}"] = st[f] - wk[f]
        rows.append(p)
    return pd.DataFrame(rows), pd.DataFrame(splits)


def coach_subs(subs):
    s = subs[subs.coach.notna()].copy()
    s["att_sub"] = (s.delta > 0).astype(float).where(s.delta.notna())
    s["def_sub"] = (s.delta < 0).astype(float).where(s.delta.notna())
    s["injury"] = (s.reason == "Injury").astype(float)
    g = s.groupby(["coach", "state"]).agg(
        subs=("minute", "size"), mean_minute=("minute", "mean"),
        att_share=("att_sub", "mean"), def_share=("def_sub", "mean"),
        mean_delta=("delta", "mean"), injury_share=("injury", "mean")).reset_index()
    games = s.groupby("coach").game_id.nunique()
    g["subs_per_game_in_state"] = g.subs / g.coach.map(games)
    # Tek satirlik ozet: geride iken ne kadar hucumcu, onde iken ne kadar savunmaci
    w = g.pivot(index="coach", columns="state", values=["mean_delta", "mean_minute", "att_share", "def_share"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w["games"] = games
    w["reactivity"] = w.get("mean_delta_trail") - w.get("mean_delta_lead")
    return g, w.reset_index()
