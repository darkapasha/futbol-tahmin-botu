"""Takim profilleri: sezonluk, lig-sezon ortalamasi ve guncel (agirlikli + lige dogru buzulmus)."""
import numpy as np
import pandas as pd

from .features import FEATURES, aggregate, recency_weights

# Bileşik stil endeksleri: lig-sezon icinde z-skor ortalamasi (isaret: +1 / -1).
# Bir endeksin bilesenlerinden hangisi doluysa o kullanilir (kaynak lige gore degisir).
STYLE = {
    "press_idx": [("ppda", -1), ("ppda_proxy", -1)],
    "control_idx": [("possession", 1), ("opp_half_share", 1), ("deep_pg", 1), ("deep_allowed_pg", -1)],
    "directness_idx": [("long_ball_share", 1), ("counter_goal_share", 1), ("fastbreak_xg_share", 1)],
    "width_style_idx": [("width_idx", 1), ("cross_goal_share", 1), ("corners_pg", 1), ("crosses_pg", 1)],
    "setpiece_threat_idx": [("setpiece_goal_share", 1), ("setpiece_xg_share", 1), ("header_goal_share", 1)],
    "setpiece_weak_idx": [("setpiece_ga_share", 1), ("header_ga_share", 1)],
    "attack_idx": [("xg_pg", 1), ("gf_pg", 1), ("shots_pg", 1)],
    "defence_idx": [("xga_pg", -1), ("ga_pg", -1), ("shots_a_pg", -1)],
    "shot_quality_idx": [("xg_per_shot", 1), ("xga_per_shot", -1)],
    "lead_protect_idx": [("ppg_scored_first", 1), ("ga90_lead", -1)],
    "comeback_idx": [("ppg_conceded_first", 1), ("gf90_trail", 1)],
}

SHRINK_K = 8  # guncel profilde lig ortalamasina dogru 8 maclik sahte gozlem
NUM = list(FEATURES)


def home_league(t):
    dom = t[t.league != "UCL"]
    return dom.groupby(["team_key", "season"]).league.agg(lambda s: s.mode().iat[0])


def season_profiles(t):
    lg = home_league(t)
    rows = []
    for (team, season), g in t.groupby(["team_key", "season"]):
        p = aggregate(g)
        p["team_key"], p["season"] = team, season
        p["league"] = lg.get((team, season))
        p["ucl_matches"] = int((g.league == "UCL").sum())
        p["coach_last"] = g.coach.dropna().iloc[-1] if g.coach.notna().any() else None
        rows.append(p)
    s = pd.DataFrame(rows).dropna(subset=["league"])
    return s


def league_profiles(t):
    rows = []
    for (league, season), g in t[t.league != "UCL"].groupby(["league", "season"]):
        p = aggregate(g)
        p["league"], p["season"] = league, season
        rows.append(p)
    return pd.DataFrame(rows)


def ref_seasons(seasons, min_matches=20):
    """Her lig icin karsilastirma sezonu: takimlarin cogunun >= min_matches maci olan son sezon.
    Yeni baslamis sezon (birkac mac, olay verisi yok) z-skor icin guvenilir degil."""
    ok = seasons[seasons.matches >= min_matches].groupby(["league", "season"]).size()
    return ok[ok >= 10].reset_index().groupby("league").season.max().to_dict()


def add_style(df, ref, season_col="season"):
    """df'deki takimlara, ref (lig-sezon takim profilleri) dagilimina gore z-skor endeksleri.
    season_col: df'de hangi sezonun dagilimiyla karsilastirilacagi."""
    df = df.copy()
    stats = ref.groupby(["league", "season"])[NUM].agg(["mean", "std"])
    key = pd.MultiIndex.from_arrays([df.league, df[season_col]])
    for idx, comps in STYLE.items():
        zs = []
        for f, sign in comps:
            mu = key.map(stats[(f, "mean")].to_dict().get)
            sd = key.map(stats[(f, "std")].to_dict().get)
            mu = pd.Series(mu, index=df.index, dtype=float)
            sd = pd.Series(sd, index=df.index, dtype=float).replace(0, np.nan)
            zs.append(sign * (df[f] - mu) / sd)
        df[idx] = pd.concat(zs, axis=1).mean(axis=1, skipna=True)
    return df


def current_profiles(t, league_prof, ref, n_last=40, half_life=15):
    """Her takimin son n_last maci, yakin maclar agir; lig ortalamasina buzulmus.
    ref: lig -> karsilastirma sezonu (ref_seasons); buzulme o sezonun ortalamasina yapilir."""
    t = t.sort_values("date")
    lg_now = home_league(t).reset_index().sort_values("season").groupby("team_key").last()
    lp = league_prof.set_index(["league", "season"])
    rows = []
    for team, g in t.groupby("team_key"):
        g = g.tail(n_last)
        w = recency_weights(g.date, half_life)
        p = aggregate(g, w)
        if team not in lg_now.index:
            continue
        league, season = lg_now.loc[team, "league"], lg_now.loc[team, "season"]
        rs = ref.get(league, season)
        base = lp.loc[(league, rs)] if (league, rs) in lp.index else None
        if base is not None:
            for f in NUM:
                if pd.notna(p[f]) and pd.notna(base[f]):
                    p[f] = (p[f] * p["weight"] + base[f] * SHRINK_K) / (p["weight"] + SHRINK_K)
        p["team_key"], p["league"], p["season"], p["ref_season"] = team, league, season, rs
        p["last_match"] = g.date.iloc[-1]
        p["coach_last"] = g.coach.dropna().iloc[-1] if g.coach.notna().any() else None
        rows.append(p)
    return pd.DataFrame(rows)
