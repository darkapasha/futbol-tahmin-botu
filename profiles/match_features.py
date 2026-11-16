"""Mac x takim ozellik tablosu (team_match) ve oyuncu degisikligi tablosu (subs).

Omurga Transfermarkt maclaridir (7 lig, hoca, dizilis, olaylar). Transfermarkt verisi
bittikten sonraki maclar xgabora ve FotMob'dan eklenir (olay/hoca bilgisi olmadan).
Diger kaynaklar ayni maca tarih + ortak team_key ile baglanir:
  xgabora   -> sut, isabetli sut, korner, faul, kart, bahis oranlari (ulke ligleri)
  Understat -> xG, npxG, PPDA, deep (ilk 5 lig)
  FotMob    -> topla oynama, pas, uzun top, orta, sut durumu (Super Lig + SL, zamanla tumu)
"""
import numpy as np
import pandas as pd

from .common import (CENTRAL, FULLBACK, LINE, WIDE_ATT, goal_kind, parse_formation,
                     src_con)


def _season(d):
    d = pd.to_datetime(d)
    return np.where(d.dt.month >= 7, d.dt.year, d.dt.year - 1)


def _keymap(con, source):
    m = pd.read_sql("SELECT name, team_key FROM team_map WHERE source = ?", con, params=(source,))
    return dict(zip(m.name, m.team_key))


# ---------------------------------------------------------------- omurga
def backbone(con):
    g = pd.read_sql(
        "SELECT game_id, league, season, date, home_club_id, away_club_id, home_club_goals hg, "
        "away_club_goals ag, home_club_manager_name home_coach, away_club_manager_name away_coach, "
        "home_club_formation home_formation, away_club_formation away_formation FROM tm_games", con)
    g["game_id"] = g.game_id.astype(str)
    g["home_key"] = "tm:" + g.home_club_id.astype("Int64").astype(str)
    g["away_key"] = "tm:" + g.away_club_id.astype("Int64").astype(str)
    g["source"] = "tm"
    tm_end = g.date.max()

    # Transfermarkt sonrasi: xgabora (Eylul 2026'ya kadar)
    km = _keymap(con, "xgabora")
    x = pd.read_sql("SELECT league, matchdate date, hometeam, awayteam, fthome hg, ftaway ag "
                    "FROM xg_matches WHERE matchdate > ?", con, params=(tm_end,))
    x["home_key"], x["away_key"] = x.hometeam.map(km), x.awayteam.map(km)
    x = x.dropna(subset=["home_key", "away_key", "hg"])
    x["game_id"] = "xg:" + x.date + ":" + x.home_key + ":" + x.away_key
    x["source"] = "xgabora"

    # FotMob: ikisinden de sonraki maclar (sabah isi her gun ekliyor)
    kf = _keymap(con, "fotmob")
    f = pd.read_sql("SELECT match_id, league, substr(utc_time,1,10) date, home_id, away_id, "
                    "home_goals hg, away_goals ag, home_coach, away_coach, home_formation, "
                    "away_formation FROM fm_matches WHERE home_goals IS NOT NULL", con)
    f["home_key"] = f.home_id.astype(str).map(kf)
    f["away_key"] = f.away_id.astype(str).map(kf)
    f = f.dropna(subset=["home_key", "away_key"])
    last = pd.concat([g[["league", "date"]], x[["league", "date"]]]).groupby("league").date.max()
    f = f[f.date > f.league.map(last).fillna("0")]
    f["game_id"] = "fm:" + f.match_id.astype(str)
    f["source"] = "fotmob"

    cols = ["game_id", "source", "league", "date", "home_key", "away_key", "hg", "ag",
            "home_coach", "away_coach", "home_formation", "away_formation"]
    allg = pd.concat([g.reindex(columns=cols), x.reindex(columns=cols), f.reindex(columns=cols)])
    allg["season"] = _season(allg.date)
    return allg.sort_values(["date", "game_id"]).reset_index(drop=True)


def to_long(g):
    """Mac basina iki satir: takimin kendi bakis acisiyla."""
    def side(h):
        o = "away" if h else "home"
        s = "home" if h else "away"
        return pd.DataFrame({
            "game_id": g.game_id, "source": g.source, "league": g.league, "season": g.season,
            "date": g.date, "is_home": int(h),
            "team_key": g[f"{s}_key"], "opp_key": g[f"{o}_key"],
            "gf": g.hg if h else g.ag, "ga": g.ag if h else g.hg,
            "coach": g[f"{s}_coach"], "opp_coach": g[f"{o}_coach"],
            "formation_raw": g[f"{s}_formation"], "opp_formation_raw": g[f"{o}_formation"],
        })
    t = pd.concat([side(True), side(False)], ignore_index=True)
    t["pts"] = np.select([t.gf > t.ga, t.gf == t.ga], [3, 1], 0)
    for p in ("", "opp_"):
        parsed = t[f"{p}formation_raw"].map(parse_formation)
        t[f"{p}formation"] = parsed.str[3]
        t[f"{p}n_def"] = parsed.str[0]
        t[f"{p}n_fwd"] = parsed.str[2]
    t["back3"] = (t.n_def == 3).astype(float).where(t.n_def.notna())
    t["back5"] = (t.n_def == 5).astype(float).where(t.n_def.notna())
    return t


# ---------------------------------------------------------------- olaylar (TM)
def _state(gd):
    return np.where(gd > 0, "lead", np.where(gd < 0, "trail", "level"))


def game_state_features(con):
    """Gol zaman cizelgesinden: durumda gecen dakika, durum basina atilan/yenilen gol,
    ilk golu kim atti, ilk yari skoru, gol turleri."""
    g = pd.read_sql("SELECT game_id, home_club_id h, away_club_id a FROM tm_games", con)
    e = pd.read_sql("SELECT game_id, minute, club_id, description FROM tm_events "
                    "WHERE type = 'Goals'", con)
    e = e.merge(g, on="game_id")
    kinds = pd.DataFrame(e.description.map(goal_kind).tolist(), index=e.index)
    e = pd.concat([e, kinds], axis=1)
    e["minute"] = e.minute.clip(upper=90)
    e = e.sort_values(["game_id", "minute"])

    rows = []
    goals_by_game = {k: v for k, v in e.groupby("game_id")}
    for gid, h, a in g.itertuples(index=False):
        ev = goals_by_game.get(gid)
        for team, opp in ((h, a), (a, h)):
            r = {"game_id": str(gid), "team_key": f"tm:{team}"}
            mins = {"lead": 0.0, "level": 0.0, "trail": 0.0}
            gf_s = {"lead": 0, "level": 0, "trail": 0}
            ga_s = {"lead": 0, "level": 0, "trail": 0}
            kind_f = dict.fromkeys(("pen", "header", "counter", "setpiece", "cross", "longshot"), 0)
            kind_a = dict(kind_f)
            gd, t0, first, ht_gf, ht_ga = 0, 0, None, 0, 0
            if ev is not None:
                for row in ev.itertuples(index=False):
                    st = "lead" if gd > 0 else "trail" if gd < 0 else "level"
                    mins[st] += max(row.minute - t0, 0)
                    t0 = row.minute
                    mine = row.club_id == team
                    (gf_s if mine else ga_s)[st] += 1
                    kd = kind_f if mine else kind_a
                    for k in kd:
                        kd[k] += bool(getattr(row, k))
                    if first is None:
                        first = int(mine)
                    if row.minute <= 45:
                        ht_gf += mine
                        ht_ga += not mine
                    gd += 1 if mine else -1
            st = "lead" if gd > 0 else "trail" if gd < 0 else "level"
            mins[st] += max(90 - t0, 0)
            for s in mins:
                r[f"min_{s}"] = mins[s]
                r[f"gf_{s}"] = gf_s[s]
                r[f"ga_{s}"] = ga_s[s]
            for k in kind_f:
                r[f"gf_{k}"] = kind_f[k]
                r[f"ga_{k}"] = kind_a[k]
            r["scored_first"] = first
            r["ht_gf"], r["ht_ga"] = ht_gf, ht_ga
            rows.append(r)
    return pd.DataFrame(rows)


def subs_table(con):
    """Her oyuncu degisikligi: dakika, sebep, o andaki skor durumu, cikan/giren oyuncunun hatti."""
    g = pd.read_sql("SELECT game_id, season, league, home_club_id h, away_club_id a, "
                    "home_club_manager_name hc, away_club_manager_name ac FROM tm_games", con)
    s = pd.read_sql("SELECT game_id, minute, club_id, player_id, player_in_id, description "
                    "FROM tm_events WHERE type = 'Substitutions'", con)
    goals = pd.read_sql("SELECT game_id, minute, club_id FROM tm_events WHERE type='Goals'", con)
    pos = pd.read_sql("SELECT player_id, position FROM tm_players", con)
    line = dict(zip(pos.player_id, pos.position.map(LINE)))
    s = s.merge(g, on="game_id")
    s["coach"] = np.where(s.club_id == s.h, s.hc, s.ac)
    s["reason"] = s.description.fillna("").str.strip(", ").replace("", "Unknown")
    s["out_line"] = s.player_id.map(line)
    s["in_line"] = s.player_in_id.map(line)
    s["delta"] = s.in_line - s.out_line  # >0 daha hucumcu, <0 daha savunmaci

    # O dakikadaki averaj (degisiklikten once atilan goller; ayni dakika belirsiz, dahil etme)
    m = s[["game_id", "club_id", "minute"]].reset_index().merge(goals, on="game_id",
                                                                suffixes=("", "_g"), how="left")
    m = m[m.minute_g < m.minute]
    m["d"] = np.where(m.club_id_g == m.club_id, 1, -1)
    s["gd"] = m.groupby("index").d.sum().reindex(s.index).fillna(0).astype(int)
    s["state"] = _state(s.gd)
    s["team_key"] = "tm:" + s.club_id.astype(str)
    s["game_id"] = s.game_id.astype(str)
    return s[["game_id", "league", "season", "team_key", "coach", "minute", "reason",
              "out_line", "in_line", "delta", "gd", "state"]]


def sub_features(subs):
    s = subs.sort_values("minute")
    grp = s.groupby(["game_id", "team_key"])
    f = pd.DataFrame({
        "n_subs": grp.size(),
        "first_sub_min": grp.minute.min(),
        "subs_by_60": grp.minute.apply(lambda m: (m <= 60).sum()),
        "first_sub_state": grp.state.first(),
        "sub_att_delta": grp.delta.sum(min_count=1),
        "injury_subs": grp.reason.apply(lambda r: (r == "Injury").sum()),
    })
    return f.reset_index()


def lineup_features(con):
    l = pd.read_sql("SELECT game_id, club_id, position FROM tm_lineups "
                    "WHERE type = 'starting_lineup'", con)
    l["wide_att"] = l.position.isin(WIDE_ATT)
    l["fullback"] = l.position.isin(FULLBACK)
    l["central"] = l.position.isin(CENTRAL)
    l["cf"] = l.position.isin({"Centre-Forward", "Second Striker"})
    f = l.groupby(["game_id", "club_id"])[["wide_att", "fullback", "central", "cf"]].sum()
    known = f.sum(axis=1)
    f["width_idx"] = (f.wide_att + f.fullback) / known.where(known >= 8)
    f = f.reset_index().rename(columns={"wide_att": "n_wide_att", "fullback": "n_fullback",
                                        "central": "n_central", "cf": "n_cf"})
    f["game_id"] = f.game_id.astype(str)
    f["team_key"] = "tm:" + f.club_id.astype(str)
    return f.drop(columns="club_id")


# ---------------------------------------------------------------- diger kaynaklar
def xgabora_features(con, games):
    km = _keymap(con, "xgabora")
    x = pd.read_sql("SELECT league, matchdate date, hometeam, awayteam, homeshots, awayshots, "
                    "hometarget, awaytarget, homecorners, awaycorners, homefouls, awayfouls, "
                    "homeyellow, awayyellow, homered, awayred, oddhome, odddraw, oddaway "
                    "FROM xg_matches", con)
    x["home_key"], x["away_key"] = x.hometeam.map(km), x.awayteam.map(km)
    inv = 1 / x[["oddhome", "odddraw", "oddaway"]]
    p = inv.div(inv.sum(axis=1), axis=0)
    x["p_home"], x["p_draw"], x["p_away"] = p.oddhome, p.odddraw, p.oddaway
    m = games[["game_id", "league", "date", "home_key", "away_key"]].merge(
        x, on=["league", "date", "home_key", "away_key"])
    out = []
    for h in (True, False):
        s, o = ("home", "away") if h else ("away", "home")
        out.append(pd.DataFrame({
            "game_id": m.game_id, "team_key": m[f"{s}_key"],
            "shots": m[f"{s}shots"], "shots_a": m[f"{o}shots"],
            "sot": m[f"{s}target"], "sot_a": m[f"{o}target"],
            "corners": m[f"{s}corners"], "corners_a": m[f"{o}corners"],
            "fouls": m[f"{s}fouls"], "fouls_a": m[f"{o}fouls"],
            "yellow": m[f"{s}yellow"], "red": m[f"{s}red"],
            "mkt_win": m[f"p_{s}"], "mkt_draw": m.p_draw, "mkt_loss": m[f"p_{o}"],
        }))
    return pd.concat(out, ignore_index=True)


def understat_features(con, long):
    km = _keymap(con, "understat")
    u = pd.read_sql("SELECT team, substr(date,1,10) date, xg, xga, npxg, npxga, ppda_att, "
                    "ppda_def, oppda_att, oppda_def, deep, deep_allowed FROM us_team_games", con)
    u["team_key"] = u.team.map(km)
    u["ppda"] = u.ppda_att / u.ppda_def.replace(0, np.nan)    # dusuk = yogun pres
    u["oppda"] = u.oppda_att / u.oppda_def.replace(0, np.nan)  # dusuk = rakip bize pres yapiyor
    u = u.dropna(subset=["team_key"])
    cols = ["xg", "xga", "npxg", "npxga", "ppda", "oppda", "ppda_att", "ppda_def",
            "oppda_att", "oppda_def", "deep", "deep_allowed"]
    keys = long[["game_id", "team_key", "date"]]
    # Saat dilimi farki icin once ayni gun, sonra +-1 gun
    res = []
    left = keys
    for shift in (0, -1, 1):
        uu = u.assign(date=(pd.to_datetime(u.date) + pd.Timedelta(days=shift)).dt.strftime("%Y-%m-%d"))
        m = left.merge(uu[["team_key", "date"] + cols], on=["team_key", "date"])
        res.append(m)
        left = left[~left.set_index(["game_id", "team_key"]).index.isin(
            m.set_index(["game_id", "team_key"]).index)]
    return pd.concat(res).drop(columns="date").drop_duplicates(["game_id", "team_key"])


def _num(v):
    try:
        return float(str(v).split()[0])
    except (ValueError, IndexError):
        return np.nan


def fotmob_features(con, long):
    kf = _keymap(con, "fotmob")
    fm = pd.read_sql("SELECT match_id, substr(utc_time,1,10) date, home_id, away_id, home_xg, "
                     "away_xg FROM fm_matches WHERE home_goals IS NOT NULL", con)
    st = pd.read_sql("SELECT match_id, key, home, away FROM fm_stats WHERE period = 'All'", con)
    keys = {"BallPossesion": "poss", "passes": "passes", "accurate_passes": "pass_acc_n",
            "opposition_half_passes": "opp_half_passes", "long_balls_accurate": "long_balls",
            "accurate_crosses": "crosses", "touches_opp_box": "box_touches",
            "shots_inside_box": "shots_in_box", "total_shots": "fm_shots",
            "big_chance": "big_chances", "interceptions": "interceptions",
            "matchstats.headers.tackles": "tackles", "expected_goals_open_play": "xg_open",
            "expected_goals_non_penalty": "fm_npxg", "own_half_passes": "own_half_passes",
            "fouls": "fm_fouls"}
    st = st[st.key.isin(keys)]
    st["key"] = st.key.map(keys)
    st["home"], st["away"] = st.home.map(_num), st.away.map(_num)
    wide = st.pivot_table(index="match_id", columns="key", values=["home", "away"])
    sh = pd.read_sql("SELECT match_id, team_id, xg, situation FROM fm_shots", con)
    sh["sp"] = sh.situation.isin(["FromCorner", "SetPiece", "FreeKick", "ThrowInSetPiece"])
    sh["fb"] = sh.situation == "FastBreak"
    shot_agg = sh.groupby(["match_id", "team_id"]).agg(
        xg_setpiece=("xg", lambda v: v[sh.loc[v.index, "sp"]].sum()),
        xg_fastbreak=("xg", lambda v: v[sh.loc[v.index, "fb"]].sum()),
        n_shots_fm=("xg", "size")).reset_index()

    out = []
    for h in (True, False):
        s, o = ("home", "away") if h else ("away", "home")
        d = pd.DataFrame({"match_id": fm.match_id, "date": fm.date,
                          "fm_team": fm[f"{s}_id"], "fm_xg": fm[f"{s}_xg"], "fm_xga": fm[f"{o}_xg"]})
        for k in keys.values():
            if (s, k) in wide.columns:
                d[k] = d.match_id.map(wide[(s, k)])
            if (o, k) in wide.columns and k in ("passes", "fm_shots", "own_half_passes"):
                d[k + "_a"] = d.match_id.map(wide[(o, k)])
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    d = d.merge(shot_agg, left_on=["match_id", "fm_team"], right_on=["match_id", "team_id"], how="left")
    d["team_key"] = d.fm_team.astype(str).map(kf)
    d = d.dropna(subset=["team_key"])
    d["opp_half_share"] = d.opp_half_passes / d.passes
    d["long_ball_share"] = d.long_balls / d.passes
    # Pres vekili (PPDA benzeri): rakibin kendi yarisindaki paslari / bizim savunma hamlelerimiz
    d["ppda_proxy"] = d.own_half_passes_a / (d.tackles + d.interceptions + d.fm_fouls).replace(0, np.nan)
    m = long[["game_id", "team_key", "date"]].merge(d, on=["team_key", "date"])
    drop = ["match_id", "date", "fm_team", "team_id", "passes_a", "own_half_passes_a", "fm_fouls"]
    return m.drop(columns=[c for c in drop if c in m.columns]).drop_duplicates(["game_id", "team_key"])


# ---------------------------------------------------------------- birlestir
def build(con=None):
    con = con or src_con()
    games = backbone(con)
    long = to_long(games)
    parts = [game_state_features(con), sub_features(subs := subs_table(con)),
             lineup_features(con), xgabora_features(con, games),
             understat_features(con, long), fotmob_features(con, long)]
    for p in parts:
        long = long.merge(p, on=["game_id", "team_key"], how="left")
    return long, subs
