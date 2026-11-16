"""Gecmis maclar: mac basina tek satir (ev sahibi bakisi) + dakika dakika gol/kirmizi kart olaylari."""
import numpy as np
import pandas as pd

from .common import TeamResolver, prof_con, src_con


def games(resolver=None):
    con = prof_con()
    g = pd.read_sql(
        "SELECT game_id, source, league, season, date, team_key home, opp_key away, gf hg, ga ag, "
        "xg_any hxg, xga_any axg, mkt_win p_h, mkt_draw p_d, mkt_loss p_a, coach home_coach, "
        "opp_coach away_coach, opp_class home_opp_class, shots_any hs, shots_a_any as_ "
        "FROM team_match WHERE is_home = 1", con)
    # FotMob'dan gelen ve team_map'te eslesmemis takimlari adla bagla
    if g.home.str.startswith("fm:").any() or g.away.str.startswith("fm:").any():
        r = resolver or TeamResolver()
        fm = pd.read_sql("SELECT home_id id, home name, league FROM fm_matches UNION "
                         "SELECT away_id, away, league FROM fm_matches", src_con())
        fix = {f"fm:{i}": r.fotmob(i, n, lg) for i, n, lg in fm.itertuples(index=False)}
        g["home"] = g.home.map(lambda k: fix.get(k, k))
        g["away"] = g.away.map(lambda k: fix.get(k, k))
    g = g.dropna(subset=["hg", "ag"]).sort_values(["date", "game_id"]).reset_index(drop=True)
    g["date"] = pd.to_datetime(g.date)
    return g


def events():
    """Transfermarkt olaylari: gol ve kirmizi kart dakikalari (takim = tm:<club_id>)."""
    con = src_con()
    e = pd.read_sql(
        "SELECT e.game_id, e.minute, e.type, e.club_id, e.description, g.home_club_id h, g.away_club_id a "
        "FROM tm_events e JOIN tm_games g USING (game_id) WHERE e.type IN ('Goals','Cards')", con)
    e = e[(e.type == "Goals") | e.description.str.contains("Red card|Second yellow", case=False, na=False)]
    e["kind"] = np.where(e.type == "Goals", "goal", "red")
    e["is_home"] = (e.club_id == e.h).astype(int)
    e["pen"] = e.description.str.contains(", Penalty", na=False) & (e.kind == "goal")
    e["game_id"] = e.game_id.astype(str)
    return e[["game_id", "minute", "kind", "is_home", "pen"]]
