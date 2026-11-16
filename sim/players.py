"""Oyuncular: muhtemel ilk 11, eksikler, kadro degeri carpani ve gol payi (golcu tahmini).

Muhtemel 11 (oncelik sirasiyla):
  1. FotMob'un o mac icin acikladigi kadro (mac saatine ~1 saat kala)
  2. Takimin FotMob'daki son ilk 11'i, eksikler (FotMob 'unavailable' + Transfermarkt sakatlik) cikarilip
     ayni hattan en degerli yedekle doldurularak
  3. Transfermarkt'taki son ilk 11 (ayni duzeltmeyle)

Kadro carpani: (ilk 11 degeri / takimin son 10 macta sahaya surdugu ortalama 11 degeri) ^ beta.
Yildiz eksikse hucum ve savunma ayni oranda zayiflar. beta kalibrasyonda gecmis maclardan kestirilir.

Gol payi: oyuncunun son 2 sezonda 90 dakikada attigi penaltisiz gol, mevkisine gore on degerle
(900 dakikalik sahte gozlem) buzulmus. Penaltilari takimin penalti atani kullanir.
"""
import numpy as np
import pandas as pd

from .common import norm, src_con

# mevki -> 90 dakikada penaltisiz gol on degeri ve hat (0 kaleci, 1 defans, 2 orta, 3 hucum)
POS = {
    "Goalkeeper": (0.0, 0), "Centre-Back": (0.04, 1), "Left-Back": (0.03, 1), "Right-Back": (0.03, 1),
    "Defensive Midfield": (0.05, 2), "Central Midfield": (0.09, 2), "Left Midfield": (0.15, 2),
    "Right Midfield": (0.15, 2), "Attacking Midfield": (0.22, 3), "Left Winger": (0.24, 3),
    "Right Winger": (0.24, 3), "Second Striker": (0.32, 3), "Centre-Forward": (0.42, 3),
}
PRIOR_MIN = 1500.0
# ilk 11 tahmininin tutma olasiligi (oyuncu gercekten oynuyor mu)
P_PLAY = {"aciklanan": 1.0, "muhtemel": 0.88, "son11": 0.8}


def fm_line(pos_id):
    if pos_id is None or pd.isna(pos_id):
        return None
    p = int(pos_id)
    if p == 11:
        return 0
    if p < 40:
        return 1
    if p < 80:
        return 2
    return 3


class PlayerBook:
    """Oyuncu gol oranlari ve FotMob -> Transfermarkt oyuncu eslemesi (asof tarihine kadar veriyle)."""

    def __init__(self, asof, con=None):
        self.con = con = con or src_con()
        self.asof = pd.Timestamp(asof)
        start = (self.asof - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
        end = self.asof.strftime("%Y-%m-%d")
        ap = pd.read_sql("SELECT player_id, player_club_id club, date, goals, minutes_played m FROM tm_appearances "
                         "WHERE date >= ? AND date < ?", con, params=(start, end))
        pen = pd.read_sql("SELECT player_id, COUNT(*) pens FROM tm_events WHERE type='Goals' AND "
                          "description LIKE '%, Penalty,%' AND date >= ? AND date < ? GROUP BY 1", con,
                          params=(start, end))
        age = (self.asof - pd.to_datetime(ap.date)).dt.days
        ap["w"] = 0.5 ** (age / 365)
        agg = ap.assign(gw=ap.goals * ap.w, mw=ap.m * ap.w).groupby("player_id").agg(
            g=("gw", "sum"), m=("mw", "sum"), goals=("goals", "sum"), apps=("m", "size"),
            full90=("m", lambda s: (s >= 88).mean()), last_club=("club", "last"))
        pl = pd.read_sql("SELECT player_id, name, sub_position, position, current_club_id, market_value_in_eur value "
                         "FROM tm_players", con).set_index("player_id")
        self.players = pl
        agg = agg.join(pl[["name", "sub_position"]], how="left").join(pen.set_index("player_id"), how="left")
        agg["pens"] = agg.pens.fillna(0)
        prior = agg.sub_position.map(lambda s: POS.get(s, (0.1, 2))[0])
        npg = (agg.g - agg.pens * 0.8).clip(lower=0)
        agg["rate90"] = (npg + prior * PRIOR_MIN / 90) / ((agg.m + PRIOR_MIN) / 90)
        self.rates = agg
        self._name_idx = None

    # -------------------------------------------------- isim eslemesi
    def _names(self):
        if self._name_idx is None:
            p = self.players.reset_index()
            p["n"] = p.name.map(norm)
            p["last"] = p.n.str.split().str[-1]
            self._name_idx = p
        return self._name_idx

    def tm_id(self, name, tm_club=None):
        p = self._names()
        n = norm(name)
        c = p[p.n == n]
        if len(c) > 1 and tm_club is not None:
            c2 = c[c.current_club_id == tm_club]
            c = c2 if len(c2) else c
        if len(c) == 1:
            return int(c.player_id.iat[0])
        if tm_club is not None:
            last = n.split()[-1] if n else ""
            c = p[(p.current_club_id == tm_club) & (p["last"] == last)]
            if len(c) == 1:
                return int(c.player_id.iat[0])
            c = p[p.current_club_id == tm_club]
            if len(c):
                import difflib
                best = difflib.get_close_matches(n, c.n.tolist(), n=1, cutoff=0.8)
                if best:
                    return int(c[c.n == best[0]].player_id.iat[0])
        return None

    # -------------------------------------------------- oyuncu bilgisi
    def info(self, tm_id, fallback_pos=None):
        r = self.rates.loc[tm_id] if tm_id is not None and tm_id in self.rates.index else None
        pos = None
        if tm_id is not None and tm_id in self.players.index:
            pos = self.players.at[tm_id, "sub_position"]
        prior, line = POS.get(pos, (None, fallback_pos))
        if prior is None:
            prior = {0: 0.0, 1: 0.035, 2: 0.09, 3: 0.3}.get(line, 0.1)
        if r is not None:
            rate, pens, full90 = r.rate90, r.pens, r.full90
        else:
            rate, pens, full90 = prior, 0.0, 0.5
        return dict(tm_id=tm_id, pos=pos, line=line if line is not None else 2, rate90=float(rate),
                    pens=float(pens), full90=float(full90) if pd.notna(full90) else 0.5)


def injured_tm_ids(asof, con):
    d = pd.Timestamp(asof).strftime("%Y-%m-%d")
    inj = pd.read_sql("SELECT player_id FROM injuries WHERE from_date <= ? AND (end_date IS NULL OR end_date >= ?)",
                      con, params=(d, d))
    return set(inj.player_id.astype(int))


def _fill(xi, bench, out_ids):
    """Eksik ilk 11 oyuncusunu ayni hattan en degerli yedekle degistir."""
    keep = [p for p in xi if p["key"] not in out_ids]
    bench = [b for b in bench if b["key"] not in out_ids]
    missing = [p for p in xi if p["key"] in out_ids]
    for m in missing:
        cand = [b for b in bench if b["line"] == m["line"]] or [b for b in bench if b["line"] != 0]
        if not cand:
            continue
        best = max(cand, key=lambda b: b.get("value") or 0)
        bench.remove(best)
        keep.append({**best, "role": "starter", "replaces": m["name"]})
    return keep, bench, missing


def lineup_from_fotmob(book, fm_team_id, tm_club, asof, match_id=None, unavailable=()):
    """FotMob kadrosu. match_id verilip o macin kadrosu varsa onu (aciklanmis 11), yoksa son maci kullanir."""
    con = book.con
    asof_s = pd.Timestamp(asof).strftime("%Y-%m-%dT%H:%M")
    q = ("SELECT l.*, m.utc_time FROM fm_lineups l JOIN fm_matches m USING (match_id) WHERE l.team_id = ? "
         "AND m.utc_time < ? ORDER BY m.utc_time DESC")
    lu = pd.read_sql(q, con, params=(fm_team_id, asof_s))
    confirmed = False
    if match_id is not None:
        cur = pd.read_sql("SELECT * FROM fm_lineups WHERE match_id = ? AND team_id = ?", con, params=(match_id, fm_team_id))
        if (cur.role == "starter").sum() >= 11:
            lu, confirmed = cur, True
    if lu.empty:
        return None
    last = lu[lu.match_id == lu.match_id.iat[0]]
    hist_vals = lu[lu.role == "starter"].groupby("match_id").market_value.sum().head(10)
    players = []
    for r in last.itertuples():
        tid = book.tm_id(r.name, tm_club)
        inf = book.info(tid, fm_line(r.position_id))
        if r.role == "starter" and fm_line(r.position_id) is not None:
            inf["line"] = fm_line(r.position_id)
        players.append({**inf, "key": f"fm{r.player_id}", "name": r.name, "role": r.role,
                        "value": float(r.market_value) if pd.notna(r.market_value) else 0.0})
    xi = [p for p in players if p["role"] == "starter"]
    bench = [p for p in players if p["role"] == "sub"]
    out = set()
    if not confirmed:
        inj = injured_tm_ids(asof, con)
        un = {str(u) for u in unavailable}
        out = {p["key"] for p in players if (p["tm_id"] in inj) or (p["key"][2:] in un)}
    xi, bench, missing = _fill(xi, bench, out)
    usual = float(hist_vals.mean()) if len(hist_vals) and hist_vals.mean() > 0 else None
    return dict(source="fotmob" + ("-aciklanan" if confirmed else "-son11"), xi=xi, bench=bench, missing=missing,
                usual_value=usual, date=str(last.utc_time.iat[0])[:10] if "utc_time" in last else "9999")


def lineup_from_tm(book, tm_club, asof):
    con = book.con
    d = pd.Timestamp(asof).strftime("%Y-%m-%d")
    games = pd.read_sql("SELECT DISTINCT game_id, date FROM tm_lineups WHERE club_id = ? AND date < ? "
                        "ORDER BY date DESC LIMIT 10", con, params=(tm_club, d))
    if games.empty:
        return None
    lu = pd.read_sql(f"SELECT * FROM tm_lineups WHERE club_id = ? AND game_id IN ({','.join(map(str, games.game_id))})",
                     con, params=(tm_club,))
    vals = valuations_asof(con, lu.player_id.unique(), asof)
    lu["value"] = lu.player_id.map(vals).fillna(0)
    usual = lu[lu.type == "starting_lineup"].groupby("game_id").value.sum().mean()
    last = lu[lu.game_id == games.game_id.iat[0]]
    players = []
    for r in last.itertuples():
        inf = book.info(int(r.player_id), POS.get(r.position, (None, 2))[1])
        players.append({**inf, "key": f"tm{r.player_id}", "name": r.player_name,
                        "role": "starter" if r.type == "starting_lineup" else "sub", "value": float(r.value)})
    xi = [p for p in players if p["role"] == "starter"]
    bench = [p for p in players if p["role"] == "sub"]
    inj = injured_tm_ids(asof, con)
    xi, bench, missing = _fill(xi, bench, {p["key"] for p in players if p["tm_id"] in inj})
    return dict(source="transfermarkt-son11", xi=xi, bench=bench, missing=missing,
                usual_value=float(usual) if usual and usual > 0 else None, date=str(games.date.iat[0])[:10])


def valuations_asof(con, ids, asof):
    if len(ids) == 0:
        return {}
    d = pd.Timestamp(asof).strftime("%Y-%m-%d")
    v = pd.read_sql(f"SELECT player_id, date, market_value_in_eur v FROM tm_valuations WHERE date < ? AND "
                    f"player_id IN ({','.join(map(str, map(int, ids)))})", con, params=(d,))
    return v.sort_values("date").groupby("player_id").v.last().to_dict()


def strength_ratio(lu, asof=None, max_age_days=45):
    """Kadro eski bir maca aitse (orn. yaz arasindan once) oran anlamsiz: 1 dondur."""
    if not lu or not lu.get("usual_value") or not np.isfinite(lu["usual_value"]):
        return 1.0
    if asof is not None and lu["date"] != "9999":
        if (pd.Timestamp(asof) - pd.Timestamp(lu["date"])).days > max_age_days:
            return 1.0
    v = sum(p.get("value") or 0 for p in lu["xi"] if np.isfinite(p.get("value") or 0))
    if not np.isfinite(v) or v <= 0:
        return 1.0
    return float(np.clip(v / lu["usual_value"], 0.5, 1.6))


def scorer_table(lu):
    """Ilk 11 + yedekler icin gol payi agirliklari ve oyundan cikma egilimi."""
    rows = []
    p_play = next((v for k, v in P_PLAY.items() if lu["source"].endswith(k)), 0.85)
    for p in lu["xi"]:
        # cikma olasiligi: mevkiye ve oyuncunun 90 dk tamamlama oranina gore
        p_off = float(np.clip(1 - p["full90"], 0.05, 0.9)) if p["line"] else 0.02
        rows.append(dict(name=p["name"], start=1, rate=p["rate90"], pens=p["pens"], p_off=p_off, line=p["line"],
                         p_play=p_play))
    bench = sorted(lu["bench"], key=lambda b: (b["line"] != 0, b["rate90"]), reverse=True)
    for b in bench[:9]:
        if b["line"] == 0:
            continue
        rows.append(dict(name=b["name"], start=0, rate=b["rate90"], pens=b["pens"], p_off=0.0, line=b["line"],
                         p_play=1.0))
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    # penalti atani: ilk 11'de en cok penalti golu olan (yoksa en yuksek gol orani)
    xi = t[t.start == 1]
    taker = xi.pens.idxmax() if xi.pens.max() > 0 else xi.rate.idxmax()
    t["pen_taker"] = 0
    t.loc[taker, "pen_taker"] = 1
    return t


def lineup_from_prematch(book, rows, fm_team_id, tm_club, asof, confirmed=False):
    """FotMob mac oncesi sayfasi: muhtemel (ya da aciklanmis) 11 + eksikler.
    Yedek listesi yoksa takimin son FotMob / Transfermarkt macindaki yedekler kullanilir."""
    rows = rows[rows.team_id == fm_team_id]
    st = rows[rows.role == "starter"]
    if len(st) < 11:
        return None
    unav = rows[rows.role == "unavailable"]

    def mk(r, role):
        tid = book.tm_id(r.name, tm_club)
        inf = book.info(tid, fm_line(r.position_id))
        if role == "starter" and fm_line(r.position_id) is not None:
            inf["line"] = fm_line(r.position_id)
        return {**inf, "key": f"fm{r.player_id}", "name": r.name, "role": role, "value": float(r.market_value) if pd.notna(r.market_value) else 0.0}

    xi = [mk(r, "starter") for r in st.itertuples()]
    bench = [mk(r, "sub") for r in rows[rows.role == "sub"].itertuples()]
    if not bench:
        prev = lineup_from_fotmob(book, fm_team_id, tm_club, asof) or (lineup_from_tm(book, tm_club, asof) if tm_club else None)
        if prev:
            names = {p["name"] for p in xi} | set(unav.name)
            bench = [b for b in prev["bench"] + prev["xi"] if b["name"] not in names][:12]
            for b in bench:
                b["role"] = "sub"
    hist_vals = pd.read_sql("SELECT l.match_id, SUM(l.market_value) v FROM fm_lineups l JOIN fm_matches m USING (match_id) "
                            "WHERE l.team_id = ? AND l.role = 'starter' AND m.utc_time < ? GROUP BY 1 "
                            "ORDER BY MAX(m.utc_time) DESC LIMIT 10", book.con,
                            params=(fm_team_id, pd.Timestamp(asof).strftime("%Y-%m-%dT%H:%M")))
    recent = hist_vals.v[hist_vals.v > 0]
    missing = [{"name": r.name, "note": r.note} for r in unav.itertuples()]
    return dict(source="fotmob-" + ("aciklanan" if confirmed else "muhtemel"), xi=xi, bench=bench, missing=missing,
                usual_value=float(recent.mean()) if len(recent) >= 3 else None, date="9999")


# Kanatci x karsisindaki bek (sim.matchups, 22.10.2026): kanatcinin gol atma olasiligi, karsisindaki bekin
# piyasa degeri (log) 1 birim dustukce ~%14 artiyor (2024/25 ve 2025/26'da ayri ayri dogrulandi:
# oyuncu duzeyi log-loss 0.4008 -> 0.3990 ve 0.3757 -> 0.3732). Takim gol beklentisine ek katkisi yok,
# yalnizca golcu olasiliklarini etkiler.
FLANK_B = -0.13
# referans: o ligdeki beklerin ortalama degeri (log EUR, 2025/26); kanatcinin gol orani zaten
# kendi liginin tipik beklerine karsi olculdugu icin fark lig ortalamasina gore alinir
FLANK_REF = {"ENG": 16.71, "ESP": 15.47, "FRA": 15.24, "GER": 15.55, "ITA": 15.62, "TUR": 14.12, "UCL": 16.28}
FLANK = {"Left Winger": ("Right-Back", "Right Midfield"), "Left Midfield": ("Right-Back", "Right Midfield"),
         "Right Winger": ("Left-Back", "Left Midfield"), "Right Midfield": ("Left-Back", "Left Midfield")}


def flank_adjust(lu, opp, league=None):
    """lu'daki kanat oyuncularinin gol oranini rakip ilk 11'deki karsi bekin degerine gore olcekler."""
    if not lu or not opp:
        return []
    notes = []
    for p in lu["xi"]:
        sides = FLANK.get(p.get("pos"))
        if not sides:
            continue
        fb = [q for q in opp["xi"] if q.get("pos") == sides[0]] or [q for q in opp["xi"] if q.get("pos") == sides[1]]
        if not fb or not fb[0].get("value"):
            continue
        lv = np.log(max(fb[0]["value"], 1e5))
        ref = FLANK_REF.get(league, FLANK_REF["UCL"])
        mult = float(np.clip(np.exp(FLANK_B * (lv - ref)), 0.75, 1.35))
        p["rate90"] = p["rate90"] * mult
        if abs(mult - 1) >= 0.1:
            notes.append(f"{p['name']} vs {fb[0]['name']}: x{mult:.2f}")
    return notes
