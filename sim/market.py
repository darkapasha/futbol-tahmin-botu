"""Bahis piyasasi harmani ve hakem etkisi (22.10.2026 testlerinde iki sezonda da isabeti artiranlar).

Piyasa: football-data.co.uk oranlari (veri thread'i her sabah futbol.sqlite fd_fixtures'a yaziyor; 6 ulke
ligi, kupalar yok). 2024/25 ve 2025/26'da 1X2 log-loss: motor 0.9704 / 0.9910, piyasa 0.9583 / 0.9771,
%80 piyasa + %20 motor 0.9592 / 0.9783. Oran varsa hedef olasiliklar bu harmandir; motorun beklenen
golleri hedefe uyacak sekilde kaydirilip mac yine simule edilir (skor, golcu, kart ciktilari tutarli kalir).

Hakem: hakemin gecmis maclarindaki kirmizi kart orani (10 maclik ortalamaya buzulmus). Kirmizi kart
log-loss 0.4555 -> 0.4551 ve 0.4558 -> 0.4522. 2.5 ust uzerindeki etkisi tutarsiz cikti, kullanilmiyor.
"""
import numpy as np
import pandas as pd

from .common import norm, src_con
from .rolling import outcome_probs, score_matrix

MARKET_W = 0.9   # 22.10 ikinci test: kapanisa yakin oranla motor payi azaldikca isabet artiyor (0/10/20%:
                 # 0.9543/0.9548/0.9555 ve 0.9760/0.9766/0.9773); simulasyon ciktilari icin %10 tutuldu
TEMP = 0.9       # harman 1X2'ye sicaklik: favoriler sistematik az tahmin ediliyordu; capraz sezonda
                 # 0.9783 -> 0.9774 ve 0.9585 -> 0.9570
LIVE_URL = "https://football-data.co.uk/fixtures.csv"
REF_C = 2.75   # logit kaymasi / (hakem kirmizi orani - ortalama); egitim sezonlarindan
REF_K = 10.0


def refresh_live():
    """Mac saatine yakin guncel oranlar (kapanis orani acilistan daha isabetli): fixtures.csv -> sim.sqlite fd_live."""
    import io
    import requests
    from .common import sim_con
    r = requests.get(LIVE_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")), on_bad_lines="skip")
    d.columns = [c.lower() for c in d.columns]
    d["date"] = pd.to_datetime(d.date, dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
    d["fetched_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    con = sim_con()
    d.to_sql("fd_live", con, if_exists="replace", index=False)
    con.commit()
    return len(d)


def temper(p, t=TEMP):
    q = np.array(p, float) ** (1 / t)
    return list(q / q.sum())


class Market:
    def __init__(self, con=None):
        con = con or src_con()
        try:
            fx = pd.read_sql("SELECT * FROM fd_fixtures", con)
        except Exception:
            fx = pd.DataFrame()
        try:  # maca yakin tazelenmis oranlar (varsa onlar oncelikli)
            from .common import sim_con
            live = pd.read_sql("SELECT * FROM fd_live", sim_con())
            fx = pd.concat([live, fx], ignore_index=True)
        except Exception:
            pass
        km = pd.read_sql("SELECT name, team_key FROM team_map WHERE source='xgabora'", con)
        km = dict(zip(km.name, km.team_key))
        if len(fx):
            fx["home"], fx["away"] = fx.hometeam.map(km), fx.awayteam.map(km)
            fx["date"] = pd.to_datetime(fx.date)
        self.fx = fx

    def odds(self, home, away, date):
        """(p_ev, p_x, p_dep, p_ust25) marjsiz; yoksa None."""
        if self.fx.empty:
            return None
        d = pd.Timestamp(date)
        m = self.fx[(self.fx.home == home) & (self.fx.away == away) & ((self.fx.date - d).abs() <= pd.Timedelta(days=2))]
        if "fetched_at" in m:  # en taze kayit once
            m = m.sort_values("fetched_at", ascending=False, na_position="last")
        if m.empty:
            return None
        r = m.iloc[0]
        trio = [(r.get("avgch"), r.get("avgcd"), r.get("avgca")), (r.get("avgh"), r.get("avgd"), r.get("avga")),
                (r.get("b365h"), r.get("b365d"), r.get("b365a"))]
        o = next((t for t in trio if all(pd.notna(x) and x > 1 for x in t)), None)
        if o is None:
            return None
        inv = 1 / np.array(o, float)
        p = inv / inv.sum()
        ou = None
        for a, b in (("avgc>2.5", "avgc<2.5"), ("avg>2.5", "avg<2.5"), ("b365>2.5", "b365<2.5")):
            if pd.notna(r.get(a)) and pd.notna(r.get(b)):
                io = 1 / np.array([r[a], r[b]], float)
                ou = float(io[0] / io.sum())
                break
        return float(p[0]), float(p[1]), float(p[2]), ou


def _poisson_targets(mh, ma):
    P3 = score_matrix([mh], [ma])
    p = outcome_probs(P3)[0]
    i, j = np.indices(P3.shape[1:])
    return p, float((P3[0] * ((i + j) > 2)).sum())


def fit_mu(mh, ma, engine_p, engine_ou, target_p, target_ou):
    """Motorun mu'larini, (hedef - motor) farkini Poisson yaklasimiyla kapatacak sekilde kaydir."""
    p0, ou0 = _poisson_targets(mh, ma)
    want_p = np.clip(p0 + (np.array(target_p) - np.array(engine_p)), 0.01, 0.98)
    want_ou = None if target_ou is None else float(np.clip(ou0 + target_ou - engine_ou, 0.05, 0.95))
    best, best_e = (mh, ma), None
    for step in (0.2, 0.05, 0.0125):
        cx, cy = np.log(best[0]), np.log(best[1])
        for dx in np.arange(-4, 5) * step:
            for dy in np.arange(-4, 5) * step:
                h, a = np.exp(cx + dx), np.exp(cy + dy)
                p, ou = _poisson_targets(h, a)
                e = (p[0] - want_p[0]) ** 2 + (p[2] - want_p[2]) ** 2 + (0 if want_ou is None else (ou - want_ou) ** 2)
                if best_e is None or e < best_e:
                    best, best_e = (h, a), e
    return best


class Referees:
    def __init__(self, con=None):
        con = con or src_con()
        tg = pd.read_sql("SELECT game_id, referee FROM tm_games WHERE referee IS NOT NULL", con)
        reds = pd.read_sql("SELECT game_id, COUNT(*) r FROM tm_events WHERE type='Cards' AND "
                           "(description LIKE '%Red card%' OR description LIKE '%Second yellow%') GROUP BY 1", con)
        tg = tg.merge(reds, on="game_id", how="left").fillna({"r": 0})
        tg["red"] = (tg.r > 0).astype(float)
        self.mean = float(tg.red.mean())
        g = tg.groupby("referee").red.agg(["sum", "size"])
        rate = (g["sum"] + REF_K * self.mean) / (g["size"] + REF_K)
        self.rate = {norm(k): float(v) for k, v in rate.items()}
        self.n = {norm(k): int(v) for k, v in g["size"].items()}

    def factor(self, name):
        """Mac basina kirmizi kart riskine carpan (logit kaymasinin kucuk olasilikta yaklasigi)."""
        if not name:
            return 1.0, None
        r = self.rate.get(norm(name))
        if r is None:
            return 1.0, None
        return float(np.clip(np.exp(REF_C * (r - self.mean)), 0.6, 1.8)), r
