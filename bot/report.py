"""Tahmin raporu: her bitmis mac icin tahmin ile sonucu karsilastirir, gunluk ozeti
data/sim.sqlite 'daily_report' tablosunda saklar (gecmis raporlar silinmez, ustune yazilmaz).

  .venv/bin/python -m bot.report            # dunun raporunu kaydet ve yazdir
  .venv/bin/python -m bot.report hafta      # son 7 gun

Olculenler: 1X2 isabeti ve log-loss, tam skor, 2.5 alt/ust, karsilikli gol, golcu
(en olasi 3 golcu adayindan gol atan), guven dilimine gore isabet.
"""
import json
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from sim.common import sim_con, src_con

TR = timezone(timedelta(hours=3))
LEAGUE_TR = {"ENG": "Premier Lig", "ESP": "La Liga", "GER": "Bundesliga", "ITA": "Serie A",
             "FRA": "Ligue 1", "TUR": "Süper Lig", "UCL": "Şampiyonlar Ligi", "UEL": "Avrupa Ligi"}

SCHEMA = """CREATE TABLE IF NOT EXISTS daily_report (
  day TEXT PRIMARY KEY, created_at TEXT, matches INTEGER, data TEXT);"""


def matches(start, end):
    """[start, end) TR gunleri arasindaki bitmis maclar: son mac-oncesi tahmin + sonuc + golculer."""
    con = sim_con()
    pr = pd.read_sql("SELECT match_id, created_at, utc_time, league, data FROM predictions", con)
    if pr.empty:
        return pd.DataFrame()
    pr = pr[pr.created_at.str[:16] <= pr.utc_time.str[:16]]
    pr = pr.sort_values("created_at").groupby("match_id").last().reset_index()
    pr["day"] = pd.to_datetime(pr.utc_time, utc=True).dt.tz_convert(TR).dt.strftime("%Y%m%d")
    pr = pr[(pr.day >= start) & (pr.day < end)]
    res = pd.read_sql("SELECT match_id, home_goals hg, away_goals ag FROM fm_matches WHERE home_goals IS NOT NULL", src_con())
    try:
        cup = pd.read_sql("SELECT match_id, home_goals hg, away_goals ag FROM cup_results WHERE home_goals IS NOT NULL", con)
        res = pd.concat([res, cup[~cup.match_id.isin(res.match_id)]])
    except Exception:
        pass
    m = pr.merge(res, on="match_id")
    if m.empty:
        return m
    goals = pd.read_sql(f"SELECT match_id, player FROM fm_shots WHERE event_type='Goal' AND own_goal=0 "
                        f"AND match_id IN ({','.join(map(str, m.match_id))})", src_con())
    scorers = goals.groupby("match_id").player.apply(set).to_dict()
    rows = []
    for r in m.itertuples():
        o = json.loads(r.data)
        hg, ag = int(r.hg), int(r.ag)
        y = 0 if hg > ag else (1 if hg == ag else 2)
        P = np.array([o["p_home"], o["p_draw"], o["p_away"]])
        cands = sorted([s for side in o["scorers"] for s in side], key=lambda s: -s["p"])[:3]
        sc = scorers.get(r.match_id)
        rows.append(dict(
            match_id=r.match_id, day=r.day, league=r.league, mac=f"{o['home_name']} {hg}-{ag} {o['away_name']}",
            hit=int(P.argmax() == y), conf=float(P.max()), ll=float(-np.log(max(P[y], 1e-9))),
            brier=float(((P - np.eye(3)[y]) ** 2).sum()),
            exact=int(o["scores"][0][0] == f"{hg}-{ag}"),
            ou=int((o["over"]["2.5"] > 0.5) == (hg + ag > 2.5)),
            kg=int((o["btts"] > 0.5) == (hg > 0 and ag > 0)),
            scorer_known=int(sc is not None),
            scorer_hit=int(sc is not None and any(c["name"] in sc for c in cands)),
            scorer_exp=float(sum(c["p"] for c in cands)) if sc is not None else np.nan,
            tahmin=["1", "X", "2"][int(P.argmax())], sonuc=["1", "X", "2"][y]))
    return pd.DataFrame(rows)


def summarize(d):
    if d.empty:
        return {}
    s = dict(mac=len(d), isabet_1x2=float(d.hit.mean()), logloss=float(d.ll.mean()), brier=float(d.brier.mean()),
             tam_skor=float(d.exact.mean()), alt_ust=float(d.ou.mean()), kg=float(d.kg.mean()))
    k = d[d.scorer_known == 1]
    if len(k):
        s.update(golcu_mac=len(k), golcu_isabet=float(k.scorer_hit.mean()), golcu_beklenen=float(k.scorer_exp.clip(upper=1).mean()))
    strong = d[d.conf >= 0.6]
    if len(strong):
        s.update(guclu_mac=len(strong), guclu_isabet=float(strong.hit.mean()))
    s["lig"] = {lg: dict(mac=len(g), isabet=float(g.hit.mean())) for lg, g in d.groupby("league")}
    # kalibrasyon: "%60 dedigimiz maclar %60 tutuyor mu?" (en olasi sonucun olasiligina gore)
    bins = pd.cut(d.conf, [0, .4, .5, .6, .7, 1.0])
    k = d.groupby(bins, observed=True).agg(n=("hit", "size"), tahmin=("conf", "mean"), gercek=("hit", "mean"))
    s["kalibrasyon"] = [dict(aralik=str(i), n=int(r.n), tahmin=float(r.tahmin), gercek=float(r.gercek))
                        for i, r in k.iterrows() if r.n >= 3]
    return s


def save_day(day):
    d = matches(day, (datetime.strptime(day, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d"))
    s = summarize(d)
    if not s:
        return None
    s["maclar"] = d[["mac", "tahmin", "sonuc", "conf", "exact"]].to_dict("records")
    con = sim_con()
    con.executescript(SCHEMA)
    # gecmis raporun ustune yazilmaz; gec gelen sonuc varsa yalnizca mac sayisi artmissa guncellenir
    old = con.execute("SELECT matches FROM daily_report WHERE day = ?", (day,)).fetchone()
    if old is None or old[0] < s["mac"]:
        con.execute("INSERT OR REPLACE INTO daily_report VALUES (?,?,?,?)",
                    (day, datetime.now(TR).isoformat(timespec="minutes"), s["mac"], json.dumps(s, ensure_ascii=False)))
        con.commit()
    return s


def pct(x):
    return f"%{100 * x:.0f}"


def text(title, s, detail=None):
    if not s:
        return f"📋 <b>{title}</b>\nBu dönemde sonuçlanmış tahmin yok."
    lines = [f"📋 <b>{title}</b> ({s['mac']} maç)",
             f"1X2 doğru: {pct(s['isabet_1x2'])} · log-loss {s['logloss']:.3f}",
             f"Tam skor: {pct(s['tam_skor'])} · 2.5 Alt/Üst: {pct(s['alt_ust'])} · KG: {pct(s['kg'])}"]
    if "guclu_isabet" in s:
        lines.append(f"🔒 Güçlü favoriler (%60+): {pct(s['guclu_isabet'])} ({s['guclu_mac']} maç)")
    if "golcu_isabet" in s:
        lines.append(f"🎯 İlk 3 golcü adayımdan biri attı: {pct(s['golcu_isabet'])} "
                     f"(beklentim {pct(s['golcu_beklenen'])}, {s['golcu_mac']} maç)")
    lines.append("Liglere göre: " + ", ".join(f"{LEAGUE_TR.get(k, k)} {pct(v['isabet'])} ({v['mac']})"
                                              for k, v in s["lig"].items()))
    if detail is None and s.get("kalibrasyon"):
        lines.append("\n🎯 <b>Kalibrasyon</b> (dediğim olasılık → gerçekleşen)")
        for k in s["kalibrasyon"]:
            lines.append(f"  {pct(k['tahmin'])} dediklerim: {pct(k['gercek'])} tuttu ({k['n']} maç)")
    if detail is not None and len(detail):
        lines.append("")
        for r in detail.itertuples():
            mark = "✅" if r.hit else "❌"
            lines.append(f"{mark} {r.mac} (tahmin {r.tahmin}{', tam skor 🎯' if r.exact else ''})")
    lines.append("\n<i>Log-loss: düşük olan iyi; yazı-tura 1.10.</i>")
    return "\n".join(lines)


def day_text(day):
    d = matches(day, (datetime.strptime(day, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d"))
    return text(f"{day[6:]}.{day[4:6]}.{day[:4]} raporu", summarize(d), d.head(30) if len(d) else None)


def range_text(days, title):
    end = (datetime.now(TR) + timedelta(days=1)).strftime("%Y%m%d")
    start = (datetime.now(TR) - timedelta(days=days)).strftime("%Y%m%d")
    return text(title, summarize(matches(start, end)))


if __name__ == "__main__":
    if sys.argv[1:2] == ["hafta"]:
        print(range_text(7, "Son 7 gün"))
    else:
        day = sys.argv[1] if len(sys.argv) > 1 else (datetime.now(TR) - timedelta(days=1)).strftime("%Y%m%d")
        print(save_day(day))
        print(day_text(day))
