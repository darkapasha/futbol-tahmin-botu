"""Telegram mesaj metinleri (HTML). Veriyi data/sim.sqlite'tan okur, gondermez."""
import html
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from sim.common import sim_con

TR = timezone(timedelta(hours=3))
LEAGUE_TR = {"ENG": "Premier Lig", "ESP": "La Liga", "GER": "Bundesliga", "ITA": "Serie A",
             "FRA": "Ligue 1", "TUR": "Süper Lig", "UCL": "Şampiyonlar Ligi", "UEL": "Avrupa Ligi"}
FLAG = {"ENG": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "ESP": "🇪🇸", "GER": "🇩🇪", "ITA": "🇮🇹", "FRA": "🇫🇷", "TUR": "🇹🇷", "UCL": "🏆", "UEL": "🟠"}
LEAGUE_ORDER = ["TUR", "UCL", "UEL", "ENG", "ESP", "ITA", "GER", "FRA"]


def e(s):
    return html.escape(str(s))


def pct(p):
    return f"%{100 * p:.0f}"


def tr_time(utc):
    return pd.Timestamp(utc).tz_convert(TR).strftime("%H:%M") if utc else "?"


def day_predictions(ymd_tr):
    """TR gunune gore her macin en son tahmini."""
    con = sim_con()
    try:
        p = pd.read_sql("SELECT match_id, created_at, utc_time, league, data FROM predictions", con)
    except Exception:
        return []
    if p.empty:
        return []
    p["day"] = pd.to_datetime(p.utc_time, utc=True).dt.tz_convert(TR).dt.strftime("%Y%m%d")
    p = p[p.day == ymd_tr].sort_values("created_at").groupby("match_id").last().reset_index()
    out = [json.loads(d) for d in p.data]
    out.sort(key=lambda o: (LEAGUE_ORDER.index(o["league"]) if o["league"] in LEAGUE_ORDER else 9, o.get("utc_time") or ""))
    return out


def pick(o):
    """En olasi sonuc ve guven etiketi."""
    ps = {"1": o["p_home"], "X": o["p_draw"], "2": o["p_away"]}
    k = max(ps, key=ps.get)
    conf = "🔒" if ps[k] >= 0.6 else ("✅" if ps[k] >= 0.47 else "⚖️")
    return k, ps[k], conf


def low_data(o):
    """Kupada verimiz az olan takim (lig disi rakip) varsa tahmin daha belirsiz."""
    return any(o.get("info", {}).get("cup_prior") or [])


def short_line(o):
    k, p, conf = pick(o)
    warn = " ⚠️" if low_data(o) else ""
    return (f"{tr_time(o.get('utc_time'))} <b>{e(o['home_name'])}</b> – <b>{e(o['away_name'])}</b>{warn}\n"
            f"   1 {pct(o['p_home'])} · X {pct(o['p_draw'])} · 2 {pct(o['p_away'])}  {conf} <b>{k}</b>"
            f" · 2.5Ü {pct(o['over']['2.5'])} · KG {pct(o['btts'])} · {o['scores'][0][0]}")


def day_summary(ymd_tr, league=None):
    ps = day_predictions(ymd_tr)
    if league:
        ps = [o for o in ps if o["league"] == league]
    d = datetime.strptime(ymd_tr, "%Y%m%d")
    if not ps:
        return f"📅 {d:%d.%m.%Y}: takip ettiğim liglerde maç yok (ya da tahminler henüz hazır değil)."
    lines = [f"📅 <b>{d:%d.%m.%Y} maç tahminleri</b> ({len(ps)} maç)"]
    cur = None
    for o in ps:
        if o["league"] != cur:
            cur = o["league"]
            lines.append(f"\n{FLAG.get(cur, '')} <b>{LEAGUE_TR.get(cur, cur)}</b>")
        lines.append(short_line(o))
    lines.append("\n🔒 güçlü favori · ✅ favori · ⚖️ dengeli"
                 + (" · ⚠️ verisi az takım (Elo / kadro değeriyle)" if any(low_data(o) for o in ps) else "")
                 + "\nDetay: /mac takım_adı")
    return "\n".join(lines)


def match_card(o, comment=None):
    k, p, conf = pick(o)
    h, a = o["home"], o["away"]
    lg = LEAGUE_TR.get(o["league"], o["league"])
    lines = [
        f"⚽ <b>{e(o['home_name'])} – {e(o['away_name'])}</b>",
        f"{FLAG.get(o['league'], '')} {lg} · {tr_time(o.get('utc_time'))}",
        "",
        f"<b>1</b> {pct(o['p_home'])}   <b>X</b> {pct(o['p_draw'])}   <b>2</b> {pct(o['p_away'])}   {conf}",
        f"Beklenen gol: {o['exp_goals'][0]:.2f} – {o['exp_goals'][1]:.2f}  (şut {o['exp_shots'][0]:.0f} – {o['exp_shots'][1]:.0f})",
        f"2.5 Üst {pct(o['over']['2.5'])} · 3.5 Üst {pct(o['over']['3.5'])} · KG Var {pct(o['btts'])}",
        "Olası skorlar: " + ", ".join(f"{s} ({pct(q)})" for s, q in o["scores"][:4]),
        f"İlk yarı: 1 {pct(o['p_ht'][0])} · X {pct(o['p_ht'][1])} · 2 {pct(o['p_ht'][2])}",
        f"İlk golü atan: {e(o['home_name'])} {pct(o['first_goal'][0])} · {e(o['away_name'])} {pct(o['first_goal'][1])}"
        f" · golsüz {pct(o['first_goal'][2])}",
        f"Kırmızı kart olasılığı: {pct(o['p_red'])}",
    ]
    sc = [s for side in o["scorers"] for s in side]
    if sc:
        top = sorted(sc, key=lambda s: -s["p"])[:6]
        lines.append("🎯 Gol atar: " + ", ".join(f"{e(s['name'])} {pct(s['p'])}" for s in top))
    for side, name in ((h, o["home_name"]), (a, o["away_name"])):
        miss = side.get("missing") or []
        if miss:
            lines.append(f"🚑 {e(name)} eksik: " + ", ".join(e(m) for m in miss[:6]))
    coaches = f"👔 {e(h.get('coach') or '?')} – {e(a.get('coach') or '?')}"
    lines.append(coaches)
    src = {"fotmob-muhtemel": "muhtemel 11", "fotmob-aciklanan": "açıklanan 11"}.get(h.get("lineup_source"), "son 11")
    lines.append(f"<i>Kadro: {src} · 10.000 simülasyon</i>")
    inf = o.get("info", {})
    if inf.get("piyasa"):
        mp, kp = inf["motor"]["p"], inf["piyasa"]["p"]
        lines.append(f"<i>📊 Simülasyon 1 {pct(mp[0])} X {pct(mp[1])} 2 {pct(mp[2])} · bahis piyasası 1 {pct(kp[0])} "
                     f"X {pct(kp[1])} 2 {pct(kp[2])} · gösterilen: %{100 * inf['piyasa']['agirlik']:.0f} piyasa harmanı</i>")
    if inf.get("referee") and inf["referee"].get("red_rate") is not None:
        r = inf["referee"]
        lines.append(f"<i>🟥 Hakem {e(r['name'])}: maçlarının {pct(r['red_rate'])}'inde kırmızı kart</i>")
    cp = inf.get("cup_prior") or []
    for side, name in zip(cp, (o["home_name"], o["away_name"])):
        if side:
            src_tr = side.replace("ilk 11 degeri", "ilk 11 değeri").replace("varsayilan", "varsayılan değer")
            lines.append(f"<i>⚠️ {e(name)} için maç verim az; gücü {e(src_tr)} ile tahmin edildi.</i>")
    if comment:
        lines += ["", f"🧠 {e(comment)}"]
    return "\n".join(lines)


def find_match(ps, query):
    from sim.common import norm
    q = norm(query)
    for o in ps:
        names = [norm(o["home_name"]), norm(o["away_name"])]
        if any(q in n or n in q for n in names if n):
            return o
    import difflib
    best, score = None, 0.0
    for o in ps:
        for n in (norm(o["home_name"]), norm(o["away_name"])):
            r = difflib.SequenceMatcher(None, q, n).ratio()
            if r > score:
                best, score = o, r
    return best if score >= 0.6 else None


def stats_text():
    con = sim_con()
    try:
        ev = pd.read_sql("SELECT * FROM pred_eval", con)
    except Exception:
        ev = pd.DataFrame()
    if ev.empty:
        return "📊 Henüz sonuçlanmış tahmin yok. İlk maçlar bittikten sonra burada başarı oranım görünecek."
    import numpy as np
    ev["date"] = pd.to_datetime(ev.ymd)
    y = np.where(ev.hg > ev.ag, 0, np.where(ev.hg == ev.ag, 1, 2))
    ev["hit"] = np.argmax(ev[["p_home", "p_draw", "p_away"]].values, 1) == y
    ev["ou_hit"] = (ev.over25 > 0.5) == ((ev.hg + ev.ag) > 2.5)
    ev["kg_hit"] = (ev.btts > 0.5) == ((ev.hg > 0) & (ev.ag > 0))
    ev["conf"] = ev[["p_home", "p_draw", "p_away"]].max(1)
    lines = ["📊 <b>Tahmin karnem</b>"]
    for label, d in (("Son 7 gün", 7), ("Son 30 gün", 30), ("Tüm zamanlar", 10000)):
        w = ev[ev.date >= pd.Timestamp.now() - pd.Timedelta(days=d)]
        if len(w):
            lines.append(f"<b>{label}</b> ({len(w)} maç): 1X2 %{100 * w.hit.mean():.0f} · 2.5 Ü/A %{100 * w.ou_hit.mean():.0f}"
                         f" · KG %{100 * w.kg_hit.mean():.0f}")
    strong = ev[ev.conf >= 0.6]
    if len(strong):
        lines.append(f"🔒 Güçlü favorilerde isabet: %{100 * strong.hit.mean():.0f} ({len(strong)} maç)")
    lg = ev.groupby("league").hit.agg(["mean", "size"])
    lines.append("Liglere göre: " + ", ".join(f"{LEAGUE_TR.get(k, k)} %{100 * r['mean']:.0f} ({int(r['size'])})" for k, r in lg.iterrows()))
    try:
        log = pd.read_sql("SELECT * FROM learn_log ORDER BY ts DESC LIMIT 3", con)
        if len(log):
            lines.append("\n🔧 Son model ayarları:")
            for r in log.itertuples():
                lines.append(f"  {r.ts[:10]} {e(r.rule)} {e(r.target)}: {e(r.old)} → {e(r.new)}")
    except Exception:
        pass
    return "\n".join(lines)


# ------------------------------------------------------------ eglenceli ozellikler
def upsets(ps, n=5):
    """Favorinin net oldugu (>= %55) maclarda en yuksek surpriz (zayif taraf kazanir) olasiligi."""
    rows = []
    for o in ps:
        fav_home = o["p_home"] >= o["p_away"]
        fav, dog = (o["p_home"], o["p_away"]) if fav_home else (o["p_away"], o["p_home"])
        if fav >= 0.55:
            rows.append((dog, o, fav_home))
    rows.sort(key=lambda r: -r[0])
    if not rows:
        return "😴 Bugün belirgin favorisi olan maç yok, sürpriz listesi boş."
    lines = ["💥 <b>Sürpriz alarmı</b> (favori kaybeder)"]
    for dog, o, fav_home in rows[:n]:
        d = o["away_name"] if fav_home else o["home_name"]
        lines.append(f"{e(o['home_name'])} – {e(o['away_name'])}: <b>{e(d)}</b> kazanır {pct(dog)}")
    return "\n".join(lines)


def top_scorers(ps, n=10):
    rows = []
    for o in ps:
        for side, team in zip(o["scorers"], (o["home_name"], o["away_name"])):
            for s in side:
                rows.append((s["p"], s["name"], team, o))
    rows.sort(key=lambda r: -r[0])
    if not rows:
        return "Bugün için golcü tahmini yok."
    lines = ["🎯 <b>Bugün gol atması en olası oyuncular</b>"]
    for p, name, team, o in rows[:n]:
        lines.append(f"{pct(p)}  {e(name)} ({e(team)})")
    return "\n".join(lines)


def coupon(ps, n=3):
    """Eglence: en guvenilir n tahmin ve hepsinin birden tutma olasiligi."""
    cand = []
    for o in ps:
        k, p, _ = pick(o)
        cand.append((p, f"{e(o['home_name'])} – {e(o['away_name'])}: <b>{k}</b> ({pct(p)})"))
        ou = o["over"]["2.5"]
        if max(ou, 1 - ou) >= 0.62:
            cand.append((max(ou, 1 - ou), f"{e(o['home_name'])} – {e(o['away_name'])}: <b>2.5 {'Üst' if ou > 0.5 else 'Alt'}</b> ({pct(max(ou, 1 - ou))})"))
    cand.sort(key=lambda r: -r[0])
    seen, chosen = set(), []
    for p, txt in cand:
        key = txt.split(":")[0]
        if key in seen:
            continue
        seen.add(key)
        chosen.append((p, txt))
        if len(chosen) == n:
            break
    if not chosen:
        return "Bugün kupon çıkmadı."
    tot = 1.0
    for p, _ in chosen:
        tot *= p
    return ("🎟 <b>Günün üçlüsü</b> (eğlence amaçlı)\n" + "\n".join(t for _, t in chosen) +
            f"\nHepsinin tutma olasılığı: {pct(tot)}\n<i>Bahis tavsiyesi değildir.</i>")


def story_text(o, events, score, shots, xg):
    h, a = o["home_name"], o["away_name"]
    lines = [f"🎬 <b>Simüle edilen maç: {e(h)} – {e(a)}</b>", "Hakem düdüğü çaldı…"]
    names = (h, a)
    sc = [0, 0]
    ht_done = False
    for m, side, kind, who in events:
        if m > 45 and not ht_done:
            lines.append(f"⏸ İlk yarı: {sc[0]}-{sc[1]}")
            ht_done = True
        if kind in ("gol", "penalti"):
            sc[side] += 1
            tag = " (penaltı)" if kind == "penalti" else ""
            lines.append(f"{m}' ⚽ GOL! {e(who or names[side])}{tag} – {e(names[side])}  {sc[0]}-{sc[1]}")
        else:
            lines.append(f"{m}' 🟥 {e(names[side])} 10 kişi kaldı!")
    if not ht_done:
        lines.append(f"⏸ İlk yarı: {sc[0]}-{sc[1]}")
    lines.append(f"🏁 Maç sonu: <b>{score[0]}-{score[1]}</b>  (şut {shots[0]}-{shots[1]}, xG {xg[0]:.2f}-{xg[1]:.2f})")
    lines.append("<i>Bu, 10.000 olası senaryodan sadece biri. Gerçek tahmin için /mac</i>")
    return "\n".join(lines)


def yesterday_recap(ymd_yday):
    con = sim_con()
    try:
        ev = pd.read_sql("SELECT * FROM pred_eval", con)
    except Exception:
        return None
    if ev.empty:
        return None
    import numpy as np
    ev = ev[ev.ymd == ymd_yday]
    if ev.empty:
        return None
    y = np.where(ev.hg > ev.ag, 0, np.where(ev.hg == ev.ag, 1, 2))
    hit = (np.argmax(ev[["p_home", "p_draw", "p_away"]].values, 1) == y)
    return f"📈 Dün: {int(hit.sum())}/{len(ev)} maçta en olası sonucu doğru bildim."
