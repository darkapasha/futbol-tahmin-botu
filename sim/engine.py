"""Monte Carlo mac motoru: her maci N kez dakika dakika oynatir.

Her simulasyonda:
  - Iki hoca da mac oncesi plan secer: olagan plan ya da rakibe ozel plan (olasilik p_plan,
    etkisi gecmiste o rakip sinifina karsi puan modelinden sapmasi).
  - Her dakika her takim icin sut sayisi ~ Poisson(beklenen gol / sut basina xG). Beklenen gol:
        mu/90 * dakika carpani * skor durumu * kirmizi kart * hocanin skor tepkisi * plan
    Hocanin geride/ondeyken tepkisi degisiklik dakikasindan once yarim, sonra tam uygulanir
    (oyuncu degisikligiyle gelen hucum / savunma hamlesi).
  - Her sutun xG'si takimin sut kalitesine gore Beta dagilimindan cekilir; gol = xG olasiligi.
  - Penaltilar ayri (0.79 xG), kirmizi kart riski dakika ve skor durumuna gore.
  - Atilan her gol, o dakikada sahada olma olasiligi x gol orani ile bir oyuncuya yazilir.

Tum simulasyonlar numpy ile vektor halinde ilerler; 10 bin simulasyon ~0.1 sn.
"""
from dataclasses import dataclass, field

import numpy as np

from .dynamics import _bin

PEN_XG = 0.79
KAPPA = 3.5          # sut xG dagiliminin yogunlugu (FotMob sut haritasi: ort 0.11, ss 0.15)
OWN_GOAL_SHARE = 0.03
MINUTES = np.arange(1, 91)
TBIN = _bin(MINUTES)


@dataclass
class Side:
    name: str
    mu: float                    # mac oncesi beklenen gol (kadro + saha dahil)
    xg_per_shot: float = 0.11    # takimin sut kalitesi (sut sayisi = mu / xG-sut)
    pen_share: float = 0.09      # gollerin penalti payi
    red_factor: float = 1.0      # takimin kirmizi kart egilimi / lig ortalamasi
    plan_p: float = 0.0          # rakibe ozel plana gecme olasiligi
    plan_att: float = 1.0        # plan: kendi gol beklentisi carpani
    plan_dfn: float = 1.0        # plan: rakibin gol beklentisi carpani
    att_trail: float = 1.0       # hoca tepkisi: gerideyken atak carpani
    dfn_trail: float = 1.0       # gerideyken yenilen gol carpani
    att_lead: float = 1.0
    dfn_lead: float = 1.0
    sub_minute: float = 62.0     # geride iken ilk hamle dakikasi
    scorers: object = None       # players.scorer_table ciktisi (DataFrame) ya da None
    info: dict = field(default_factory=dict)


def _state_factor(S, gd, half):
    return S[(np.clip(gd, -2, 2) + 2) * 2 + half]


def simulate(home: Side, away: Side, dyn: dict, n=10000, seed=None):
    rng = np.random.default_rng(seed)
    T, S, R = np.array(dyn["time"]), np.array(dyn["state"]), np.array(dyn["red"])
    red_t, red_s = np.array(dyn["red_time"]), np.array(dyn["red_state"])
    sides = (home, away)

    # plan secimi
    plan = np.stack([rng.random(n) < s.plan_p for s in sides], 1)              # (n,2)
    att_plan = np.where(plan, [s.plan_att for s in sides], 1.0)
    dfn_plan = np.where(plan, [s.plan_dfn for s in sides], 1.0)
    base = np.array([s.mu for s in sides])[None, :] / 90.0 * att_plan * dfn_plan[:, ::-1]
    pen_sh = np.array([s.pen_share for s in sides])
    # penaltisiz sut basina xG (penaltilar takim xG/sut ortalamasini yukari ceker)
    xps = np.array([s.xg_per_shot * 0.93 for s in sides]).clip(0.05, 0.25)
    AK = xps * KAPPA
    redf = np.array([s.red_factor for s in sides])
    a_tr, d_tr = np.array([s.att_trail for s in sides]), np.array([s.dfn_trail for s in sides])
    a_ld, d_ld = np.array([s.att_lead for s in sides]), np.array([s.dfn_lead for s in sides])
    subm = np.array([s.sub_minute for s in sides])

    goals = np.zeros((n, 2), int)
    reds = np.zeros((n, 2), int)
    shots = np.zeros((n, 2), int)
    xg = np.zeros((n, 2))
    ht = None
    first = np.full(n, -1)          # ilk golu atan taraf
    goal_log = []                   # (sim, taraf, dakika, penalti mi)
    red_log = []                    # (sim, taraf, dakika)

    for m in MINUTES:
        half = int(m > 45)
        gd = goals - goals[:, ::-1]                                  # takim bakisi
        lam = base * T[TBIN[m - 1]] * _state_factor(S, gd, half)
        rstate = (reds > 0).astype(int) * 2 + (reds[:, ::-1] > 0).astype(int)
        lam = lam * R[rstate]
        # hoca tepkisi (degisiklik dakikasindan once yarim etki)
        ramp = np.where(m >= subm, 1.0, 0.5)
        trail, lead = gd < 0, gd > 0
        lam = lam * np.where(trail, a_tr ** ramp, 1.0) * np.where(lead, a_ld ** ramp, 1.0)
        lam = lam * np.where(trail[:, ::-1], d_tr[::-1] ** ramp[::-1], 1.0) * np.where(lead[:, ::-1], d_ld[::-1] ** ramp[::-1], 1.0)

        # penaltilar
        pen = rng.random((n, 2)) < lam * pen_sh / PEN_XG
        pen_goal = pen & (rng.random((n, 2)) < PEN_XG)
        # akan oyun ve duran top sutlari
        lam_np = lam * (1 - pen_sh)
        k = np.minimum(rng.poisson(lam_np / xps), 3)
        shots += k + pen
        g_np = np.zeros((n, 2), int)
        si, ti = np.nonzero(k)
        if len(si):
            c = k[si, ti]
            rs, rt = np.repeat(si, c), np.repeat(ti, c)
            a = AK[rt]
            x = rng.beta(a, KAPPA - a)
            np.add.at(xg, (rs, rt), x)
            np.add.at(g_np, (rs, rt), (rng.random(len(x)) < x).astype(int))
        xg += pen * PEN_XG
        scored = g_np + pen_goal
        if scored.any():
            new_first = (first < 0) & (scored.sum(1) > 0)
            first = np.where(new_first, np.where(scored[:, 0] > 0, 0, 1), first)
            si, ti = np.nonzero(scored)
            c = scored[si, ti]
            # ayni dakikada akan oyun + penalti ayni anda nadir; penalti golu ilk kayda yazilir
            rs, rt = np.repeat(si, c), np.repeat(ti, c)
            first_of = np.r_[True, (rs[1:] != rs[:-1]) | (rt[1:] != rt[:-1])]
            goal_log.append((rs, rt, np.full(len(rs), m), pen_goal[rs, rt] & first_of))
            goals += scored
        # kirmizi kart (en fazla 2)
        p_red = red_t[TBIN[m - 1]] * red_s[np.clip(gd, -2, 2) + 2] * redf
        new_red = (rng.random((n, 2)) < p_red) & (reds < 2)
        if new_red.any():
            si, ti = np.nonzero(new_red)
            red_log.append((si, ti, np.full(len(si), m)))
        reds += new_red
        if m == 45:
            ht = goals.copy()

    gl = None
    if goal_log:
        cols = [np.concatenate(c) for c in zip(*goal_log)]
        gl = np.zeros(len(cols[0]), dtype=[("sim", int), ("side", int), ("minute", int), ("pen", bool)])
        gl["sim"], gl["side"], gl["minute"], gl["pen"] = cols
    rl = [np.concatenate(c) for c in zip(*red_log)] if red_log else None
    return dict(goals=goals, ht=ht, reds=reds, shots=shots, xg=xg, first=first, plan=plan, goal_log=gl,
                red_log=rl, n=n)


def story(res, home: Side, away: Side, sim=None, seed=None):
    """Tek bir simulasyonun dakika dakika olaylari: [(dakika, taraf, 'gol'|'penalti'|'kirmizi', oyuncu)].
    sim verilmezse en olasi skorlardan birini veren rastgele bir simulasyon secilir."""
    rng = np.random.default_rng(seed)
    if sim is None:
        sim = int(rng.integers(res["n"]))
    ev = []
    gl = res["goal_log"]
    if gl is not None:
        for r in gl[gl["sim"] == sim]:
            side = (home, away)[r["side"]]
            t = side.scorers
            name = None
            if t is not None and len(t):
                if r["pen"]:
                    name = t.name.iat[int(np.argmax(t.pen_taker.values))]
                else:
                    ramp = np.clip((r["minute"] - 55) / 30.0, 0, 1)
                    w = np.where(t.start.values == 1, 1 - t.p_off.values * ramp, 0.3 * ramp) * t.rate.values
                    name = t.name.iat[int(rng.choice(len(t), p=w / w.sum()))] if w.sum() > 0 else None
            ev.append((int(r["minute"]), int(r["side"]), "penalti" if r["pen"] else "gol", name))
    rl = res.get("red_log")
    if rl is not None:
        for s_, t_, m_ in zip(*rl):
            if s_ == sim:
                ev.append((int(m_), int(t_), "kirmizi", None))
    ev.sort()
    return sim, [tuple(x) for x in ev], res["goals"][sim].tolist()


# ---------------------------------------------------------------- ozet
def summarize(res, home: Side, away: Side, top_scores=8):
    g = res["goals"]
    n = res["n"]
    h, a = g[:, 0], g[:, 1]
    out = {}
    out["p_home"], out["p_draw"], out["p_away"] = float((h > a).mean()), float((h == a).mean()), float((h < a).mean())
    out["exp_goals"] = [float(h.mean()), float(a.mean())]
    out["exp_xg"] = res["xg"].mean(0).round(2).tolist()
    out["exp_shots"] = res["shots"].mean(0).round(1).tolist()
    tot = h + a
    out["over"] = {f"{x}.5": float((tot > x).mean()) for x in range(0, 6)}
    out["btts"] = float(((h > 0) & (a > 0)).mean())
    out["clean_sheet"] = [float((a == 0).mean()), float((h == 0).mean())]
    hm = np.minimum(h, 9) * 10 + np.minimum(a, 9)
    cnt = np.bincount(hm, minlength=100) / n
    order = np.argsort(-cnt)[:top_scores]
    out["scores"] = [(f"{i // 10}-{i % 10}", float(cnt[i])) for i in order]
    mat = cnt.reshape(10, 10)
    out["score_matrix"] = mat[:7, :7].round(4).tolist()
    ht = res["ht"]
    hr = np.sign(ht[:, 0] - ht[:, 1])
    fr = np.sign(h - a)
    out["ht_ft"] = {f"{x}/{y}": float(((hr == i) & (fr == j)).mean())
                    for i, x in ((1, "1"), (0, "X"), (-1, "2")) for j, y in ((1, "1"), (0, "X"), (-1, "2"))}
    out["p_ht"] = [float((hr == 1).mean()), float((hr == 0).mean()), float((hr == -1).mean())]
    out["first_goal"] = [float((res["first"] == 0).mean()), float((res["first"] == 1).mean()), float((res["first"] < 0).mean())]
    out["p_red"] = float((res["reds"].sum(1) > 0).mean())
    out["plan_rate"] = res["plan"].mean(0).round(2).tolist()
    # handikap (Asya -1, +1 gibi basit)
    d = h - a
    out["margin"] = {str(k): float((d == k).mean()) for k in range(-4, 5)}
    out["scorers"] = [_scorers(res, s, i) for i, s in enumerate((home, away))]
    return out


def _scorers(res, side: Side, idx):
    t = side.scorers
    gl = res["goal_log"]
    if t is None or len(t) == 0 or gl is None:
        return []
    gl = gl[gl["side"] == idx]
    if len(gl) == 0:
        return []
    m = gl["minute"]
    ramp = np.clip((m - 55) / 30.0, 0, 1)[:, None]
    start = t.start.values[None, :] == 1
    p_on = np.where(start, 1 - t.p_off.values[None, :] * ramp, 0.0)
    # yedekler: cikan oyuncularin yerine; en fazla ~0.6 olasilikla sahada
    bench = ~start
    if bench.any():
        bw = t.rate.values * bench[0]
        bw = bw / bw.sum() if bw.sum() > 0 else bw
        p_on = np.where(bench, np.minimum(1.0, 2.5 * bw[None, :]) * 0.6 * ramp, p_on)
    w = p_on * t.rate.values[None, :]
    w = w / w.sum(1, keepdims=True)
    rng = np.random.default_rng(7)
    u = rng.random(len(gl))
    pick = (w.cumsum(1) < u[:, None]).sum(1).clip(0, len(t) - 1)
    # penalti -> penalti atani (sahadaysa)
    taker = int(np.argmax(t.pen_taker.values))
    pick = np.where(gl["pen"], taker, pick)
    own = rng.random(len(gl)) < OWN_GOAL_SHARE
    pick = np.where(own & ~gl["pen"], -1, pick)
    n = res["n"]
    mat = np.zeros((n, len(t)), int)
    ok = pick >= 0
    np.add.at(mat, (gl["sim"][ok], pick[ok]), 1)
    play = t.p_play.values if "p_play" in t else 1.0
    p_any = (mat > 0).mean(0) * play
    p_two = (mat > 1).mean(0) * play
    order = np.argsort(-p_any)
    return [dict(name=t.name.iat[i], p=float(p_any[i]), p2=float(p_two[i]), start=int(t.start.iat[i]))
            for i in order if p_any[i] >= 0.02][:10]
