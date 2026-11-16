"""Mac ici dinamikler: ~11 bin Transfermarkt macinin gol / kirmizi kart dakikalarindan.

Her takim-dakika icin beklenen gol = (mac oncesi mu / 90) * zaman[dk] * durum[skor farki, yari] * kart[kirmizi]
Carpanlar Poisson modeli olarak (yinelemeli oransal uydurma) birlikte kestirilir; mac oncesi mu
puan modelinden geldigi icin "guclu takim zaten onde olur" etkisi ayiklanmis olur.
Ayni sekilde kirmizi kart riski: dakika ve skor durumuna gore.

Dakika 45 ve 90 uzatmalari da icerir (Transfermarkt uzatmayi 45/90'a yaziyor); motor 90 adim
oynatip bu carpanlari kullandigi icin uzatma kendiliginden gelir.
"""
import numpy as np
import pandas as pd

GD_LEVELS = [-2, -1, 0, 1, 2]


def _bin(m):
    """dakika (1..90) -> zaman kutusu: 5'er dakika, 45 ve 90 ayri (uzatma)."""
    m = np.asarray(m)
    b = (m - 1) // 5
    return np.where(m == 45, 18, np.where(m == 90, 19, b))


N_BINS = 20


def build_arrays(g, ev, pred):
    """Mac x dakika dizileri: gol ve kirmizi kart sayilari (ev/dep)."""
    tm = g[g.source == "tm"].merge(pred, on="game_id")
    tm = tm[tm.game_id.isin(ev.game_id.unique()) | ((tm.hg == 0) & (tm.ag == 0))]
    idx = pd.Index(tm.game_id)
    n = len(tm)
    goals = np.zeros((n, 2, 91))
    reds = np.zeros((n, 2, 91))
    e = ev[ev.game_id.isin(idx)]
    gi = idx.get_indexer(e.game_id)
    side = 1 - e.is_home.values  # 0 = ev, 1 = dep
    mins = e.minute.clip(1, 90).values
    for arr, k in ((goals, "goal"), (reds, "red")):
        mk = (e.kind == k).values
        np.add.at(arr, (gi[mk], side[mk], mins[mk]), 1)
    # olay sayisi skorla tutmuyorsa (eksik olay) maci at
    ok = (goals[:, 0].sum(1) == tm.hg.values) & (goals[:, 1].sum(1) == tm.ag.values)
    pen = e[(e.kind == "goal")].pen.mean()
    return tm[ok].reset_index(drop=True), goals[ok], reds[ok], float(pen)


def fit(g, ev, pred, iters=40):
    tm, goals, reds, pen_share = build_arrays(g, ev, pred)
    n = len(tm)
    mu = np.stack([tm.mu_h.values, tm.mu_a.values], 1)  # (n,2)
    # durum: dakika m basindaki skor farki (takim bakisi) ve kirmizi kartlar
    cg = np.cumsum(goals, axis=2)[:, :, :-1]  # m-1'e kadar -> m=1..90
    cr = np.cumsum(reds, axis=2)[:, :, :-1]
    y = goals[:, :, 1:]  # (n,2,90)
    gd = np.stack([cg[:, 0] - cg[:, 1], cg[:, 1] - cg[:, 0]], 1).clip(-2, 2).astype(int) + 2
    half = (np.arange(1, 91) > 45).astype(int)[None, None, :] * np.ones_like(gd)
    st = gd * 2 + half  # 10 durum
    own_r = (cr > 0).astype(int)
    opp_r = own_r[:, ::-1]
    rs = own_r * 2 + opp_r  # 0: yok, 1: rakip eksik, 2: biz eksik, 3: ikisi de
    tb = np.broadcast_to(_bin(np.arange(1, 91))[None, None, :], y.shape)
    base = np.broadcast_to(mu[:, :, None] / 90.0, y.shape)
    T, S, R = np.ones(N_BINS), np.ones(10), np.ones(4)
    yf, bf, tbf, stf, rsf = y.ravel(), base.ravel(), tb.ravel(), st.ravel(), rs.ravel()
    for _ in range(iters):
        for arr, idx, K in ((T, tbf, N_BINS), (S, stf, 10), (R, rsf, 4)):
            lam = bf * T[tbf] * S[stf] * R[rsf]
            o = np.bincount(idx, yf, K)
            e = np.bincount(idx, lam, K)
            arr *= np.where(e > 0, (o + 1) / (e + 1), 1.0)
        S /= S[4]  # esit skor, ilk yari = 1
        R /= R[0]
        lam = bf * T[tbf] * S[stf] * R[rsf]
        T *= yf.sum() / lam.sum()
    expo_state = np.bincount(stf, minlength=10) / len(tm)

    # kirmizi kart riski: takim-dakika basina, zaman kutusu x skor farki
    yr = reds[:, :, 1:].ravel()
    alive = (np.broadcast_to(cr, y.shape).ravel() < 1)  # ilk kirmizi icin risk
    rt = np.bincount(tbf[alive], yr[alive], N_BINS) / np.maximum(np.bincount(tbf[alive], minlength=N_BINS), 1)
    gdf = (gd.ravel())[alive]
    rg = np.bincount(gdf, yr[alive], 5) / np.maximum(np.bincount(gdf, minlength=5), 1)
    rg = rg / (yr[alive].sum() / alive.sum())
    out = dict(
        n_games=int(n), time=T.tolist(), state=S.tolist(), red=R.tolist(), state_exposure=expo_state.tolist(),
        red_time=rt.tolist(), red_state=rg.tolist(), pen_share=pen_share,
        goals_pg=float(goals.sum() / n), reds_pg=float(reds.sum() / n),
    )
    return out


def describe(d):
    st = np.array(d["state"]).reshape(5, 2)
    print(f"{d['n_games']} mac, mac basina gol {d['goals_pg']:.2f}, kirmizi {d['reds_pg']:.3f}, penalti payi {d['pen_share']:.3f}")
    print("zaman carpani (5 dk):", np.round(d["time"], 2))
    for i, g in enumerate(GD_LEVELS):
        print(f"  skor farki {g:+d}: ilk yari {st[i,0]:.2f}  ikinci yari {st[i,1]:.2f}")
    print("kirmizi (yok, rakip eksik, biz eksik, ikisi):", np.round(d["red"], 2))
    print("kirmizi riski dk:", np.round(np.array(d["red_time"]) * 1000, 2), "skor farkina gore:", np.round(d["red_state"], 2))


def coach_state(g, ev, pred, d, k=6.0):
    """Hocanin skor durumuna tepkisi: genel dinamik modele gore, gerideyken / ondeyken
    takiminin attigi ve yedigi gol orani. Sonuc hoca basina carpanlar (k beklenen gollük buzulme)."""
    tm, goals, reds, _ = build_arrays(g, ev, pred)
    T, S, R = np.array(d["time"]), np.array(d["state"]), np.array(d["red"])
    mu = np.stack([tm.mu_h.values, tm.mu_a.values], 1)
    cg = np.cumsum(goals, axis=2)[:, :, :-1]
    cr = np.cumsum(reds, axis=2)[:, :, :-1]
    y = goals[:, :, 1:]
    gdr = np.stack([cg[:, 0] - cg[:, 1], cg[:, 1] - cg[:, 0]], 1)
    gd = gdr.clip(-2, 2).astype(int) + 2
    half = np.broadcast_to((np.arange(1, 91) > 45).astype(int)[None, None, :], gd.shape)
    own_r = (cr > 0).astype(int)
    rs = own_r * 2 + own_r[:, ::-1]
    tb = _bin(np.arange(1, 91))[None, None, :]
    lam = mu[:, :, None] / 90.0 * T[tb] * S[gd * 2 + half] * R[rs]
    rows = []
    coaches = np.stack([tm.home_coach.values, tm.away_coach.values], 1)
    for s_name, mask in (("trail", gdr < 0), ("lead", gdr > 0)):
        for side in (0, 1):
            o = (y[:, side] * mask[:, side]).sum(1)
            e = (lam[:, side] * mask[:, side]).sum(1)
            oa = (y[:, 1 - side] * mask[:, side]).sum(1)
            ea = (lam[:, 1 - side] * mask[:, side]).sum(1)
            rows.append(pd.DataFrame({"coach": coaches[:, side], "state": s_name, "gf": o, "xgf": e, "ga": oa, "xga": ea}))
    r = pd.concat(rows).dropna(subset=["coach"]).groupby(["coach", "state"]).sum()
    r["att"] = (r.gf + k) / (r.xgf + k)
    r["dfn"] = (r.ga + k) / (r.xga + k)
    w = r[["att", "dfn", "xgf"]].unstack("state")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    return w.reset_index()
