"""Profil ozelliklerinin tanimi: her ozellik = agirlikli (pay toplami / payda toplami).

Ayni tanimlar takim, hoca ve rakip gucune gore bolunmus hoca profillerinde kullanilir.
Payda '1' ise pay sutununun dolu oldugu mac sayisidir (mac basina ortalama).
Yalnizca pay ve paydanin ikisi de dolu olan maclar sayilir; boylece kaynagi olmayan
lig/mac ortalamayi bozmaz.
"""
import numpy as np
import pandas as pd

# ad: (pay, payda, aciklama, kaynak)
FEATURES = {
    # Sonuc ve guc
    "ppg": ("pts", "1", "mac basina puan", "tm"),
    "gf_pg": ("gf", "1", "mac basina atilan gol", "tm"),
    "ga_pg": ("ga", "1", "mac basina yenilen gol", "tm"),
    "home_ppg": ("pts_home", "is_home", "iç sahada mac basina puan", "tm"),
    "away_ppg": ("pts_away", "is_away", "deplasmanda mac basina puan", "tm"),
    "home_gd_pg": ("gd_home", "is_home", "iç saha averaj / mac", "tm"),
    "away_gd_pg": ("gd_away", "is_away", "deplasman averaj / mac", "tm"),
    # Sut ve sut kalitesi
    "xg_pg": ("xg_any", "1", "mac basina xG (Understat, yoksa FotMob)", "us/fm"),
    "xga_pg": ("xga_any", "1", "mac basina yenilen xG", "us/fm"),
    "npxg_pg": ("npxg", "1", "penaltisiz xG / mac", "us"),
    "npxga_pg": ("npxga", "1", "penaltisiz yenilen xG / mac", "us"),
    "shots_pg": ("shots_any", "1", "sut / mac", "xg/fm"),
    "shots_a_pg": ("shots_a_any", "1", "rakip sut / mac", "xg/fm"),
    "xg_per_shot": ("xg_any", "shots_any", "sut basina xG (sut kalitesi)", "us/fm+xg"),
    "xga_per_shot": ("xga_any", "shots_a_any", "rakip sut basina xG", "us/fm+xg"),
    "sot_ratio": ("sot", "shots", "isabetli sut orani", "xg"),
    "finishing": ("gf", "xg_any", "gol / xG (bitiricilik, >1 iyi)", "us/fm"),
    "keeping": ("ga", "xga_any", "yenilen gol / yenilen xG (<1 iyi)", "us/fm"),
    # Pres, topla oynama, tempo
    "ppda": ("ppda_att", "ppda_def", "rakip pas / bizim savunma hamlesi (dusuk = yogun pres)", "us"),
    "oppda": ("oppda_att", "oppda_def", "bize yapilan pres (dusuk = cok baski yiyoruz)", "us"),
    "ppda_proxy": ("ppda_proxy", "1", "FotMob pres vekili: rakibin kendi yari pasi / (tackle+intercept+faul)", "fm"),
    "deep_pg": ("deep", "1", "ceza sahasina yakin tamamlanan pas / mac", "us"),
    "deep_allowed_pg": ("deep_allowed", "1", "rakibe izin verilen deep / mac", "us"),
    "possession": ("poss", "1", "topla oynama %", "fm"),
    "opp_half_share": ("opp_half_passes", "passes", "paslarin rakip yarida olani (saha hakimiyeti)", "fm"),
    "long_ball_share": ("long_balls", "passes", "isabetli uzun top / pas (direktlik)", "fm"),
    "passes_pg": ("passes", "1", "pas / mac", "fm"),
    "box_touches_pg": ("box_touches", "1", "rakip ceza sahasinda topla bulusma / mac", "fm"),
    "counter_goal_share": ("gf_counter", "gf_nonown", "kontra gol payi", "tm"),
    "fastbreak_xg_share": ("xg_fastbreak", "fm_xg_shots", "hizli hucum xG payi", "fm"),
    # Kanat / merkez
    "width_idx": ("width_idx", "1", "ilk 11'de kanat oyuncusu + bek orani", "tm"),
    "wide_att_pg": ("n_wide_att", "1", "ilk 11'de kanat hucumcu sayisi", "tm"),
    "cross_goal_share": ("gf_cross", "gf_nonown", "ortadan gelen gol payi", "tm"),
    "crosses_pg": ("crosses", "1", "isabetli orta / mac", "fm"),
    "corners_pg": ("corners", "1", "korner / mac", "xg"),
    "corners_a_pg": ("corners_a", "1", "rakip korner / mac", "xg"),
    # Duran top
    "setpiece_goal_share": ("gf_setpiece", "gf_nonown", "duran toptan gol payi (penalti haric)", "tm"),
    "setpiece_ga_share": ("ga_setpiece", "ga_nonown", "duran toptan yenilen gol payi", "tm"),
    "header_goal_share": ("gf_header", "gf_nonown", "kafa gol payi", "tm"),
    "header_ga_share": ("ga_header", "ga_nonown", "kafa ile yenilen gol payi", "tm"),
    "pen_for_pg": ("gf_pen", "1", "penalti golu / mac", "tm"),
    "setpiece_xg_share": ("xg_setpiece", "fm_xg_shots", "duran top xG payi", "fm"),
    # Oyun durumu (skor onde / esit / geride)
    "share_leading": ("min_lead", "min_total", "onde gecen dakika payi", "tm"),
    "share_trailing": ("min_trail", "min_total", "geride gecen dakika payi", "tm"),
    "gf90_lead": ("gf_lead", "min_lead90", "onde iken 90 dk basina gol", "tm"),
    "ga90_lead": ("ga_lead", "min_lead90", "onde iken 90 dk basina yenilen", "tm"),
    "gf90_level": ("gf_level", "min_level90", "esitken 90 dk basina gol", "tm"),
    "ga90_level": ("ga_level", "min_level90", "esitken 90 dk basina yenilen", "tm"),
    "gf90_trail": ("gf_trail", "min_trail90", "geride iken 90 dk basina gol", "tm"),
    "ga90_trail": ("ga_trail", "min_trail90", "geride iken 90 dk basina yenilen", "tm"),
    "ppg_scored_first": ("pts_sf", "is_sf", "ilk golu atinca mac basina puan (skoru koruma)", "tm"),
    "ppg_conceded_first": ("pts_cf", "is_cf", "ilk golu yiyince mac basina puan (geri donus)", "tm"),
    "second_half_gd_pg": ("gd_2h", "1", "ikinci yari averaji / mac", "tm"),
    # Disiplin, degisiklik
    "fouls_pg": ("fouls", "1", "faul / mac", "xg"),
    "yellow_pg": ("yellow", "1", "sari kart / mac", "xg"),
    "red_pg": ("red", "1", "kirmizi kart / mac", "xg"),
    "first_sub_min": ("first_sub_min", "1", "ilk degisiklik dakikasi", "tm"),
    "subs_by_60_pg": ("subs_by_60", "1", "60. dakikaya kadar degisiklik", "tm"),
    # Dizilis
    "back3_share": ("back3", "1", "3'lu savunma ile baslama orani", "tm"),
    "back5_share": ("back5", "1", "5'li savunma ile baslama orani", "tm"),
    "strikers_pg": ("n_fwd", "1", "dizilisteki forvet sayisi", "tm"),
    # Piyasa
    "mkt_win": ("mkt_win", "1", "bahis oranlarindan ortalama kazanma ihtimali", "xg"),
    "elo_diff_avg": ("elo_diff", "1", "ortalama Elo farki (ic saha dahil; >0 favori)", "elo"),
}


def prepare(t):
    """Ozellik paylarinin ihtiyac duydugu yardimci sutunlari ekler."""
    t = t.copy()
    t["is_away"] = 1 - t.is_home
    t["pts_home"] = t.pts.where(t.is_home == 1)
    t["pts_away"] = t.pts.where(t.is_home == 0)
    gd = t.gf - t.ga
    t["gd_home"] = gd.where(t.is_home == 1)
    t["gd_away"] = gd.where(t.is_home == 0)
    t["xg_any"] = t.xg.fillna(t.fm_xg) if "fm_xg" in t else t.xg
    t["xga_any"] = t.xga.fillna(t.fm_xga) if "fm_xga" in t else t.xga
    t["shots_any"] = t.shots.fillna(t.get("fm_shots"))
    t["shots_a_any"] = t.shots_a.fillna(t.get("fm_shots_a"))
    t["fm_xg_shots"] = t.get("fm_xg").where(t.get("n_shots_fm").notna()) if "fm_xg" in t else np.nan
    has_ev = t.min_lead.notna()
    t["gf_nonown"] = t.gf.where(has_ev)
    t["ga_nonown"] = t.ga.where(has_ev)
    t["min_total"] = (t.min_lead + t.min_level + t.min_trail)
    for s in ("lead", "level", "trail"):
        t[f"min_{s}90"] = t[f"min_{s}"] / 90
    t["is_sf"] = (t.scored_first == 1).astype(float).where(t.scored_first.notna())
    t["is_cf"] = (t.scored_first == 0).astype(float).where(t.scored_first.notna())
    t["pts_sf"] = t.pts.where(t.scored_first == 1)
    t["pts_cf"] = t.pts.where(t.scored_first == 0)
    t["gd_2h"] = (t.gf - t.ht_gf) - (t.ga - t.ht_ga)
    return t


def aggregate(t, w=None):
    """Agirlikli profil (tek takim/hoca/bolum icin). Donus: Series, ozellik + n_<ozellik> yok."""
    w = pd.Series(1.0, index=t.index) if w is None else w
    out = {"matches": len(t), "weight": float(w.sum())}
    for name, (num, den, *_ ) in FEATURES.items():
        if num not in t:
            out[name] = np.nan
            continue
        n = t[num].astype(float)
        d = pd.Series(1.0, index=t.index) if den == "1" else t[den].astype(float)
        ok = n.notna() & d.notna()
        ds = (d[ok] * w[ok]).sum()
        out[name] = (n[ok] * w[ok]).sum() / ds if ds > 0 else np.nan
    f = t.formation.dropna()
    if len(f):
        vc = f.value_counts(normalize=True)
        out["main_formation"] = vc.index[0]
        out["main_formation_share"] = float(vc.iloc[0])
        out["formation_entropy"] = float(-(vc * np.log2(vc)).sum())
        out["formations"] = ", ".join(f"{k} {v:.0%}" for k, v in vc.head(4).items())
    return pd.Series(out)


def recency_weights(dates, half_life=15):
    """Son maca 1, her 'half_life' mac geriye yarim agirlik."""
    order = pd.Series(range(len(dates)), index=dates.index)
    k = order.max() - order
    return 0.5 ** (k / half_life)
