"""Tum maclar (lig + Sampiyonlar Ligi) uzerinden tek bir Elo; rakip gucu ayrimi icin.

Disaridaki clubelo tablosu isim eslestirmesi istedigi icin kendi Elo'muzu hesapliyoruz.
Baslangic puani lige gore verilir, SL maclari ligler arasi farki zamanla duzeltir.
"""
import numpy as np

SEED = {"ENG": 1600, "ESP": 1580, "GER": 1570, "ITA": 1570, "FRA": 1530, "TUR": 1450, "UCL": 1400}
K, HOME_ADV = 20, 60


def add_elo(long):
    """long (mac x takim) tablosuna elo_pre, opp_elo_pre, elo_diff ekler."""
    g = long[long.is_home == 1].sort_values(["date", "game_id"])
    league_of = (long[long.league != "UCL"].sort_values("date")
                 .groupby("team_key").league.first().to_dict())
    rating, pre = {}, {}
    for gid, h, a, gf, ga in g[["game_id", "team_key", "opp_key", "gf", "ga"]].itertuples(index=False):
        rh = rating.setdefault(h, SEED.get(league_of.get(h), 1400))
        ra = rating.setdefault(a, SEED.get(league_of.get(a), 1400))
        pre[(gid, h)], pre[(gid, a)] = rh, ra
        exp = 1 / (1 + 10 ** ((ra - rh - HOME_ADV) / 400))
        res = 1.0 if gf > ga else 0.5 if gf == ga else 0.0
        mult = np.log1p(abs(gf - ga)) + 1 if gf != ga else 1
        d = K * mult * (res - exp)
        rating[h], rating[a] = rh + d, ra - d
    idx = list(zip(long.game_id, long.team_key))
    long["elo_pre"] = [pre.get(k) for k in idx]
    long["opp_elo_pre"] = [pre.get(k) for k in zip(long.game_id, long.opp_key)]
    long["elo_diff"] = long.elo_pre - long.opp_elo_pre + np.where(long.is_home == 1, HOME_ADV, -HOME_ADV)
    # Rakip gucu sinifi: >+75 zayif rakip, <-75 guclu rakip
    long["opp_class"] = np.select([long.elo_diff > 75, long.elo_diff < -75], ["weaker", "stronger"], "similar")
    return rating
