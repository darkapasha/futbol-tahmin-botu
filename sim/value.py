"""Kadro piyasa degeri: takim gucu icin on bilgi (yukselen takimlar, yaz transferleri, yeni hoca).

Her oyuncunun o tarihteki kulubu = son forma giydigi kulup, sonrasinda transfer olduysa yeni kulup.
Takim degeri = o kulubun en degerli 18 oyuncusunun (o tarihteki son degerlemesiyle) toplami.
14'ten az oyuncusu bulunan takimin degeri bilinmiyor sayilir (alt ligden gelenlerin kadrosu
Transfermarkt verimizde eksik gorunur); onlar icin yukselen takim on bilgisi kullanilir.
Transfermarkt verisi bittikten sonra (Temmuz 2026) FotMob kadrolarindaki guncel degerler eklenir.
"""
import numpy as np
import pandas as pd

from .common import src_con

TOP_N = 18
MIN_PLAYERS = 14  # kapsadigimiz liglerin disindan gelen (orn. 2. ligden cikan) takimin kadrosu eksik gorunur


class ValueBook:
    def __init__(self, con=None):
        con = con or src_con()
        self.val = pd.read_sql("SELECT player_id, date, market_value_in_eur v FROM tm_valuations", con)
        self.val["date"] = pd.to_datetime(self.val.date)
        self.val = self.val.sort_values("date")
        ap = pd.read_sql("SELECT player_id, player_club_id club, date FROM tm_appearances", con)
        tr = pd.read_sql("SELECT player_id, to_club_id club, transfer_date date FROM tm_transfers", con)
        ev = pd.concat([ap.assign(kind=0), tr.assign(kind=1)])
        ev["date"] = pd.to_datetime(ev.date, errors="coerce")
        self.ev = ev.dropna(subset=["date", "club"]).sort_values("date")
        # FotMob guncel kadro degerleri (oyuncu basina son mac)
        fm = pd.read_sql("SELECT l.team_id, l.player_id, l.market_value v, m.utc_time FROM fm_lineups l "
                         "JOIN fm_matches m USING (match_id) WHERE l.market_value IS NOT NULL", con)
        fm["date"] = pd.to_datetime(fm.utc_time.str[:10])
        self.fm = fm.sort_values("date")
        tmap = pd.read_sql("SELECT name, team_key FROM team_map WHERE source='fotmob'", con)
        self.fm_key = {int(n): k for n, k in zip(tmap.name, tmap.team_key) if str(n).isdigit()}
        self.tm_end = self.ev.date.max()

    def at(self, asof, fm_resolve=None):
        """team_key -> log(kadro degeri)."""
        asof = pd.Timestamp(asof)
        ev = self.ev[(self.ev.date < asof) & (self.ev.date >= asof - pd.Timedelta(days=500))]
        club = ev.groupby("player_id").club.last()
        v = self.val[self.val.date < asof].groupby("player_id").v.last()
        d = pd.DataFrame({"club": club}).join(v, how="inner").dropna()
        d["team"] = "tm:" + d.club.astype(int).astype(str)
        grp = d.groupby("team").v
        out = grp.apply(lambda s: s.nlargest(TOP_N).sum())
        out = out[(out > 0) & (grp.size() >= MIN_PLAYERS)]
        if asof > self.tm_end + pd.Timedelta(days=20) and len(self.fm):
            f = self.fm[(self.fm.date < asof) & (self.fm.date >= asof - pd.Timedelta(days=120))]
            if len(f):
                last = f.groupby("player_id").last()
                fg = last.groupby("team_id").v
                fv = fg.apply(lambda s: s.nlargest(TOP_N).sum())[fg.size() >= MIN_PLAYERS]
                for tid, val in fv.items():
                    k = self.fm_key.get(int(tid)) or (fm_resolve(int(tid)) if fm_resolve else None)
                    if k and val > 0:
                        out[k] = val
        return np.log(out).to_dict()
