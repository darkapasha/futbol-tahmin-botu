"""Bir maci simulasyona hazirlar: puanlar + kadro + hoca planlari + takim ozellikleri -> engine.Side.

  p = Predictor("2026-10-22")
  out = p.predict("TUR", "tm:141", "tm:36")            # Galatasaray - Fenerbahce
"""
import numpy as np
import pandas as pd

from . import engine, hist, players, ratings
from .common import CUPS, TeamResolver, load_params, norm, prof_con, sim_con, src_con

CAP = (0.7, 1.4)


class Predictor:
    def __init__(self, asof=None, games=None, with_players=True, params=None, coach_state=None, coach_plan=None):
        self.params = params or load_params()
        self.con = src_con()
        self.resolver = TeamResolver(self.con)
        self.g = games if games is not None else hist.games(self.resolver)
        self.dyn = self.params["dynamics"]
        sc = sim_con()
        cs = coach_state if coach_state is not None else pd.read_sql("SELECT * FROM coach_state", sc)
        cs = cs.assign(n=cs.coach.map(norm))
        self.cstate = cs.drop_duplicates("n").set_index("n")
        cp = coach_plan if coach_plan is not None else pd.read_sql("SELECT * FROM coach_plan", sc)
        cp = cp.assign(cn=cp.coach.map(norm))
        self.cplan = (cp.sort_values("n", ascending=False).drop_duplicates(["cn", "opp_class"])
                      .set_index(["cn", "opp_class"]).sort_index())
        self.with_players = with_players
        self.use_coach = self.params.get("use_coach", True)
        self.use_coach_state = self.params.get("use_coach_state", True)
        self.use_coach_plan = self.params.get("use_coach_plan", True)
        pc = prof_con()
        self.team = pd.read_sql("SELECT * FROM team_current", pc).set_index("team_key")
        lg = pd.read_sql("SELECT * FROM league_season", pc)
        # yeni baslamis sezon (birkac mac) lig ortalamasi icin guvenilmez: en az 150 maclik son sezon
        self.lg = lg[lg.matches >= 150].sort_values("season").groupby("league").last()
        cprof = pd.read_sql("SELECT coach, mean_minute_trail FROM coach_profile", pc)
        cprof["n"] = cprof.coach.map(norm)
        self.cprof = cprof.drop_duplicates("n").set_index("n")
        tmap = pd.read_sql("SELECT name, team_key FROM team_map WHERE source='fotmob'", self.con)
        self.fm_of = {k: int(n) for n, k in zip(tmap.name, tmap.team_key) if str(n).isdigit()}
        self.set_date(asof or pd.Timestamp.now().normalize())

    def set_date(self, asof):
        """Puanlari ve oyuncu defterini bu tarihe kadarki veriyle yeniden kurar."""
        self.asof = pd.Timestamp(asof)
        adj = self.params.get("adjust", {})
        boost = {t: v[0] for t, v in adj.get("new_coach", {}).items()}
        vals = None
        if self.params.get("ratings", {}).get("val_att") or self.params.get("ratings", {}).get("val_def"):
            from .rolling import value_book
            vals = value_book().at(self.asof, lambda tid: self.resolver.by_src.get(("fotmob", str(tid))))
        self.r = ratings.fit(self.g, self.asof, self.params.get("ratings"), values=vals, boost=boost)
        self.book = players.PlayerBook(self.asof, self.con) if self.with_players else None
        self._elo = None
        self._market = self._refs = None

    # ------------------------------------------------------------ yardimcilar
    def coach_of(self, team_key, fm_id=None):
        """En guncel kaynaktaki hoca: FotMob son maci ya da Transfermarkt/profil (hangisi yeniyse)."""
        fm_id = fm_id or self.fm_of.get(team_key)
        best = (None, "")
        if fm_id:
            r = self.con.execute("SELECT CASE WHEN home_id=? THEN home_coach ELSE away_coach END, utc_time FROM fm_matches "
                                 "WHERE (home_id=? OR away_id=?) AND utc_time < ? ORDER BY utc_time DESC LIMIT 1",
                                 (fm_id, fm_id, fm_id, self.asof.strftime("%Y-%m-%dT%H:%M"))).fetchone()
            if r and r[0]:
                best = (r[0], r[1][:10])
        if team_key in self.team.index:
            t = self.team.loc[team_key]
            if pd.notna(t.coach_last) and str(t.last_match)[:10] > best[1]:
                best = (t.coach_last, str(t.last_match)[:10])
        return best[0]

    def lineup(self, team_key, fm_id=None, match_id=None, unavailable=(), pre_rows=None, confirmed=False):
        """Oncelik: FotMob mac oncesi sayfasi (muhtemel / aciklanan 11). Yoksa FotMob ve
        Transfermarkt'tan hangisinin son kadrosu daha yeniyse o."""
        if self.book is None:
            return None
        fm_id = fm_id or self.fm_of.get(team_key)
        tm_club = int(team_key[3:]) if team_key.startswith("tm:") else None
        if pre_rows is not None and fm_id:
            lu = players.lineup_from_prematch(self.book, pre_rows, fm_id, tm_club, self.asof, confirmed)
            if lu:
                return lu
        cands = []
        if fm_id:
            lu = players.lineup_from_fotmob(self.book, fm_id, tm_club, self.asof, match_id, unavailable)
            if lu:
                cands.append(lu)
        if tm_club:
            lu = players.lineup_from_tm(self.book, tm_club, self.asof)
            if lu:
                cands.append(lu)
        if not cands:
            return None
        return max(cands, key=lambda l: (l["source"].endswith("aciklanan"), l["date"]))

    def _team_feat(self, team_key, league):
        lg = self.lg.loc[league] if league in self.lg.index else self.lg.loc["ENG"]
        t = self.team.loc[team_key] if team_key in self.team.index else None

        def f(col, default):
            v = t[col] if t is not None and col in t and pd.notna(t[col]) else (lg[col] if col in lg and pd.notna(lg[col]) else default)
            return float(v)
        xps = f("xg_per_shot", 0.11)
        gf = f("gf_pg", 1.4)
        pen = f("pen_for_pg", 0.12) / max(gf, 0.5)
        red = f("red_pg", 0.1) / max(float(lg["red_pg"]) if pd.notna(lg.get("red_pg")) else 0.1, 0.02)
        return dict(xg_per_shot=float(np.clip(xps, 0.07, 0.2)), pen_share=float(np.clip(pen, 0.03, 0.2)), red_factor=float(np.clip(red, 0.5, 2.0)))

    def _coach(self, name, opp_class):
        out = dict(plan_p=0.0, plan_att=1.0, plan_dfn=1.0, att_trail=1.0, dfn_trail=1.0, att_lead=1.0,
                   dfn_lead=1.0, sub_minute=62.0)
        if not name or not self.use_coach:
            return out
        n = norm(name)
        if self.use_coach_state and n in self.cstate.index:
            c = self.cstate.loc[n]
            for k in ("att_trail", "dfn_trail", "att_lead", "dfn_lead"):
                if pd.notna(c.get(k)):
                    out[k] = float(np.clip(c[k], *CAP))
        if self.use_coach_plan and (n, opp_class) in self.cplan.index:
            p = self.cplan.loc[(n, opp_class)]
            q = float(p.p_plan)
            # planin ortalama etkisi = olasilik x plan etkisi -> plan etkisi = ortalama / olasilik
            # (dusuk olasilikta asiri buyumesin diye en fazla 2 kat)
            amp = 1 / max(q, 0.5)
            out.update(plan_p=q, plan_att=float(np.clip(1 + (p.att - 1) * amp, *CAP)),
                       plan_dfn=float(np.clip(1 + (p.dfn - 1) * amp, *CAP)))
        if n in self.cprof.index and pd.notna(self.cprof.at[n, "mean_minute_trail"]):
            out["sub_minute"] = float(self.cprof.at[n, "mean_minute_trail"])
        return out

    def _cup_team(self, key, name, fm_id=None, pre_rows=None):
        """Avrupa kupasinda verimiz az olan rakip (orn. Ararat, Ferencvaros): ClubElo ve ilk 11 degerinden puan."""
        if key in self.r.att and self.r.weight.get(key, 0) >= 5:
            return None
        if self._elo is None:
            from .elo_prior import EloPrior
            self._elo = EloPrior(self.r, self.asof, self.con)
            self._elo.fit_value(self.r, self.resolver, self.con)
        xi = None
        if pre_rows is not None and fm_id is not None:
            st = pre_rows[(pre_rows.team_id == fm_id) & (pre_rows.role == "starter")]
            xi = float(st.market_value.fillna(0).sum()) if len(st) >= 11 else None
        out = self._elo.rating(name or key, xi)
        if out is None:
            return None
        a, d, src = out
        self.r.att[key], self.r.dfn[key] = a, d
        self.r.weight.setdefault(key, 0.0)
        return src

    # ------------------------------------------------------------ tahmin
    def setup(self, league, home, away, fm_home=None, fm_away=None, match_id=None, neutral=False,
              home_coach=None, away_coach=None, unavailable=(), use_lineups=True, pre_rows=None, confirmed=False,
              home_name=None, away_name=None, referee=None):
        info = {}
        if league in CUPS:
            info["cup_prior"] = [self._cup_team(home, home_name, fm_home, pre_rows),
                                 self._cup_team(away, away_name, fm_away, pre_rows)]
        mu_h, mu_a = self.r.rate(home, away, league, neutral)
        net_h, net_a = self.r.net(home, league), self.r.net(away, league)
        t = self.params.get("class_t", 0.2)
        cls = lambda d: "stronger" if d > t else ("weaker" if d < -t else "similar")
        info["mu_base"] = [mu_h, mu_a]
        lus = [None, None]
        if use_lineups:
            lus = [self.lineup(home, fm_home, match_id, unavailable, pre_rows, confirmed),
                   self.lineup(away, fm_away, match_id, unavailable, pre_rows, confirmed)]
            beta = self.params.get("lineup_beta", 0.1)
            rh, ra = (players.strength_ratio(l, self.asof) for l in lus)
            mu_h *= (rh / ra) ** beta
            mu_a *= (ra / rh) ** beta
            info["lineup_ratio"] = [rh, ra]
            if self.params.get("use_flank_matchups", True):
                info["flank"] = players.flank_adjust(lus[0], lus[1], league) + players.flank_adjust(lus[1], lus[0], league)
        coaches = [home_coach or self.coach_of(home, fm_home), away_coach or self.coach_of(away, fm_away)]
        sides = []
        for i, (key, mu, d, lu) in enumerate(((home, mu_h, net_a - net_h, lus[0]), (away, mu_a, net_h - net_a, lus[1]))):
            s = engine.Side(name=key, mu=mu, **self._team_feat(key, league), **self._coach(coaches[i], cls(d)))
            s.scorers = players.scorer_table(lu) if lu else None
            s.info = dict(coach=coaches[i], opp_class=cls(d), lineup_source=lu["source"] if lu else None,
                          missing=[m["name"] for m in lu["missing"]] if lu else [],
                          xi=[p["name"] for p in lu["xi"]] if lu else [])
            sides.append(s)
        # ogrenme dongusunun ayarlari (yalnizca kalici hata goruldugunde konur)
        adj = self.params.get("adjust", {})
        # goal_scale_global: iki sezonluk backtest'te motor mac basina ~0.06 gol fazla veriyordu (2.88 / 2.81)
        gs = adj.get("goal_scale", {}).get(league, 1.0) * self.params.get("goal_scale_global", 1.0)
        hf = adj.get("hfa", {}).get(league, 0.0) if not neutral else 0.0
        ta = adj.get("team", {})
        mu_h *= gs * np.exp(hf) * ta.get(home, {}).get("att", 1.0) * ta.get(away, {}).get("def", 1.0)
        mu_a *= gs * np.exp(-hf) * ta.get(away, {}).get("att", 1.0) * ta.get(home, {}).get("def", 1.0)
        sides[0].mu, sides[1].mu = mu_h, mu_a
        info["mu"] = [mu_h, mu_a]
        if referee and self.params.get("use_referee", True):
            if self._refs is None:
                from .market import Referees
                self._refs = Referees(self.con)
            f, rate = self._refs.factor(referee)
            for s in sides:
                s.red_factor *= f
            info["referee"] = dict(name=referee, red_rate=rate, factor=round(f, 2))
        return sides[0], sides[1], info

    def predict(self, league, home, away, n=10000, seed=None, date=None, **kw):
        kw_date = date
        h, a, info = self.setup(league, home, away, **kw)
        kw = {"date": kw_date}
        dyn = self.dyn
        lf = self.params.get("adjust", {}).get("level_factor", 1.0)
        if lf != 1.0:
            dyn = {**dyn, "state": [v * lf if i in (4, 5) else v for i, v in enumerate(dyn["state"])]}
        res = engine.simulate(h, a, dyn, n=n, seed=seed)
        out = engine.summarize(res, h, a)
        # piyasa harmani: oran varsa hedef = %80 piyasa + %20 motor; mu'lar kaydirilip yeniden simule edilir
        if self.params.get("use_market", True) and league not in CUPS:
            from . import market
            if self._market is None:
                self._market = market.Market(self.con)
            mk = self._market.odds(home, away, kw.get("date") or self.asof)
            if mk:
                w = self.params.get("market_w", market.MARKET_W)
                eng = [out["p_home"], out["p_draw"], out["p_away"]]
                tgt = market.temper([w * m + (1 - w) * e for m, e in zip(mk[:3], eng)],
                                    self.params.get("market_temp", market.TEMP))
                tou = None if mk[3] is None else w * mk[3] + (1 - w) * out["over"]["2.5"]
                mh, ma = market.fit_mu(h.mu, a.mu, eng, out["over"]["2.5"], tgt, tou)
                info["motor"] = dict(p=eng, over25=out["over"]["2.5"], mu=[h.mu, a.mu])
                info["piyasa"] = dict(p=list(mk[:3]), over25=mk[3], agirlik=w)
                h.mu, a.mu = mh, ma
                res = engine.simulate(h, a, dyn, n=n, seed=seed)
                out = engine.summarize(res, h, a)
                info["mu"] = [mh, ma]
        out.update(info=info, home=h.info, away=a.info, league=league, home_key=home, away_key=away)
        return out

    def story(self, league, home, away, seed=None, **kw):
        """Eglence: maci bir kez oynat, dakika dakika olaylari dondur."""
        h, a, info = self.setup(league, home, away, **kw)
        res = engine.simulate(h, a, self.dyn, n=1, seed=seed)
        _, events, score = engine.story(res, h, a, sim=0, seed=seed)
        return events, score, res["shots"][0].tolist(), res["xg"][0].round(2).tolist()
