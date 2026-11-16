"""Uctan uca test: bot komutlarini sahte Telegram nesneleriyle calistirir (gercek mesaj gondermez),
motorun temel ozelliklerini ve veri akisini dogrular.

  .venv/bin/python -m tests.test_bot
"""
import asyncio
import json
import re
import sqlite3
import time
import traceback

import numpy as np

from bot import telegram_bot as tb
from bot import views

SENT = []


class Chat:
    id = -1
    full_name = "test"
    title = None

    async def send_message(self, text, **kw):
        SENT.append(text)


class User:
    id = 999999
    first_name = "Test"
    username = "test"


class Query:
    def __init__(self, data):
        self.data = data

    async def answer(self):
        pass


class Upd:
    def __init__(self, data=None):
        self.effective_chat = Chat()
        self.effective_user = User()
        self.callback_query = Query(data) if data else None


class Ctx:
    def __init__(self, *args):
        self.args = list(args)


def html_ok(t):
    """Telegram HTML: yalnizca izinli etiketler, acilan kapanmali."""
    tags = re.findall(r"</?(\w+)[^>]*>", t)
    allowed = {"b", "i", "u", "s", "a", "code", "pre"}
    if set(tags) - allowed:
        return False
    for tag in allowed:
        if len(re.findall(fr"<{tag}[ >]", t)) != len(re.findall(fr"</{tag}>", t)):
            return False
    return len(t) <= 4096


RESULTS = []


def check(name, cond, info=""):
    RESULTS.append((name, bool(cond), info))
    print(("OK  " if cond else "HATA") + f" {name} {info}", flush=True)


async def run_cmd(name, fn, *args, data=None):
    SENT.clear()
    t = time.time()
    try:
        await fn(Upd(data), Ctx(*args))
        ok = bool(SENT) and all(html_ok(s) for s in SENT)
        check(f"/{name} {' '.join(args)}", ok, f"{len(SENT)} mesaj, {time.time() - t:.1f} sn")
    except Exception:
        check(f"/{name} {' '.join(args)}", False, traceback.format_exc(limit=3))
    return list(SENT)


def engine_tests():
    from sim import engine
    from sim.common import load_params
    dyn = load_params()["dynamics"]
    h, a = engine.Side("h", mu=1.6), engine.Side("a", mu=1.1)
    t = time.time()
    r = engine.simulate(h, a, dyn, n=20000, seed=1)
    dt = time.time() - t
    s = engine.summarize(r, h, a)
    check("motor: 20k simulasyon < 1.5 sn", dt < 1.5, f"{dt:.2f} sn")
    check("motor: olasiliklar toplami 1", abs(s["p_home"] + s["p_draw"] + s["p_away"] - 1) < 1e-9)
    check("motor: beklenen gol mu'ya yakin", abs(s["exp_goals"][0] - 1.6) < 0.15 and abs(s["exp_goals"][1] - 1.1) < 0.15,
          str(s["exp_goals"]))
    check("motor: guclu taraf favori", s["p_home"] > s["p_away"])
    r2 = engine.simulate(engine.Side("h", mu=1.3), engine.Side("a", mu=1.3), dyn, n=20000, seed=2)
    s2 = engine.summarize(r2, h, a)
    check("motor: esit takimlarda simetri", abs(s2["p_home"] - s2["p_away"]) < 0.02, f"{s2['p_home']:.3f}/{s2['p_away']:.3f}")
    check("motor: beraberlik makul (%22-%32)", 0.22 < s2["p_draw"] < 0.32, f"{s2['p_draw']:.3f}")
    check("motor: kirmizi kart orani makul (%10-%30)", 0.10 < s2["p_red"] < 0.30, f"{s2['p_red']:.3f}")
    same = engine.summarize(engine.simulate(h, a, dyn, n=5000, seed=9), h, a)
    same2 = engine.summarize(engine.simulate(h, a, dyn, n=5000, seed=9), h, a)
    check("motor: ayni tohum ayni sonuc", same["p_home"] == same2["p_home"])


def data_tests():
    con = sqlite3.connect("data/sim.sqlite")
    n = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    check("veri: tahmin tablosu dolu", n > 0, f"{n} satir")
    rows = con.execute("SELECT data FROM predictions ORDER BY created_at DESC LIMIT 50").fetchall()
    bad = 0
    for (d,) in rows:
        o = json.loads(d)
        ps = [o["p_home"], o["p_draw"], o["p_away"]]
        if not (abs(sum(ps) - 1) < 1e-6 and all(np.isfinite(ps)) and 0 < o["exp_goals"][0] < 6):
            bad += 1
    check("veri: tahminler gecerli (toplam 1, sonlu)", bad == 0, f"{bad} hatali")
    from sim.common import load_params
    p = load_params()
    check("parametre: hoca etkileri kapali (backtest karari)", p.get("use_coach_plan") is False and p.get("use_coach_state") is False)


async def bot_tests(day):
    ps = views.day_predictions(day)
    check("bot: gunun tahminleri okunuyor", len(ps) > 0, f"{len(ps)} mac")
    orig_ymd = tb.ymd
    tb.ymd = lambda offset=0: (__import__("datetime").datetime.strptime(day, "%Y%m%d")
                               + __import__("datetime").timedelta(days=offset)).strftime("%Y%m%d")
    try:
        await run_cmd("start", tb.cmd_start)
        await run_cmd("bugun", tb.cmd_today)
        await run_cmd("yarin", tb.cmd_tomorrow)
        await run_cmd("lig", tb.cmd_league, "GER")
        await run_cmd("lig", tb.cmd_league, "XYZ")
        if ps:
            # yorumu hazir olan bir mac sec (yoksa ilk mac; yorum o an uretilir, yavas olabilir)
            con = sqlite3.connect("data/sim.sqlite")
            have = {r[0] for r in con.execute("SELECT match_id FROM commentary")}
            pick = next((o for o in ps if o["match_id"] in have), ps[0])
            team = pick["home_name"]
            out = await run_cmd("mac", tb.cmd_match, team)
            check("bot: /mac karti yorum iceriyor", any("🧠" in s for s in out))
            await run_cmd("mac", tb.cmd_match, "olmayantakimxyz")
            await run_cmd("buton", tb.on_button, data=f"m:{pick['match_id']}")
            await run_cmd("simule", tb.cmd_simulate, team)
            await run_cmd("tahmin", tb.cmd_guess, team, "2-1")
        await run_cmd("surpriz", tb.cmd_upsets)
        await run_cmd("golcu", tb.cmd_scorers)
        await run_cmd("kupon", tb.cmd_coupon)
        await run_cmd("istatistik", tb.cmd_stats)
        await run_cmd("puan", tb.cmd_board)
        await run_cmd("rapor", tb.cmd_report)
        await run_cmd("rapor", tb.cmd_report, "hafta")
    finally:
        tb.ymd = orig_ymd
        con = tb.db()
        con.execute("DELETE FROM guesses WHERE user_id = ?", (User.id,))  # yalnizca testin kendi kaydi
        con.commit()


def main(day="20261024"):
    engine_tests()
    data_tests()
    asyncio.run(bot_tests(day))
    bad = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} test gecti")
    for r in bad:
        print("  HATA:", r[0], r[2][:300])


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
