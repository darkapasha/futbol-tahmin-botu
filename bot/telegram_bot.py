"""Futbol tahmin Telegram botu (python-telegram-bot).

  .venv/bin/python -m bot.telegram_bot          # systemd: futbol-bot.service

Komutlar: /bugun /yarin /mac <takim> /lig <kod> /istatistik /abone /dur /yardim
Eglence: /simule <takim> /surpriz /golcu /kupon /tahmin <takim> <skor> /puan
Zamanli isler: her sabah 09:30 (TR) abonelere gunun ozeti; her 15 dk aciklanan 11'lerle
tahmin tazeleme ve olasilik belirgin degistiyse abonelere bildirim; pazartesi 10:00 haftalik rapor.
Sabah ozeti kisa; her mac icin "detay" butonu yorumlu karti acar.
Abone listesi data/bot.sqlite'ta; bot kimseye kendiliginden yazmaz, yalnizca /abone olanlara.
"""
import asyncio
import logging
import sqlite3
from datetime import datetime, time, timedelta, timezone

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

from config import DATA, ROOT

from . import commentary, report, views

TR = timezone(timedelta(hours=3))
BOT_DB = DATA / "bot.sqlite"
log = logging.getLogger("futbol-bot")


def token():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("TELEGRAM_BOT_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("TELEGRAM_BOT_TOKEN .env icinde yok")


def db():
    con = sqlite3.connect(BOT_DB)
    con.execute("CREATE TABLE IF NOT EXISTS subscribers (chat_id INTEGER PRIMARY KEY, name TEXT, since TEXT, active INTEGER)")
    con.execute("CREATE TABLE IF NOT EXISTS sent (key TEXT PRIMARY KEY, at TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS guesses (user_id INTEGER, name TEXT, match_id INTEGER, hg INTEGER, "
                "ag INTEGER, at TEXT, PRIMARY KEY (user_id, match_id))")
    return con


def subscribers():
    return [r[0] for r in db().execute("SELECT chat_id FROM subscribers WHERE active = 1")]


def ymd(offset=0):
    return (datetime.now(TR) + timedelta(days=offset)).strftime("%Y%m%d")


HELP = (
    "⚽ <b>Futbol Tahmin Botu</b>\n"
    "Her maçı taktik verilerle 10.000 kez simüle edip olasılıkları çıkarırım.\n\n"
    "/bugun – bugünün maçları\n/yarin – yarının maçları\n/mac takım – tek maç detayı ve yorum\n"
    "/lig TUR|ENG|ESP|ITA|GER|FRA|UCL|UEL – tek lig\n/istatistik – tahmin karnem\n"
    "/rapor – dünün tahmin raporu (/rapor hafta, /rapor ay)\n"
    "/abone – her sabah özet gönderilsin\n/dur – aboneliği bitir\n\n"
    "🎲 <b>Eğlence</b>\n/simule takım – maçı bir kez oynat, dakika dakika izle\n/surpriz – sürpriz alarmı\n"
    "/golcu – bugünün gol adayları\n/kupon – günün üçlüsü\n/tahmin takım 2-1 – skor tahminin (botla yarış)\n"
    "/puan – tahmin ligi puan durumu\n\n"
    "<i>Olasılıklar eğlence ve bilgi amaçlıdır, bahis tavsiyesi değildir.</i>"
)


async def send_long(chat, text, **kw):
    """Telegram 4096 karakter siniri: satir sonlarindan bol."""
    while text:
        if len(text) <= 4000:
            await chat.send_message(text, parse_mode=ParseMode.HTML, disable_web_page_preview=True, **kw)
            return
        cut = text.rfind("\n", 0, 4000)
        await chat.send_message(text[:cut], parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        text = text[cut + 1:]


def match_buttons(ps):
    rows = [[InlineKeyboardButton(f"{o['home_name']} – {o['away_name']}"[:60], callback_data=f"m:{o['match_id']}")]
            for o in ps[:30]]
    return InlineKeyboardMarkup(rows) if rows else None


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_chat.send_message(HELP, parse_mode=ParseMode.HTML)


async def cmd_day(update: Update, ctx, offset=0):
    d = ymd(offset)
    ps = views.day_predictions(d)
    await send_long(update.effective_chat, views.day_summary(d))
    if ps:
        await update.effective_chat.send_message("Detay için maça dokun:", reply_markup=match_buttons(ps))


async def cmd_today(update, ctx):
    await cmd_day(update, ctx, 0)


async def cmd_tomorrow(update, ctx):
    await cmd_day(update, ctx, 1)


async def cmd_league(update, ctx):
    code = (ctx.args[0].upper() if ctx.args else "")
    if code not in views.LEAGUE_TR:
        await update.effective_chat.send_message("Lig kodu: TUR, ENG, ESP, ITA, GER, FRA, UCL, UEL. Örnek: /lig UEL")
        return
    await send_long(update.effective_chat, views.day_summary(ymd(0), code))


async def show_match(chat, o):
    txt = None
    try:
        r = sqlite3.connect(DATA / "sim.sqlite").execute(
            "SELECT text FROM commentary WHERE match_id = ? ORDER BY created_at DESC LIMIT 1", (o["match_id"],)).fetchone()
        txt = r[0] if r else None
    except sqlite3.Error:
        pass
    if txt is None:
        await chat.send_message("🧠 Yorum hazırlanıyor…")
        txt = await asyncio.to_thread(commentary.get, o, datetime.now(timezone.utc).isoformat(timespec="seconds"))
    await send_long(chat, views.match_card(o, txt))


async def cmd_match(update, ctx):
    q = " ".join(ctx.args)
    if not q:
        await update.effective_chat.send_message("Örnek: /mac Galatasaray")
        return
    ps = views.day_predictions(ymd(0)) + views.day_predictions(ymd(1)) + views.day_predictions(ymd(2))
    o = views.find_match(ps, q)
    if not o:
        await update.effective_chat.send_message(f"Önümüzdeki 3 günde “{q}” maçı bulamadım.")
        return
    await show_match(update.effective_chat, o)


async def on_button(update: Update, ctx):
    qd = update.callback_query
    await qd.answer()
    mid = int(qd.data.split(":")[1])
    from sim.run import latest
    o = latest(mid)
    if o:
        await show_match(update.effective_chat, o)


async def cmd_report(update, ctx):
    """/rapor -> dun; /rapor hafta | ay"""
    arg = (ctx.args[0].lower() if ctx.args else "")
    if arg.startswith("hafta"):
        t = await asyncio.to_thread(report.range_text, 7, "Son 7 gün")
    elif arg.startswith("ay"):
        t = await asyncio.to_thread(report.range_text, 30, "Son 30 gün")
    else:
        t = await asyncio.to_thread(report.day_text, ymd(-1))
    await send_long(update.effective_chat, t)


async def job_health(ctx: ContextTypes.DEFAULT_TYPE):
    """Sabah 09:20: veri kaynaklarindan biri kesildiyse abonelere kisa uyari (gunde bir kez)."""
    from . import health
    probs = await asyncio.to_thread(health.check)
    if not probs:
        return
    con = db()
    key = f"health:{ymd(0)}"
    if con.execute("SELECT 1 FROM sent WHERE key = ?", (key,)).fetchone():
        return
    # Ayni sorun listesi 7 gun icinde zaten gittiyse her sabah tekrar etme
    import hashlib
    hkey = "health:ayni:" + hashlib.md5("|".join(sorted(probs)).encode()).hexdigest()[:12]
    row = con.execute("SELECT at FROM sent WHERE key = ?", (hkey,)).fetchone()
    if row and datetime.fromisoformat(row[0]) > datetime.now(TR) - timedelta(days=7):
        return
    con.execute("INSERT OR REPLACE INTO sent VALUES (?,?)", (hkey, datetime.now(TR).isoformat()))
    text = ("⚠️ <b>Veri uyarısı</b>\n" + "\n".join(f"• {views.e(p)}" for p in probs) +
            "\n<i>Tahminler bu yüzden eksik ya da daha az isabetli olabilir.</i>")
    for chat_id in subscribers():
        try:
            await ctx.bot.send_message(chat_id, text, parse_mode=ParseMode.HTML)
        except Exception as ex:
            log.warning("gonderilemedi %s: %s", chat_id, ex)
    con.execute("INSERT INTO sent VALUES (?,?)", (key, datetime.now(TR).isoformat()))
    con.commit()


async def job_weekly(ctx: ContextTypes.DEFAULT_TYPE):
    t = await asyncio.to_thread(report.range_text, 7, "Haftalık rapor (son 7 gün)")
    for chat_id in subscribers():
        try:
            await ctx.bot.send_message(chat_id, t, parse_mode=ParseMode.HTML)
        except Exception as ex:
            log.warning("gonderilemedi %s: %s", chat_id, ex)


async def cmd_stats(update, ctx):
    await send_long(update.effective_chat, views.stats_text())


async def cmd_sub(update, ctx):
    c = update.effective_chat
    con = db()
    con.execute("INSERT OR REPLACE INTO subscribers VALUES (?,?,?,1)",
                (c.id, c.full_name or c.title or "", datetime.now(TR).isoformat(timespec="minutes")))
    con.commit()
    await c.send_message("✅ Abone oldun. Her sabah 09:30'da günün tahminlerini göndereceğim. Bitirmek için /dur")


async def cmd_stop(update, ctx):
    con = db()
    con.execute("UPDATE subscribers SET active = 0 WHERE chat_id = ?", (update.effective_chat.id,))
    con.commit()
    await update.effective_chat.send_message("Abonelik bitti. Komutlarla yine sorabilirsin.")


# ------------------------------------------------------------ eglence
def upcoming():
    return views.day_predictions(ymd(0)) + views.day_predictions(ymd(1)) + views.day_predictions(ymd(2))


async def cmd_simulate(update, ctx):
    q = " ".join(ctx.args)
    o = views.find_match(upcoming(), q) if q else None
    if not o:
        await update.effective_chat.send_message("Örnek: /simule Fenerbahçe (önümüzdeki 3 günün maçlarından)")
        return

    def run():
        from sim.match import Predictor
        global _PRED
        if _PRED is None or (datetime.now() - _PRED[1]).total_seconds() > 6 * 3600:
            _PRED = (Predictor(), datetime.now())
        return _PRED[0].story(o["league"], o["home_key"], o["away_key"], seed=None,
                              home_coach=o["home"].get("coach"), away_coach=o["away"].get("coach"),
                              fm_home=None, fm_away=None)
    await update.effective_chat.send_message("🎬 Maç oynanıyor…")
    events, score, shots, xg = await asyncio.to_thread(run)
    await send_long(update.effective_chat, views.story_text(o, events, score, shots, xg))


_PRED = None


async def cmd_upsets(update, ctx):
    await send_long(update.effective_chat, views.upsets(views.day_predictions(ymd(0))))


async def cmd_scorers(update, ctx):
    await send_long(update.effective_chat, views.top_scorers(views.day_predictions(ymd(0))))


async def cmd_coupon(update, ctx):
    await send_long(update.effective_chat, views.coupon(views.day_predictions(ymd(0))))


async def cmd_guess(update, ctx):
    import re
    txt = " ".join(ctx.args)
    m = re.search(r"(\d+)\s*[-:]\s*(\d+)\s*$", txt)
    if not m:
        await update.effective_chat.send_message("Örnek: /tahmin Galatasaray 2-1")
        return
    o = views.find_match(upcoming(), txt[:m.start()].strip())
    if not o:
        await update.effective_chat.send_message("Önümüzdeki 3 günde bu takımın maçını bulamadım.")
        return
    if o.get("utc_time") and datetime.now(timezone.utc) >= datetime.fromisoformat(o["utc_time"].replace("Z", "+00:00")):
        await update.effective_chat.send_message("Bu maç başladı, tahmin kapandı.")
        return
    u = update.effective_user
    con = db()
    con.execute("INSERT OR REPLACE INTO guesses VALUES (?,?,?,?,?,?)",
                (u.id, u.first_name or u.username or "?", o["match_id"], int(m.group(1)), int(m.group(2)),
                 datetime.now(TR).isoformat(timespec="minutes")))
    con.commit()
    await update.effective_chat.send_message(
        f"📝 Tahminin kaydedildi: {o['home_name']} {m.group(1)}-{m.group(2)} {o['away_name']}\n"
        f"🤖 Benim en olası skorum: {o['scores'][0][0]}")


def leaderboard():
    """Tam skor 3 puan, sonuc (1/X/2) 1 puan. Bot kendi en olasi skoruyla yarisir."""
    import pandas as pd
    con = db()
    g = pd.read_sql("SELECT * FROM guesses", con)
    if g.empty:
        return "Henüz kimse tahmin yapmadı. /tahmin takım 2-1 ile başla!"
    from sim.common import src_con
    res = pd.read_sql("SELECT match_id, home_goals rh, away_goals ra FROM fm_matches WHERE home_goals IS NOT NULL", src_con())
    g = g.merge(res, on="match_id")
    if g.empty:
        return "Tahmin edilen maçlar henüz bitmedi."
    from sim.run import latest
    bot_rows = []
    for mid in g.match_id.unique():
        o = latest(int(mid))
        if o:
            h, a = map(int, o["scores"][0][0].split("-"))
            bot_rows.append(dict(user_id=0, name="🤖 Bot", match_id=mid, hg=h, ag=a))
    b = pd.DataFrame(bot_rows).merge(res, on="match_id") if bot_rows else pd.DataFrame()
    allg = pd.concat([g, b], ignore_index=True)
    sgn = lambda x, y: (x > y).astype(int) - (x < y).astype(int)
    exact = (allg.hg == allg.rh) & (allg.ag == allg.ra)
    allg["pts"] = exact * 3 + (~exact & (sgn(allg.hg, allg.ag) == sgn(allg.rh, allg.ra))) * 1
    t = allg.groupby(["user_id", "name"]).agg(puan=("pts", "sum"), mac=("pts", "size")).reset_index().sort_values("puan", ascending=False)
    lines = ["🏆 <b>Tahmin ligi</b> (tam skor 3, sonuç 1 puan)"]
    for i, r in enumerate(t.itertuples(), 1):
        lines.append(f"{i}. {views.e(r.name)} – {r.puan} puan ({r.mac} maç)")
    return "\n".join(lines)


async def cmd_board(update, ctx):
    await send_long(update.effective_chat, await asyncio.to_thread(leaderboard))


# ------------------------------------------------------------ zamanli isler
async def job_morning(ctx: ContextTypes.DEFAULT_TYPE):
    d = ymd(0)
    key = f"morning:{d}"
    con = db()
    if con.execute("SELECT 1 FROM sent WHERE key = ?", (key,)).fetchone():
        return
    ps = views.day_predictions(d)
    text = views.day_summary(d)
    if not ps:
        nxt = next(((k, views.day_predictions(ymd(k))) for k in (1, 2) if views.day_predictions(ymd(k))), None)
        text = "☕ Günaydın! Bugün takip ettiğim 7 ligde (Süper Lig, ŞL, Premier Lig, La Liga, Serie A, Bundesliga, Ligue 1) maç yok."
        if nxt:
            text += f"\nSıradaki maç günü {'yarın' if nxt[0] == 1 else 'öbür gün'}: {len(nxt[1])} maç. Tahminler için /yarin"
    try:
        await asyncio.to_thread(report.save_day, ymd(-1))
    except Exception as ex:
        log.warning("rapor kaydedilemedi: %s", ex)
    recap = views.yesterday_recap(ymd(-1))
    if recap:
        recap += " Ayrıntı: /rapor"
        text = recap + "\n\n" + text
    for chat_id in subscribers():
        try:
            chat = await ctx.bot.get_chat(chat_id)
            await send_long(chat, text)
            if ps:
                await chat.send_message("Detay ve yorum için maça dokun:", reply_markup=match_buttons(ps))
        except Exception as ex:  # engelleyen kullanici vs.
            log.warning("gonderilemedi %s: %s", chat_id, ex)
    con.execute("INSERT INTO sent VALUES (?,?)", (key, datetime.now(TR).isoformat()))
    con.commit()


async def job_lineups(ctx: ContextTypes.DEFAULT_TYPE):
    from . import pipeline
    try:
        changed = await asyncio.to_thread(pipeline.lineups)
    except Exception as ex:
        log.warning("kadro tazeleme hatasi: %s", ex)
        return
    for new, old, d in changed:
        if d < 0.05:
            continue
        txt = (f"📋 <b>Maç öncesi güncelleme: {views.e(new['home_name'])} – {views.e(new['away_name'])}</b>\n"
               f"İlk 11 ve son oranlarla:1 {views.pct(old['p_home'])}→{views.pct(new['p_home'])} · "
               f"X {views.pct(old['p_draw'])}→{views.pct(new['p_draw'])} · 2 {views.pct(old['p_away'])}→{views.pct(new['p_away'])}")
        for chat_id in subscribers():
            try:
                await ctx.bot.send_message(chat_id, txt, parse_mode=ParseMode.HTML)
            except Exception as ex:
                log.warning("gonderilemedi %s: %s", chat_id, ex)


COMMANDS = [("bugun", "Bugünün maç tahminleri"), ("yarin", "Yarının maç tahminleri"),
            ("mac", "Tek maç detayı ve yorum: /mac takım"), ("lig", "Tek lig: /lig TUR"),
            ("istatistik", "Tahmin karnem"), ("rapor", "Dünün raporu (/rapor hafta, /rapor ay)"), ("simule", "Maçı bir kez oynat: /simule takım"),
            ("surpriz", "Sürpriz alarmı"), ("golcu", "Bugünün gol adayları"), ("kupon", "Günün üçlüsü"),
            ("tahmin", "Skor tahminin: /tahmin takım 2-1"), ("puan", "Tahmin ligi"),
            ("abone", "Her sabah özet al"), ("dur", "Aboneliği bitir"), ("yardim", "Yardım")]


async def post_init(app):
    from telegram import BotCommand
    await app.bot.set_my_commands([BotCommand(c, d) for c, d in COMMANDS])


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    app = Application.builder().token(token()).post_init(post_init).build()
    for name, fn in (("start", cmd_start), ("yardim", cmd_start), ("help", cmd_start), ("bugun", cmd_today),
                     ("yarin", cmd_tomorrow), ("mac", cmd_match), ("lig", cmd_league), ("istatistik", cmd_stats), ("rapor", cmd_report),
                     ("abone", cmd_sub), ("dur", cmd_stop), ("simule", cmd_simulate), ("surpriz", cmd_upsets),
                     ("golcu", cmd_scorers), ("kupon", cmd_coupon), ("tahmin", cmd_guess), ("puan", cmd_board)):
        app.add_handler(CommandHandler(name, fn))
    app.add_handler(CallbackQueryHandler(on_button, pattern=r"^m:"))
    jq = app.job_queue
    jq.run_daily(job_morning, time(9, 30, tzinfo=TR))
    jq.run_daily(job_weekly, time(10, 0, tzinfo=TR), days=(1,))
    jq.run_daily(job_health, time(9, 20, tzinfo=TR))
    jq.run_repeating(job_lineups, interval=900, first=60)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
