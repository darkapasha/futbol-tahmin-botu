"""Yerel LLM (Ollama, gemma3:4b) ile kisa taktik mac onu yorumu.

Model yalnizca verilen sayilari ve olgulari yorumlar; prompt'ta uydurma yasak. Sonuc
data/sim.sqlite 'commentary' tablosunda (mac + tahmin zamani) saklanir, tekrar uretilmez.
Ollama yoksa ya da cevap vermezse kurallarla yazilmis kisa bir yorum doner.
"""
import json
import sqlite3

import pandas as pd
import requests

from sim.common import PROFILE_DB, ro, sim_con

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "gemma3:4b"
FALLBACK_MODEL = "qwen2.5:latest"

SCHEMA = """CREATE TABLE IF NOT EXISTS commentary (
  match_id INTEGER, created_at TEXT, model TEXT, text TEXT, PRIMARY KEY (match_id, created_at));"""

STYLE_TR = {
    "press_idx": ("yüksek pres yapar", "geride bekler, az pres yapar"),
    "control_idx": ("topa sahip olup oyunu kurar", "topu rakibe bırakır"),
    "directness_idx": ("direkt oynar, uzun top ve hızlı hücum sever", "sabırlı pasla oynar"),
    "width_style_idx": ("kanatlardan ve ortayla oynar", "merkezden oynar"),
    "setpiece_threat_idx": ("duran toplarda tehlikelidir", None),
    "setpiece_weak_idx": ("duran toplarda gol yemeye yatkındır", None),
    "lead_protect_idx": ("öne geçince skoru iyi korur", "öne geçince gol yemeye yatkındır"),
    "comeback_idx": ("geriye düşünce maçı çevirebilir", "geriye düşünce toparlanmakta zorlanır"),
}


def team_style(team_key):
    con = ro(PROFILE_DB)
    t = pd.read_sql("SELECT * FROM team_current WHERE team_key = ?", con, params=(team_key,))
    if t.empty:
        return [], None
    r = t.iloc[0]
    notes = []
    for col, (hi, lo) in STYLE_TR.items():
        v = r.get(col)
        if pd.isna(v):
            continue
        if v > 0.6:
            notes.append(hi)
        elif v < -0.6 and lo:
            notes.append(lo)
    return notes, r.get("main_formation")


def level(x, cuts, words):
    return words[0] if x < cuts[0] else (words[1] if x < cuts[1] else words[2])


def favorite(o):
    h, a = o["home_name"], o["away_name"]
    d = o["p_home"] - o["p_away"]
    if abs(d) < 0.1:
        return "güçler dengeli, beraberlik de güçlü bir ihtimal"
    fav = h if d > 0 else a
    return f"{fav} {'açık' if abs(d) > 0.35 else 'hafif'} favori"


def clean(txt):
    """Modelin kural disi eklediklerini (baslik, emoji satiri, madde) ayikla."""
    import re
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    keep = [l for l in lines if not (len(l) < 70 and ("Yorum" in l or l.startswith(("#", "*", "⚽"))))]
    t = " ".join(keep).replace("**", "")
    # model bazen yasaklanan yuzdeleri uyduruyor: yuzde iceren cumleyi at
    sents = re.split(r"(?<=[.!?])\s+", t)
    t = " ".join(x for x in sents if "%" not in x) or t
    return re.sub(r"[🌀-🫿☀-➿]️?", "", t).strip()


def facts(o):
    h, a = o["home_name"], o["away_name"]
    hs, hf = team_style(o["home_key"])
    as_, af = team_style(o["away_key"])
    plan = lambda s: {"stronger": "güçlü rakibe karşı", "weaker": "kendinden zayıf rakibe karşı",
                      "similar": "denk rakibe karşı"}.get(s.get("opp_class"), "")
    f = {
        "mac": f"{h} - {o['away_name']} ({o['league']})",
        "kim_favori": favorite(o),
        "gol_beklentisi": level(sum(o["exp_goals"]), (2.3, 3.0), ("az gollü", "orta", "bol gollü")),
        "karsilikli_gol": level(o["btts"], (0.45, 0.58), ("ihtimal düşük", "yarı yarıya", "ihtimal yüksek")),
        "en_olasi_skorlar": ", ".join(s for s, p in o["scores"][:3]),
        "ev_hocasi": o["home"].get("coach"), "dep_hocasi": o["away"].get("coach"),
        "ev_dizilis": hf, "dep_dizilis": af,
        "ev_oyun_tarzi": hs, "dep_oyun_tarzi": as_,
        "ev_durum": plan(o["home"]), "dep_durum": plan(o["away"]),
        "ev_eksikler": o["home"].get("missing", [])[:5], "dep_eksikler": o["away"].get("missing", [])[:5],
        "gol_adaylari": [f"{s['name']} %{100 * s['p']:.0f}" for side in o["scorers"] for s in side[:2]],
    }
    return f


PROMPT = """Sen deneyimli bir Türk futbol yorumcususun. Aşağıdaki VERİLERE dayanarak bu maç için 4-5 cümlelik,
akıcı Türkçe bir maç önü taktik yorumu yaz. Kurallar:
- Yalnızca verilen bilgileri kullan; oyuncu, sakatlık, istatistik UYDURMA.
- İki takımın oyun tarzının nasıl çarpışacağını, hocaların olası planını ve kritik oyuncuları anlat.
- Sonda tek cümleyle en olası senaryoyu söyle. Sayı ve yüzde yazma.
- Başlık, madde işareti, emoji kullanma.

VERİLER:
{facts}
"""


def generate(o, model=MODEL, timeout=240):
    f = facts(o)
    body = {"model": model, "prompt": PROMPT.format(facts=json.dumps(f, ensure_ascii=False, indent=1)),
            "stream": False, "options": {"temperature": 0.6, "num_predict": 320, "num_ctx": 2048}}
    try:
        r = requests.post(OLLAMA, json=body, timeout=timeout)
        r.raise_for_status()
        txt = clean(r.json().get("response", ""))
        if len(txt) > 80:
            return txt, model
    except Exception as e:  # noqa
        print("  ollama hatasi:", e)
    return rule_based(o, f), "kural"


def rule_based(o, f=None):
    f = f or facts(o)
    h, a = o["home_name"], o["away_name"]
    fav = h if o["p_home"] > o["p_away"] + 0.1 else (a if o["p_away"] > o["p_home"] + 0.1 else None)
    s = [f"{h} ile {a} arasındaki maçta " + (f"favori {fav}." if fav else "güçler dengeli.")]
    if f["ev_oyun_tarzi"]:
        s.append(f"{h} genelde {', '.join(f['ev_oyun_tarzi'][:2])}.")
    if f["dep_oyun_tarzi"]:
        s.append(f"{a} ise {', '.join(f['dep_oyun_tarzi'][:2])}.")
    tot = sum(o["exp_goals"])
    s.append("Bol gollü bir maç bekleniyor." if tot > 3 else ("Az gollü, kontrollü bir maç bekleniyor." if tot < 2.3 else "Orta tempolu bir maç bekleniyor."))
    return " ".join(s)


def get(o, created_at, force=False):
    con = sim_con()
    con.executescript(SCHEMA)
    if not force:
        r = con.execute("SELECT text FROM commentary WHERE match_id = ? ORDER BY created_at DESC LIMIT 1",
                        (o["match_id"],)).fetchone()
        if r:
            return clean(r[0])
    txt, model = generate(o)
    con.execute("INSERT OR REPLACE INTO commentary VALUES (?,?,?,?)", (o["match_id"], created_at, model, txt))
    con.commit()
    return txt
