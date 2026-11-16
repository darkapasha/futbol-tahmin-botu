# Futbol Tahmin Botu — veri altyapısı (1. adım)

Ligler: Premier Lig, La Liga, Bundesliga, Serie A, Ligue 1, Süper Lig, Şampiyonlar Ligi. Sezonlar 2021/22'den bugüne.
Tek veritabanı: `data/futbol.sqlite`. Telegram token'ı `.env` içinde (git'e girmez).

## Komutlar (proje klasöründen)

```
.venv\Scripts\python -m src.load_static              # hazır veri setlerini yükle (xgabora, Transfermarkt, sakatlık)
.venv\Scripts\python -m src.scrape_understat         # ilk 5 lig xG (önbellekli, ~30 istek)
.venv\Scripts\python -m src.scrape_fotmob history    # Süper Lig + ŞL geçmiş xG/kadro/hakem (yavaş, kaldığı yerden devam eder)
.venv\Scripts\python -m src.scrape_fotmob day 20261018
.venv\Scripts\python -m src.team_map                 # takım adlarını kaynaklar arasında eşleştir
.venv\Scripts\python -m src.daily                    # sabah işi: dün + bugün + güncel xG
.venv\Scripts\python -m src.statsbomb list|fetch     # olay verisi, gerekince
```

## Tablolar

| Tablo | Kaynak | İçerik |
|---|---|---|
| `xg_matches` | xgabora (football-data.co.uk) | sonuç, İY, şut, isabetli şut, korner, kart, faul, açılış/maks oranlar, Elo, form. Eylül 2026'ya kadar |
| `fd_matches`, `fd_fixtures` | football-data.co.uk (sunucudan) | güncel sezon sonuç, xG, açılış/kapanış oranları, hakem (yalnız ENG); yaklaşan maçların oranları. Her sabah yenilenir |
| `elo` | xgabora (clubelo) | takım Elo geçmişi |
| `tm_games`, `tm_club_games` | Transfermarkt | maç, hoca adı, hakem, diziliş, seyirci, sıralama. 6 Temmuz 2026'ya kadar, ŞL dahil |
| `tm_lineups`, `tm_appearances`, `tm_events` | Transfermarkt | ilk 11/yedek, pozisyon, dakika, gol/asist/kart, oyuncu değişikliği |
| `tm_players`, `tm_valuations`, `tm_transfers`, `tm_clubs` | Transfermarkt | oyuncu profili, piyasa değeri, transferler |
| `injuries` | salimt (Transfermarkt) | sakatlık: başlangıç/bitiş, kaçırılan maç. Eylül 2026'ya kadar |
| `us_matches`, `us_team_games` | Understat | ilk 5 lig maç xG + takım bazında npxG, PPDA, deep, xPts |
| `fm_fixtures`, `fm_matches`, `fm_stats`, `fm_lineups`, `fm_unavailable`, `fm_shots` | FotMob | Süper Lig + ŞL (ve sabah işinde tüm hedef ligler): xG, tüm maç istatistikleri, diziliş, hoca, hakem, oyuncu puanı, eksikler, şut haritası |
| `team_map` | — | her kaynağın takım adı/kimliği → ortak `team_key` (`tm:<club_id>` ya da yeni çıkanlar için `x:<ad>`) |

## Notlar

- football-data.co.uk ve Transfermarkt'ın asıl indirme adresi (r2.dev) Türkiye'den erişim engelli. İlki için xgabora'nın aynası, ikincisi için Hugging Face aynası (`ngeorgea/transfermarkt-player-scores`) kullanıldı. Güncel sonuçlar FotMob'dan geliyor.
- SofaScore yalnızca tarayıcı taklidiyle (`curl_cffi`) cevap veriyor; şimdilik yedek kaynak, `src/net.py` içinde `impersonate=True` ile kullanılabilir.
- Kazıyıcılar istekler arasında 3–7 sn bekler, 403/429'da uzun bekler. Bitmiş sezonlar `data/cache` altında saklanır, tekrar çekilmez.
- Wyscout (figshare) henüz indirilmedi; taktik kalibrasyon adımında gerekirse eklenecek.

## Tahmin motoru ve bot (3.–7. adım)

- `sim/`: maç simülasyon motoru, kalibrasyon, backtest, günlük öğrenme döngüsü. Ayrıntı: `sim/README.md`.
- `bot/`: sabah işi (`bot.pipeline`), Ollama yorumları (`bot.commentary`), Telegram botu @Mactahminbot (`bot.telegram_bot`).
- systemd: `futbol-predict.timer` (07:45 TR, öğren + tahmin + yorum), `futbol-bot.service` (bot, 09:30 abonelere özet).
- Test: `.venv/bin/python -m tests.test_bot`, `.venv/bin/python -m tests.test_learn`.
