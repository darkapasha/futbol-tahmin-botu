# ⚽ Futbol Tahmin Botu

Premier Lig, La Liga, Bundesliga, Serie A, Ligue 1, Süper Lig, Şampiyonlar Ligi ve Avrupa Ligi
maçları için olasılık tahmini yapan Telegram botu (**@Mactahminbot**).

## Nasıl tahmin ediyor?

1. **Veri**: 2021'den bugüne sonuçlar, oranlar, Elo, xG (Understat, FotMob), kadrolar, sakatlıklar,
   hakemler, Transfermarkt piyasa değerleri. Her sabah otomatik güncellenir.
2. **Takım gücü**: hücum/savunma puanları (gol + xG karışımı), kadro değeri, ilk 11'in kimlerden oluştuğu.
3. **Simülasyon**: maç dakika dakika 10.000 kez oynatılır (skor durumu, kırmızı kart, oyuncu bazında gol payı).
4. **Piyasa harmanı**: son olasılık %80 bahis oranı + %20 motor. Oranlardaki "favori-sürpriz" kayması düzeltilir.
5. **Kendi kendine öğrenme**: her sabah dünkü maçlara bakar; parametreler yalnızca kalıcı ve anlamlı bir
   sapma görülürse değişir (değişiklikler `sim/params_history/` içinde).

## Dürüst sonuç

Geriye dönük testte motor tek başına bahis piyasasını geçemiyor (2025/26 log-loss 0.987, piyasa 0.977).
Taktik eşleşmeleri, form, yorgunluk, motivasyon, derbi gibi onlarca fikir denendi; test sezonlarında
iyileştirmeyen hepsi reddedildi. Ayrıntılar ve tüm deneylerin sonuçları: [sim/README.md](sim/README.md)

Bu bir merak projesi; bahis tavsiyesi değildir.

## Bot

`/bugun` `/yarin` `/mac` `/lig` `/istatistik` `/simule` `/surpriz` `/golcu` `/kupon` `/tahmin` `/puan` `/rapor` · sabah kısa özet + maç başına düğmeler · yerel gemma3 ile kısa yorum ·
pazartesi haftalık rapor · veri kaynağı kesilirse abonelere tek seferlik uyarı.

Kurulum ve tablolar: [TEKNIK.md](TEKNIK.md) · Telegram token'ı yalnızca sunucudaki `.env` içinde.

---
Claude Code ile birlikte geliştirildi.
