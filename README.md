# ⚽ Futbol Tahmin Botu

> Her maçı dakika dakika **10.000 kez** oynatıp olasılık çıkaran, her sabah kendi hatalarına bakan bir Telegram botu.

Telegram'da **@Mactahminbot**. Premier Lig, La Liga, Bundesliga, Serie A, Ligue 1, **Süper Lig**,
Şampiyonlar Ligi ve Avrupa Ligi.

⚠️ *Bu bir merak projesidir, bahis tavsiyesi değildir.*

---

## Bot ne veriyor?

Her sabah kısa bir özet: bugünün maçları ve her biri için düğme. Düğmeye basınca:

```
⚽ Beşiktaş – Galatasaray
🇹🇷 Süper Lig · 20:00

1 %38   X %27   2 %35   ⚖️
Beklenen gol: 1.41 – 1.33  (şut 13 – 12)
2.5 Üst %51 · 3.5 Üst %29 · KG Var %56
Olası skorlar: 1-1 (%12), 1-0 (%10), 2-1 (%9), 0-1 (%8)
İlk yarı: 1 %33 · X %42 · 2 %25
İlk golü atan: Beşiktaş %47 · Galatasaray %45 · golsüz %8
... golcü adayları ve yerel yapay zekânın kısa yorumu
```
*(rakamlar örnek)*

| Komut | Ne yapar |
|---|---|
| `/bugun` · `/yarin` | Günün / yarının maçları |
| `/mac <takım>` | Tek maçın ayrıntılı tahmini |
| `/lig <lig>` | Bir ligin haftası |
| `/simule <ev> <deplasman>` | İstediğin iki takımı karşılaştır |
| `/golcu` | Gol atma olasılığı en yüksek oyuncular |
| `/surpriz` | Sürpriz olasılığı yüksek maçlar |
| `/kupon` | Olasılıklardan örnek bir kupon (eğlence için) |
| `/tahmin` · `/puan` | Sen de tahmin et, botla yarış, puan tablosu |
| `/istatistik` · `/rapor` | Botun isabeti: günlük, haftalık |
| `/abone` · `/dur` | Sabah özetini aç / kapat |

Pazartesi haftalık rapor gelir. Bir veri kaynağı kesilirse abonelere tek seferlik uyarı gider.

## Nasıl tahmin ediyor?

```
  veri topla ──▶ takım gücü ──▶ simülasyon ──▶ piyasa harmanı ──▶ yorum ──▶ Telegram
  (her sabah)    (hücum/savunma, (10.000 maç,   (%80 oran +       (gemma3,
                  kadro değeri,   dakika dakika)  %20 motor)        yerel)
                  ilk 11)
                         ▲                                │
                         └────── her sabah dünkü maçlara bakıp öğrenme ◀┘
```

1. **Veri** (2021'den bugüne): sonuçlar, bahis oranları, Elo, xG (Understat, FotMob), muhtemel ve kesin
   ilk 11'ler, sakatlıklar, hakemler, Transfermarkt piyasa değerleri.
2. **Takım gücü**: Poisson hücum/savunma puanları (gerçek gol ile xG karışımı), kadro değerinden ön bilgi,
   sahadaki 11'in kimlerden oluştuğu. Avrupa kupalarında tanınmayan kulüpler için Elo ve kadro değeri.
3. **Simülasyon**: maç dakika dakika oynatılır. Skor durumu (öndeki takım yavaşlar), kırmızı kart, hakemin
   kart eğilimi, oyuncu bazında gol payı (kanat oyuncusu × karşısındaki bek eşleşmesi dahil). 10.000 maç ~0,3 sn.
4. **Piyasa harmanı**: son olasılık %80 bahis oranı + %20 motor. Oranlardaki "sürprizi abartma" kayması
   eğitim döneminden öğrenilen bir düzeltmeyle giderilir.
5. **Öğrenme**: her sabah dünkü maçlar değerlendirilir. Parametreler yalnızca **kalıcı ve istatistiksel olarak
   anlamlı** bir sapma görülürse değişir; her değişiklik `sim/params_history/` içinde saklanır.

## Dürüst sonuçlar

Kural: her fikir **2024'ten önceki** veriyle kurulur ve **hem 2024/25 hem 2025/26** sezonunu iyileştirmezse reddedilir.

| | 2024/25 log-loss | 2025/26 log-loss |
|---|---|---|
| Motor tek başına | 0,965 | 0,987 |
| Bahis piyasası | 0,958 | 0,977 |

**Motor tek başına piyasayı geçemiyor** (beklenen bir şey: piyasa milyonlarca dolarlık bilgiyi fiyatlıyor).
Bu yüzden son tahmin piyasaya yaslanıyor; motor skor, golcü ve "ya şöyle olsaydı" sorularında işe yarıyor.

**Denenip reddedilenler** (test sezonlarında iyileştirmedi): teknik direktör planı ve ruh hali, taktik/diziliş
eşleşmeleri, oyuncu form puanları, kanat oyuncusu tipi, sakatlıktan dönüş, yorgunluk, motivasyon ve derbi etkisi,
kesin 11'e ayrı ağırlık, hakemin alt/üst etkisi.

**Sürpriz analizi**: 23.000 maç çiftinde 66 özellikle (dinlenme, Avrupa maçı, derbi, form, yeni hoca, hakem,
xG şansı, rotasyon...) "favori neden kaybeder?" arandı. Düzeltme sonrası kalıcı bir desen bulunamadı;
sadece zayıf ipuçları (favorinin rotasyonlu 11'i, 4 gün sonra Şampiyonlar Ligi maçı olması).

Bütün deneylerin sayıları: [sim/README.md](sim/README.md)

## Yapı

```
src/        veri toplama: FotMob, Understat, football-data, Transfermarkt, sakatlıklar, takım adı eşleme
profiles/   takım, maç ve teknik direktör profilleri, kendi Elo'su
sim/        puanlar, dinamikler, oyuncular, simülasyon motoru, piyasa harmanı, kalibrasyon, öğrenme, testler
bot/        sabah işi, yorum (Ollama), Telegram botu, sağlık kontrolü, raporlar
deploy/     systemd servisleri ve zamanlayıcılar (07:00 veri, 07:45 tahmin)
tests/      bot ve öğrenme testleri
```

Kurulum ve tabloların ayrıntısı: [TEKNIK.md](TEKNIK.md) · Telegram token'ı yalnızca sunucudaki `.env` içinde.

---

Oracle Cloud'daki küçük bir ARM sunucuda 7/24 çalışıyor. Claude ile birlikte, vibe coding ile yapıldı 🎧
