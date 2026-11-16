# Maç simülasyon motoru (3. adım) ve backtest (4. adım)

```
.venv/bin/python -m sim.run 20261024              # o günün maçları: FotMob'dan muhtemel 11 + 10 bin simülasyon
.venv/bin/python -m sim.run show <match_id>        # kaydedilmiş tahminin tamamı (JSON)
.venv/bin/python -m sim.calibrate                  # tüm parametreleri geçmişten yeniden kestir (~20 sn)
.venv/bin/python -m sim.rolling tune               # güç modeli parametrelerini log-loss ile ayarla -> sim/ratings_tuned.json
.venv/bin/python -m sim.backtest 2025-08-01 2026-06-01 [--set use_coach=false] [--table ad]
.venv/bin/python -m sim.learn [report|replay|rollback]   # günlük öğrenme döngüsü
```

Ana veritabanını ve profilleri yalnızca okur. Kendi çıktıları: `data/sim.sqlite` (tahminler, kalibrasyon
tabloları, backtest, öğrenme kaydı) ve `sim/params.json` (global parametreler; eski halleri
`sim/params_history/` altında, hiçbiri silinmez).

## Nasıl çalışıyor

1. **Takım gücü** (`ratings.py`): her takımın hücum ve savunma puanı, gol ile xG karışımına (xG ağırlığı
   0.75) Poisson modeli. Yakın maçlar ağır (yarı ömür 150 gün). Her takım kendi liginin ortalamasına,
   kadro piyasa değerine göre kaydırılmış bir ön bilgiye büzülür (`value.py`); ligi yeni gelen takımlar
   ortalamanın altından başlar. Ligler arası seviye Şampiyonlar Ligi maçlarıyla bağlanır.
2. **Maç içi dinamikler** (`dynamics.py`): 11 bin Transfermarkt maçının gol ve kırmızı kart dakikalarından,
   maç öncesi beklentiye göre: dakika profili (uzatmalar dahil), skor durumu (geride olan takım 2. yarıda
   %10 daha çok gol atar, önde olan %9 daha az), kırmızı kart (10 kişi kalan takımın golü %50 düşer,
   rakibinki %64 artar) ve kırmızı kart riski (dakika ve skora göre).
3. **Kadro** (`players.py`, `fixtures.py`): FotMob'un maç öncesi sayfasından muhtemel / açıklanan 11 ve
   eksikler; yoksa son ilk 11'den sakatlar çıkarılıp aynı hattan en değerli yedekle doldurulur. İlk 11'in
   değeri takımın alışık olduğu 11'e göre düşükse güç o oranda (üs 0.1) düşer. Golcü payları: oyuncunun son 2
   sezondaki penaltısız gol oranı, mevkisine göre büzülmüş; penaltıyı takımın penaltıcısı atar.
4. **Monte Carlo** (`engine.py`): maç 90 dakika boyunca dakika dakika 10.000 kez oynanır. Her dakika her
   takım için şut sayısı ~ Poisson(beklenen gol / şut başına xG); her şutun xG'si takımın şut kalitesine göre
   Beta dağılımından çekilir, gol o olasılıkla olur. Penaltılar, kırmızı kartlar, skor durumu, oyundan
   çıkan oyuncular (golcü seçiminde) her dakika güncellenir. 10 bin simülasyon ~0.3 sn.
5. **Çıktı**: skor dağılımı, 1X2, ilk yarı, İY/MS, alt/üst 0.5–5.5, KG, ilk golü kim atar, kırmızı kart,
   beklenen şut ve xG, gol atma olasılığı en yüksek oyuncular.

## Teknik direktör etkileri (denendi, şimdilik kapalı)

Hocanın rakibe göre plan değiştirmesi (`coach_plan`) ve skor durumuna tepkisi (`coach_state`) geçmişten
kestirilip motora bağlandı; ancak 2025/26 sezonu backtest'inde ikisi de isabeti **düşürdü** (aşağıdaki
tablo). Gerçek hoca etkisi zaten takım puanının içinde; kalan sapmalar gürültü. Bu yüzden
`params.json`'da `use_coach_plan` / `use_coach_state` kapalı. Hoca adı, dizilişi ve alışkanlıkları
yorum metninde kullanılıyor. Veri arttıkça (`--set use_coach=true` ile) yeniden denenebilir.

## Backtest (2025/26 sezonu, yalnızca o güne kadarki veriyle)

| 2025/26 (2.246 maç) | 1X2 log-loss | 2.5 üst log-loss | KG log-loss |
|---|---|---|---|
| Yalnız güç modeli (Poisson) | 0.9879 | – | – |
| **Motor, son hali** (kadro değeri + kadro, hoca etkisi kapalı) | **0.9871** | 0.6798 | 0.6899 |
| Motor + hocanın skor tepkisi | 0.9882 | 0.6791 | 0.6879 |
| Motor + hoca planı + skor tepkisi (ilk sürüm) | 0.9939 | 0.6803 | 0.6897 |
| Bahis piyasası (oranı olan 2.058 maçta; motor aynı maçlarda 0.9910) | 0.9771 | 0.6754 | – |

| 2024/25 (2.281 maç), son hal | 1X2 log-loss | 2.5 üst log-loss |
|---|---|---|
| Motor | 0.9649 | 0.6751 |
| Bahis piyasası (aynı maçlarda motor 0.9704) | 0.9583 | 0.6679 |

Kalibrasyon iyi: %60–70 dediği maçların %65–70'i tutuyor, beraberlik %25 tahmin / %25 gerçek,
maç başı gol 2.87 / 2.81, golcü olasılıkları %20–30 diliminde %23 gerçekleşiyor.

Log-loss düşük olan iyi (1X2'de 1.0986 = yazı tura). Bahis piyasası en güçlü tahminci; motor ona yakın
ama altında. Fark büyük ölçüde piyasanın maç günü haberlerini (kadro, sakatlık, motivasyon) bilmesinden.

## Günlük öğrenme döngüsü (`learn.py`)

Her sabah dünün tahminleri sonuçlarla karşılaştırılır (`pred_eval`). Model **yalnızca** kalıcı ve
anlamlı bir hata görürse değişir: ligde gol seviyesi ya da ev sahibi avantajı son 120 günde istatistiksel
olarak (z > 3) sapıyorsa, beraberlikler sürekli eksik/fazla tahmin ediliyorsa, bir takım son 8 maçta
beklentinin çok dışındaysa. Her ayar küçük adımlı ve sınırlı, `learn_log` tablosuna kanıtıyla yazılır,
`python -m sim.learn rollback` ile geri alınır. Takım puanları zaten her gün yeni sonuçlarla yeniden
kurulduğu için sıradan bir sürpriz sonuç modeli oynatmaz. Hoca değişimleri örüntü olarak kaydedilir.

## Taktik eşleşme testi (22.10.2026, `python -m sim.tactics`)

Diziliş × diziliş (en sık 8 diziliş), 3/4/5'li savunma eşleşmesi, oyun tarzı çarpışmaları (pres × rakibin
topa sahip olması, direkt oyun × rakibin presi, kanat oyunu × 3'lü savunma, duran top tehdidi × zaafı,
kontra × topa sahip olma; bir önceki sezonun stil endeksleriyle) ve eksikler yüzünden alışık dizilişten
sapma, puan modelinin üzerine L2 cezalı Poisson modeliyle eklendi. 2021–24'te eğitildi, sonraki
sezonlarda denendi:

| Sezon | Mevcut model | Tam taktik modeli | Yalnız stil | Yalnız savunma hattı |
|---|---|---|---|---|
| 2024/25 log-loss / isabet | **0.9657** / %54.2 | 0.9666 / %54.0 | 0.9659 / %54.2 | 0.9657 / %54.4 |
| 2025/26 log-loss / isabet | 0.9878 / %52.6 | 0.9880 / %53.0 | 0.9881 / %52.8 | **0.9876** / %52.6 |

Fark gürültü düzeyinde (en fazla 0.0009), iki sezonda birden iyileştiren yok; mevcut model kaldı.
Bulunan etkiler küçük: direkt oyun rakip pres yaparken %2 fazla gol, topa sahip olan rakibe karşı %2.6
az; diziliş eşleşmeleri ±%10 ama örneklem küçük ve sezondan sezona tutmuyor. Puan modeli bu farkların
çoğunu zaten takımın sonuçlarından öğreniyor.

## Oyuncu eşleşme testleri (22.10.2026, `python -m sim.matchups`, `python -m sim.matchups2`)

Kanat × karşı bek, forvet × stoper, orta saha × orta saha (piyasa değeri ve FotMob form puanı; form = aynı
sezon aynı turnuvadaki önceki maç puanları), kanatçı tipi (golcü / oyun kurucu) × bek ve sakatlıktan /
uzun aradan dönen oyuncu. Eğitim < 2024-08, sınama 2024/25 ve 2025/26.

Takım tahmini (1X2 log-loss; FotMob puanı olan maçlar, 2024/25: 1352, 2025/26: 1254 maç):

| Varyant | 2024/25 | 2025/26 |
|---|---|---|
| Puan modeli | 0.9531 | 0.9895 |
| + ilk 11 değer farkı (motorda zaten var) | 0.9508 | **0.9877** |
| + ilk 11 form farkı | 0.9507 | 0.9877 |
| + kanat/bek form eşleşmesi | **0.9497** | 0.9883 |
| + sakatlıktan dönenler | 0.9498 | 0.9883 |

Golcü tahmini (kanatçı gol attı mı, log-loss):

| Varyant | 2024/25 | 2025/26 |
|---|---|---|
| Oyuncu değeri + golcü tipi | 0.41336 | 0.39539 |
| **+ karşı bekin değeri** | **0.41152** | 0.39484 |
| + kendi formu + karşı bekin formu | 0.41226 | 0.39447 |
| + tip × bek etkileşimi | 0.41246 | 0.39448 |
| + sakatlıktan dönüş | 0.41234 | 0.39441 |

Karar: iki sezonda da tutarlı iyileştiren tek şey, golcü olasılığında karşı bekin piyasa değeri; motora
eklendi (`players.flank_adjust`, `use_flank_matchups`). Form puanı, tip etkileşimi ve dönüş etkisi
sezondan sezona tutarsız ya da sıfır; eklenmedi. FotMob geçmişi doldukça tekrar denenebilir.

## Altı ek parametre testi (22.10.2026, `python -m sim.extras`)

Her biri ayrı eklendi; eğitim < 2024-08, sınama 2024/25 ve 2025/26 (1X2 log-loss, düşük iyi).

| Parametre | 2024/25 (mevcut → yeni) | 2025/26 (mevcut → yeni) | Karar |
|---|---|---|---|
| 1. Bahis oranları, %80 piyasa + %20 motor | 0.9704 → **0.9592** | 0.9910 → **0.9783** | **eklendi** (oran varsa) |
| 2. Güç modelinde xG ağırlığı 0.5 / **0.75** / 1.0 | 0.9646 / **0.9634** / 0.9639 | 0.9878 / **0.9873** / 0.9885 | 0.75 zaten kullanılıyor |
| 3. Yorgunluk (dinlenme, hafta içi Avrupa, Avrupa deplasmanı) | 0.9657 → 0.9638 | 0.9878 → 0.9880 | tutarsız, eklenmedi |
| 4. Motivasyon (şampiyonluk / küme yarışı, sezon sonu) + derbi | 0.9657 → 0.9655 | 0.9878 → 0.9880 | etkisiz, eklenmedi |
| 5. Kesin ilk 11 ağırlığı (maç günü 11'i, ayrı katsayı) | 0.9657 → 0.9639 | 0.9878 → 0.9888 | tutarsız; mevcut β=0.1 kaldı |
| 6a. Hakem → kırmızı kart (kırmızı kart log-loss) | 0.4555 → **0.4551** | 0.4558 → **0.4522** | **eklendi** |
| 6b. Hakem → 2.5 üst | 0.6751 → 0.6752 | 0.6799 → 0.6791 | tutarsız, eklenmedi |
| 6c. Hava durumu | – | – | geçmiş veri yok, test edilemedi |

Piyasa tek başına (2024/25 0.9583, 2025/26 0.9771) harmandan da biraz iyi; motor ağırlığı %20'de
tutuldu ki simülasyonun skor / golcü / kart çıktıları anlamlı kalsın ve piyasanın yanıldığı yerde motor
düzeltebilsin. Oranlar football-data.co.uk'den (fd_fixtures); Şampiyonlar / Avrupa Ligi'nde oran yok,
orada yalnız motor kullanılır. Bulunan ama eklenmeyen etkiler: hafta içi Avrupa maçı oynayan takım
%9 az gol atıyor, rakibi %10 çok; derbilerde gol %8 az — ama sezondan sezona isabeti artırmadı.

## İsabet testleri 2 (22.10.2026, `python -m sim.odds_tests`)

football-data.co.uk sezon dosyaları (Avg = açılış, AvgC = kapanış), 6 ülke ligi. 1X2 log-loss:

| | 2024/25 | 2025/26 |
|---|---|---|
| Motor | 0.9699 | 0.9910 |
| Açılış oranı | 0.9576 | 0.9771 |
| Kapanış oranı | **0.9543** | **0.9760** |
| Kapanış %100 / %90 / %80 + motor | 0.9543 / 0.9548 / 0.9555 | 0.9760 / 0.9766 / 0.9773 |
| Oran hareketini ileri taşıma (k=0.25 / 0.5) | 0.9553 / 0.9552 | 0.9774 / 0.9777 |
| Harman + sıcaklık 0.9 (çapraz sezon ayarı) | 0.9585 → **0.9570** | 0.9783 → **0.9774** |
| Harman + beraberlik çarpanı | 0.9586 | 0.9783 |

2.5 üst: kapanış 0.6639 / 0.6713, açılış 0.6678 / 0.6750, motor 0.6778 / 0.6847.

Uygulananlar: piyasa payı %80 → %90; harman olasılığa sıcaklık 0.9 (favoriler sistematik az tahmin
ediliyordu; "Sürpriz örüntüleri" başlığındaki favori-sürpriz yanlılığı bulgusuyla aynı yönde);
maçtan 40–90 dk önce fixtures.csv'den taze oranla ve açıklanan 11 ile tahmin bir kez yenileniyor.
Uygulanmayanlar: oran hareketini ileri taşıma, beraberlik çarpanı.
Değer bahsi (motor olasılığı × oran > 1 + eşik): 2024/25 ve 2025/26'da tüm eşiklerde zarar (−%10 ile
−%29), bu yüzden "değerli bahis" işareti eklenmedi. Kupa maçları için ücretsiz oran kaynağı bulunamadı
(FotMob vermiyor, football-data kupaları kapsamıyor).
