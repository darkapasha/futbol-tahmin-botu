# Takım ve hoca taktik profilleri (2. adım)

```
.venv/bin/python -m profiles.build                  # hepsini hesapla (~1.5 dk)
.venv/bin/python -m profiles.build show Galatasaray # takımın güncel profili + hocası
```

Ana veritabanını (`data/futbol.sqlite`) yalnızca okur, sonuçları `data/profiles.sqlite`'a yazar
(kazıyıcılarla kilit çakışması olmasın diye). Birlikte sorgu: `ATTACH 'data/futbol.sqlite' AS src`.
Her çalıştırmada tablolar baştan yazılır; sabah işinden sonra çalıştırmak yeterli.

## Tablolar

| Tablo | Satır | İçerik |
|---|---|---|
| `team_match` | maç × takım | Tüm ham özellikler; simülatör kalibrasyonu için temel tablo |
| `subs` | oyuncu değişikliği | dakika, sebep, o anki skor durumu, çıkan/giren hattı, `delta` (+ hücumcu, − savunmacı) |
| `team_season` | takım × sezon | tüm kupalar dahil sezon profili + stil endeksleri |
| `league_season` | lig × sezon | lig ortalamaları (büzülme ve hoca karşılaştırması için) |
| `team_current` | takım | son 40 maç, yakın maç ağır (yarı ömür 15 maç), 8 maçlık lig ortalamasına büzülmüş. **Simülatörün okuyacağı tablo** |
| `coach_profile` | hoca (≥10 maç) | kariyer profili, `rel_*` = lig ortalamasına göre fark, `adapt_*` = güçlü rakibe karşı − zayıf rakibe karşı, değişiklik alışkanlığı |
| `coach_split` | hoca × bölüm | aynı özellikler `opp_class=stronger/similar/weaker`, `is_home=1/0` için |
| `coach_subs` | hoca × skor durumu | değişiklik dakikası, hücumcu/savunmacı değişiklik payı |
| `feature_dict` | özellik | tanım, kaynak, takım-sezon kapsama oranı |

## Kaynak eşleme

| Özellik grubu | Kaynak | Kapsam |
|---|---|---|
| Sonuç, iç saha/deplasman, diziliş, hoca | Transfermarkt `tm_games` | 7 lig, 2021–Mayıs 2026 |
| Oyun durumu (önde/eşit/geride dakika ve goller), ilk gol, 2. yarı | `tm_events` gol dakikaları | 7 lig |
| Gol türü: duran top, kafa, kontra, orta, penaltı | `tm_events` gol açıklaması | 7 lig |
| Kanat/merkez (ilk 11'de kanat + bek oranı) | `tm_lineups` pozisyonları | 7 lig |
| Değişiklik alışkanlığı | `tm_events` + `tm_players` pozisyonu | 7 lig |
| Şut, isabet, korner, faul, kart, bahis ihtimalleri | xgabora `xg_matches` | ülke ligleri (ŞL yok), Eylül 2026'ya kadar |
| xG, npxG, PPDA (pres), OPPDA, deep | Understat `us_team_games` | sadece ilk 5 lig |
| Topla oynama, pas, rakip yarı payı, uzun top, orta, ceza sahası teması, duran top / hızlı hücum xG'si | FotMob `fm_stats`, `fm_shots` | şimdilik birkaç maç; FotMob geçmişi doldukça kendiliğinden dolar |
| Rakip gücü (`elo_pre`, `elo_diff`, `opp_class`) | kendi Elo'muz (`profiles/elo.py`) | tüm maçlar; ±75 Elo eşiği |

## Stil endeksleri (lig-sezon içi z-skor; 0 = lig ortalaması)

`press_idx` (−PPDA), `control_idx` (topla oynama / saha hakimiyeti / deep), `directness_idx`
(uzun top, kontra gol, hızlı hücum xG), `width_style_idx` (kanat oyuncusu, orta golü, korner),
`setpiece_threat_idx`, `setpiece_weak_idx`, `attack_idx`, `defence_idx`, `shot_quality_idx`,
`lead_protect_idx` (ilk golü atınca puan, öndeyken yenilen), `comeback_idx`.
Bir endeksin bileşenlerinden hangisi doluysa o kullanılır; kaynak lige göre değiştiği için
endeksler ligler arası değil, lig içinde karşılaştırılmalı.

## Bilinen boşluklar

- **Süper Lig ve Şampiyonlar Ligi'nde xG/pres/topla oynama yok**: Understat bu ligleri kapsamıyor,
  FotMob geçmişi henüz dolmadı. İlk 5 ligde de topla oynama/pas/uzun top FotMob'dan gelecek.
- `team_map`'te FotMob takım kimlikleri az; FotMob maçları ancak eşleşen takımlar için bağlanıyor.
- Mayıs 2026 sonrası maçlarda (xgabora) olay, diziliş, hoca yok; güncel hoca FotMob'dan gelene
  kadar son Transfermarkt hocası yazılır.
- `adapt_ppda` / `adapt_xg_pg` kısmen rakibin kendisini yansıtır (güçlü rakip topu daha çok tutar);
  simülatörde hoca etkisi olarak kullanmadan önce takımın kendi ortalamasına göre düzeltilmeli.
- Maç içi diziliş değişikliği verisi yok (Transfermarkt sadece başlangıç dizilişi veriyor).
