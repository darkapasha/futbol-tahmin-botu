"""Sabah guncellemesi: guncel sezonun eksik sonuclari + fikstur + guncel sezon xG + profiller.

Kullanim:  .venv\\Scripts\\python -m src.daily
Ileride Telegram botu bunu her sabah calistirip tahminleri bunun ustune kuracak.
"""
import subprocess
import sys

from config import LEAGUES, SEASONS
from src import scrape_footballdata, scrape_fotmob, scrape_understat, team_map


def main():
    # Guncel sezonun fiksturu 7 lig icin bastan okunur; eksik kalan her bitmis
    # mac (kacirilan gunler dahil) tamamlanir, bugunun maclari fikstura girer.
    scrape_fotmob.history(list(LEAGUES), SEASONS[-1:])
    scrape_understat.main(fresh=True)
    # Guncel sezon kapanis oranlari ve hakemler, yaklasan maclarin oranlari
    scrape_footballdata.main()
    team_map.main()
    # Takim ve hoca profilleri guncel veriyle yeniden uretilir
    subprocess.run([sys.executable, "-m", "profiles.build"], check=True)


if __name__ == "__main__":
    main()
