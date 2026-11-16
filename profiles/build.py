"""Tum profilleri hesaplayip data/profiles.sqlite'a yazar.

Kullanim:  .venv\\Scripts\\python -m profiles.build
           .venv\\Scripts\\python -m profiles.build show "Galatasaray"   # takim + hocasi
"""
import sys
import time

import pandas as pd

from . import coach, team
from .common import PROFILE_DB, out_con, src_con
from .elo import add_elo
from .features import FEATURES, prepare
from .match_features import build as build_matches


def run():
    t0 = time.time()
    src = src_con()
    long, subs = build_matches(src)
    add_elo(long)
    long = prepare(long)
    print(f"mac x takim: {len(long)} satir ({time.time() - t0:.0f} sn)")

    seasons = team.season_profiles(long)
    leagues = team.league_profiles(long)
    seasons = team.add_style(seasons, seasons)
    ref = team.ref_seasons(seasons)
    current = team.add_style(team.current_profiles(long, leagues, ref), seasons, "ref_season")
    cprof, csplit = coach.coach_profiles(long, leagues)
    csubs, csubs_wide = coach.coach_subs(subs)
    if not cprof.empty:
        cprof = cprof.merge(csubs_wide, on="coach", how="left", suffixes=("", "_subs"))

    names = pd.read_sql("SELECT club_id, name FROM tm_clubs", src)
    names = dict(zip("tm:" + names.club_id.astype(str), names.name))
    for df in (seasons, current):
        df.insert(0, "team_name", df.team_key.map(names).fillna(df.team_key.str[2:]))

    fdict = pd.DataFrame([(k, *v) for k, v in FEATURES.items()],
                         columns=["feature", "numerator", "denominator", "description", "source"])
    fdict["coverage_team_season"] = [seasons[k].notna().mean() for k in FEATURES]

    out = out_con()
    tables = {"team_match": long, "subs": subs, "team_season": seasons, "league_season": leagues,
              "team_current": current, "coach_profile": cprof, "coach_split": csplit,
              "coach_subs": csubs, "feature_dict": fdict}
    for name, df in tables.items():
        df.to_sql(name, out, if_exists="replace", index=False)
        print(f"  {name}: {len(df)}")
    out.close()
    print(f"bitti -> {PROFILE_DB} ({time.time() - t0:.0f} sn)")


def show(q):
    con = out_con()
    pd.set_option("display.width", 200, "display.max_rows", 200)
    cur = pd.read_sql("SELECT * FROM team_current WHERE team_name LIKE ?", con, params=(f"%{q}%",))
    for _, r in cur.iterrows():
        print(r.dropna().to_string())
        c = pd.read_sql("SELECT * FROM coach_profile WHERE coach = ?", con, params=(r.coach_last,))
        if len(c):
            print("\n--- hoca ---")
            print(c.iloc[0].dropna().to_string())


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "show":
        show(sys.argv[2])
    else:
        run()
