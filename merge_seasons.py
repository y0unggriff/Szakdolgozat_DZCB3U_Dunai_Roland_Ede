import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path(r"C:\Egyetem\Szakdolgozat\Szakdoga adatok")
OUTPUT = INPUT_DIR / "merged_database.csv"

# bemeneti fájlok
FILE_201718 = INPUT_DIR / "transfermarkt_fbref_201718.csv"
FILE_201819 = INPUT_DIR / "transfermarkt_fbref_201819.csv"
FILE_201920 = INPUT_DIR / "transfermarkt_fbref_201920.csv"
FILE_202021 = INPUT_DIR / "cleaned_2020-21.csv"
FILE_202122 = INPUT_DIR / "2021-2022_Footbal_Player_Stats.csv"
FILE_202223 = INPUT_DIR / "cleaned_2022-23.csv"
FILE_202324 = INPUT_DIR / "cleaned_2023-24.csv"
FILE_202425 = INPUT_DIR / "players_data-2024_2025.csv"

# végső oszloprend
TARGET_COLS = [
    "player",
    "league",
    "season",
    "team",
    "nationality",
    "position",
    "age",
    "matches_played",
    "minutes",
    "goals",
    "assists",
    "xG",
    "xA",
    "xG/90",
    "xA/90",
    "shots_total",
    "shots_on_target",
    "tackles_total",
    "tackles_won",
    "shots_blocked",
    "passes_blocked",
    "interceptions",
    "clearances",
    "total_passes",
    "passes_completed",
    "shot_creating_action",
    "goal_creating_action",
    "total_take_ons",
    "take_ons_won",
    "progressive_passes",
    "key_passes",
    "errors",
    "goals_against",
    "saves",
    "saves_%",
    "clean sheets",
    "penatly_saves_%",
]

# liganévmappelés
LEAGUE_MAP = {
    "eng Premier League": "Premier League",
    "es La Liga": "La Liga",
    "it Serie A": "Serie A",
    "fr Ligue 1": "Ligue 1",
    "de Bundesliga": "Bundesliga",
}


def load_201718_19_20(path, season_str):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)

    out = pd.DataFrame()
    out["player"] = df["player"]
    out["league"] = df["league"]
    out["season"] = season_str
    out["team"] = df["squad"]
    out["nationality"] = df["nationality"]
    out["position"] = df["position"]
    out["age"] = pd.to_numeric(df["age"], errors="coerce")
    out["matches_played"] = pd.to_numeric(df["games"], errors="coerce")
    out["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    out["goals"] = pd.to_numeric(df["goals"], errors="coerce")
    out["assists"] = pd.to_numeric(df["assists"], errors="coerce")
    out["xG"] = pd.to_numeric(df["xg"], errors="coerce")
    out["xA"] = pd.to_numeric(df["xa"], errors="coerce")
    out["xG/90"] = pd.to_numeric(df["xg_per90"], errors="coerce")
    out["xA/90"] = pd.to_numeric(df["xa_per90"], errors="coerce")
    out["shots_total"] = pd.to_numeric(df["shots_total"], errors="coerce")
    out["shots_on_target"] = pd.to_numeric(df["shots_on_target"], errors="coerce")
    out["tackles_total"] = pd.to_numeric(df["tackles"], errors="coerce")
    out["tackles_won"] = pd.to_numeric(df["tackles_won"], errors="coerce")
    out["shots_blocked"] = pd.to_numeric(df["blocked_shots"], errors="coerce")
    out["passes_blocked"] = pd.to_numeric(df["blocked_passes"], errors="coerce")
    out["interceptions"] = pd.to_numeric(df["interceptions"], errors="coerce")
    out["clearances"] = pd.to_numeric(df["clearances"], errors="coerce")
    out["total_passes"] = pd.to_numeric(df["passes"], errors="coerce")
    out["passes_completed"] = pd.to_numeric(df["passes_completed"], errors="coerce")
    out["shot_creating_action"] = pd.to_numeric(df["sca"], errors="coerce")
    out["goal_creating_action"] = pd.to_numeric(df["gca"], errors="coerce")
    out["total_take_ons"] = pd.to_numeric(df["dribbles"], errors="coerce")
    out["take_ons_won"] = pd.to_numeric(df["dribbles_completed"], errors="coerce")
    out["progressive_passes"] = pd.to_numeric(df["progressive_passes"], errors="coerce")
    out["key_passes"] = pd.to_numeric(df["assisted_shots"], errors="coerce")
    out["errors"] = pd.to_numeric(df["errors"], errors="coerce")

    # kapusstatok
    out["goals_against"] = pd.to_numeric(df.get("goals_against_gk"), errors="coerce")
    out["saves"] = pd.to_numeric(df.get("saves"), errors="coerce")
    out["saves_%"] = pd.to_numeric(df.get("save_pct"), errors="coerce")
    out["clean sheets"] = pd.to_numeric(df.get("clean_sheets"), errors="coerce")

    pens_saved = pd.to_numeric(df.get("pens_saved"), errors="coerce")
    pens_att = pd.to_numeric(df.get("pens_att_gk"), errors="coerce")
    out["penatly_saves_%"] = (pens_saved / pens_att.replace(0, np.nan)) * 100

    return out


def load_cleaned(path, season_str):
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)

    out = pd.DataFrame()
    out["player"] = df["player"]
    out["league"] = df["comp"]
    out["season"] = season_str
    out["team"] = df["squad"]
    out["nationality"] = df["nation"]
    out["position"] = df["pos"]
    out["age"] = pd.to_numeric(df["age"], errors="coerce")
    out["matches_played"] = pd.to_numeric(df["Matches Played"], errors="coerce")

    out["minutes"] = pd.to_numeric(df["Avg Mins per Match"], errors="coerce")

    out["goals"] = pd.to_numeric(df["Goals"], errors="coerce")
    out["assists"] = pd.to_numeric(df["Assists"], errors="coerce")
    out["xG"] = pd.to_numeric(df["Expected Goals"], errors="coerce")
    out["xA"] = np.nan
    out["xG/90"] = np.nan
    out["xA/90"] = np.nan
    out["shots_total"] = pd.to_numeric(df["Total Shots"], errors="coerce")

    sot_pct = pd.to_numeric(df["% Shots on target"], errors="coerce")
    out["shots_on_target"] = (out["shots_total"] * sot_pct / 100).round()

    out["tackles_total"] = pd.to_numeric(df["Tackles attempted"], errors="coerce")
    out["tackles_won"] = pd.to_numeric(df["Tackles Won"], errors="coerce")
    out["shots_blocked"] = pd.to_numeric(df["Shots blocked"], errors="coerce")
    out["passes_blocked"] = pd.to_numeric(df["Passes blocked"], errors="coerce")
    out["interceptions"] = pd.to_numeric(df["Interceptions"], errors="coerce")
    out["clearances"] = pd.to_numeric(df["Clearances"], errors="coerce")
    out["total_passes"] = pd.to_numeric(df["Passes Attempted"], errors="coerce")
    out["passes_completed"] = pd.to_numeric(df["Passes Completed"], errors="coerce")

    n90 = out["minutes"] / 90
    sca90 = pd.to_numeric(df["Shot creating actions p 90"], errors="coerce")
    gca90 = pd.to_numeric(df["Goal creating actions p 90"], errors="coerce")
    out["shot_creating_action"] = (sca90 * n90).round()
    out["goal_creating_action"] = (gca90 * n90).round()

    out["total_take_ons"] = pd.to_numeric(df["Take ons attempted"], errors="coerce")
    succ_pct = pd.to_numeric(df["% Successful take-ons"], errors="coerce")
    out["take_ons_won"] = (out["total_take_ons"] * succ_pct / 100).round()

    out["progressive_passes"] = pd.to_numeric(df["Progressive Passes"], errors="coerce")
    out["key_passes"] = pd.to_numeric(df["Key passes"], errors="coerce")
    out["errors"] = pd.to_numeric(df["Errors made"], errors="coerce")

    out["goals_against"] = pd.to_numeric(df["Goals Against"], errors="coerce")
    out["saves"] = pd.to_numeric(df["Saves"], errors="coerce")
    out["saves_%"] = pd.to_numeric(df["Saves %"], errors="coerce")
    out["clean sheets"] = pd.to_numeric(df["Clean Sheets"], errors="coerce")
    out["penatly_saves_%"] = pd.to_numeric(df["% Penalty saves"], errors="coerce")

    return out


def load_fbref_season(path, season_str):

    df = pd.read_csv(path, sep=";", encoding="latin-1", low_memory=False)

    out = pd.DataFrame()
    out["player"] = df["Player"]
    out["league"] = df["Comp"]
    out["season"] = season_str
    out["team"] = df["Squad"]
    out["nationality"] = df["Nation"]
    out["position"] = df["Pos"]
    out["age"] = pd.to_numeric(df["Age"], errors="coerce")
    out["matches_played"] = pd.to_numeric(df["MP"], errors="coerce")
    out["minutes"] = pd.to_numeric(df["Min"], errors="coerce")

    n90 = pd.to_numeric(df["90s"], errors="coerce")


    def total(col):
        v = pd.to_numeric(df[col], errors="coerce")
        return (v * n90).round()

    out["goals"] = total("Goals")
    out["assists"] = total("Assists")
    out["xG"] = np.nan  
    out["xA"] = np.nan
    out["xG/90"] = np.nan
    out["xA/90"] = np.nan
    out["shots_total"] = total("Shots")
    out["shots_on_target"] = total("SoT")
    out["tackles_total"] = total("Tkl")
    out["tackles_won"] = total("TklWon")
    out["shots_blocked"] = total("BlkSh")
    out["passes_blocked"] = total("BlkPass")
    out["interceptions"] = total("Int")
    out["clearances"] = total("Clr")
    out["total_passes"] = total("PasTotAtt")
    out["passes_completed"] = total("PasTotCmp")
    out["shot_creating_action"] = total("SCA")
    out["goal_creating_action"] = total("GCA")

    if "DriAtt" in df.columns:
        out["total_take_ons"] = total("DriAtt")
        out["take_ons_won"] = total("DriSucc")
    else:
        out["total_take_ons"] = total("ToAtt")
        out["take_ons_won"] = total("ToSuc")

    out["progressive_passes"] = total("PasProg")
    out["key_passes"] = total("PasAss")
    out["errors"] = total("Err")

    out["goals_against"] = np.nan
    out["saves"] = np.nan
    out["saves_%"] = np.nan
    out["clean sheets"] = np.nan
    out["penatly_saves_%"] = np.nan

    return out


def load_202425(path):
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    out = pd.DataFrame()
    out["player"] = df["Player"]
    out["league"] = df["Comp"]
    out["season"] = "2024-2025"
    out["team"] = df["Squad"]
    out["nationality"] = df["Nation"]
    out["position"] = df["Pos"]
    out["age"] = pd.to_numeric(df["Age"], errors="coerce")
    out["matches_played"] = pd.to_numeric(df["MP"], errors="coerce")
    out["minutes"] = pd.to_numeric(df["Min"], errors="coerce")
    out["goals"] = pd.to_numeric(df["Gls"], errors="coerce")
    out["assists"] = pd.to_numeric(df["Ast"], errors="coerce")
    out["xG"] = pd.to_numeric(df["xG"], errors="coerce")
    out["xA"] = pd.to_numeric(df["xA"], errors="coerce")

    n90 = out["minutes"] / 90
    out["xG/90"] = (out["xG"] / n90.replace(0, np.nan)).round(2)
    out["xA/90"] = (out["xA"] / n90.replace(0, np.nan)).round(2)

    out["shots_total"] = pd.to_numeric(df["Sh"], errors="coerce")
    out["shots_on_target"] = pd.to_numeric(df["SoT"], errors="coerce")
    out["tackles_total"] = pd.to_numeric(df["Tkl"], errors="coerce")
    out["tackles_won"] = pd.to_numeric(df["TklW"], errors="coerce")

    out["shots_blocked"] = pd.to_numeric(df.get("Sh_stats_defense"), errors="coerce")
    out["passes_blocked"] = pd.to_numeric(df.get("Pass"), errors="coerce")
    out["interceptions"] = pd.to_numeric(df["Int"], errors="coerce")
    out["clearances"] = pd.to_numeric(df["Clr"], errors="coerce")
    out["total_passes"] = pd.to_numeric(df["Att"], errors="coerce")
    out["passes_completed"] = pd.to_numeric(df["Cmp"], errors="coerce")
    out["shot_creating_action"] = pd.to_numeric(df["SCA"], errors="coerce")
    out["goal_creating_action"] = pd.to_numeric(df["GCA"], errors="coerce")
    out["total_take_ons"] = pd.to_numeric(df["Att_stats_possession"], errors="coerce")
    out["take_ons_won"] = pd.to_numeric(df["Succ"], errors="coerce")
    out["progressive_passes"] = pd.to_numeric(df["PrgP"], errors="coerce")
    out["key_passes"] = pd.to_numeric(df["KP"], errors="coerce")
    out["errors"] = pd.to_numeric(df["Err"], errors="coerce")

    out["goals_against"] = pd.to_numeric(df.get("GA"), errors="coerce")
    out["saves"] = pd.to_numeric(df.get("Saves"), errors="coerce")
    out["saves_%"] = pd.to_numeric(df.get("Save%"), errors="coerce")
    out["clean sheets"] = pd.to_numeric(df.get("CS"), errors="coerce")

    pksv = pd.to_numeric(df.get("PKsv"), errors="coerce")
    pkatt = pd.to_numeric(df.get("PKatt_stats_keeper"), errors="coerce")
    out["penatly_saves_%"] = (pksv / pkatt.replace(0, np.nan)) * 100

    return out


def main():

    frames = []

    frames.append(load_201718_19_20(FILE_201718, "2017-2018"))
    print(f"    {len(frames[-1])} sor")

    frames.append(load_201718_19_20(FILE_201819, "2018-2019"))
    print(f"    {len(frames[-1])} sor")

    frames.append(load_201718_19_20(FILE_201920, "2019-2020"))
    print(f"    {len(frames[-1])} sor")

    frames.append(load_cleaned(FILE_202021, "2020-2021"))
    print(f"    {len(frames[-1])} sor")
    frames.append(load_fbref_season(FILE_202122, "2021-2022"))
    print(f"    {len(frames[-1])} sor")

    frames.append(load_cleaned(FILE_202223, "2022-2023"))
    print(f"    {len(frames[-1])} sor")

    frames.append(load_cleaned(FILE_202324, "2023-2024"))
    print(f"    {len(frames[-1])} sor")

    frames.append(load_202425(FILE_202425))
    print(f"    {len(frames[-1])} sor")

    df_all = pd.concat(frames, ignore_index=True)
    print(f"\nÖsszefűzés után: {len(df_all)} sor")

    df_all["league"] = df_all["league"].replace(LEAGUE_MAP)

    # oszlopsorrend
    df_all = df_all[TARGET_COLS]

    # rendezés
    df_all = df_all.sort_values(["season", "player"]).reset_index(drop=True)

    print(f"\nVégső tábla: {df_all.shape[0]} sor, {df_all.shape[1]} oszlop")
    print("Szezononként:")
    print(df_all["season"].value_counts().sort_index())

    df_all.to_csv(OUTPUT, index=False)
    print(f"\nMentve: {OUTPUT}")


if __name__ == "__main__":
    main()
