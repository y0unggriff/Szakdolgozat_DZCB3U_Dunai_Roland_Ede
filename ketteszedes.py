import pandas as pd

INPUT = r"C:\Egyetem\Szakdolgozat\Szakdoga adatok\merged_database_clean_aggregated.csv"
OUTPUT_OF = r"C:\Egyetem\Szakdolgozat\Szakdoga adatok\outfield_players.csv"
OUTPUT_GK = r"C:\Egyetem\Szakdolgozat\Szakdoga adatok\goalkeepers.csv"


COMMON_COLS = [
    "player",
    "player_id",
    "league",
    "season",
    "team",
    "nationality",
    "position",
    "age",
    "matches_played",
    "minutes",
]

OUTFIELD_COLS = COMMON_COLS + [
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
]


GOALKEEPER_COLS = COMMON_COLS + [
    "total_passes",
    "passes_completed",
    "clearances",
    "errors",
    "goals_against",
    "saves",
    "saves_%",
    "clean sheets",
    "penatly_saves_%",
]


def main():
    print(f"Betöltés: {INPUT}")
    df = pd.read_csv(INPUT)
    print(f"  Összes sor: {len(df)}")

    of_mask = df["position"] != "GK"
    gk_mask = df["position"] == "GK"

    df_of = df[of_mask][OUTFIELD_COLS].copy()
    df_gk = df[gk_mask][GOALKEEPER_COLS].copy()

    print()
    print(f"Mezőnyjátékosok: {len(df_of)} sor, {len(df_of.columns)} oszlop")
    print(f"Kapusok:         {len(df_gk)} sor, {len(df_gk.columns)} oszlop")

    df_of.to_csv(OUTPUT_OF, index=False)
    df_gk.to_csv(OUTPUT_GK, index=False)

    print()
    print(f"Mentve: {OUTPUT_OF}")
    print(f"Mentve: {OUTPUT_GK}")


if __name__ == "__main__":
    main()
