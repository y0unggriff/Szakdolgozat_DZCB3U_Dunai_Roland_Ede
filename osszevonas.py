import pandas as pd

INPUT = r"C:\Egyetem\Szakdolgozat\Szakdoga adatok\merged_database_clean.csv"
OUTPUT = r"C:\Egyetem\Szakdolgozat\Szakdoga adatok\merged_database_clean_aggregated.csv"


sum_cols = [
    "matches_played",
    "minutes",
    "goals",
    "assists",
    "xG",
    "xA",
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
    "clean sheets",
]


avg_cols = ["xG/90", "xA/90", "saves_%", "penatly_saves_%"]


group_keys = ["player", "season"]


def aggregate_df(df, sum_cols, avg_cols, group_keys):
    agg_dict = {}

    agg_dict["team"] = (
        "team",
        lambda x: " / ".join(sorted(set(str(v) for v in x if pd.notna(v)))),
    )
    agg_dict["league"] = (
        "league",
        lambda x: " / ".join(sorted(set(str(v) for v in x if pd.notna(v)))),
    )

    agg_dict["nationality"] = (
        "nationality",
        lambda x: x.mode()[0] if not x.mode().empty else None,
    )
    agg_dict["position"] = (
        "position",
        lambda x: x.mode()[0] if not x.mode().empty else None,
    )

    agg_dict["age"] = ("age", "max")

    agg_dict["player_id"] = ("player_id", "first")


    for col in sum_cols:
        if col in df.columns:
            agg_dict[col] = (col, lambda x: x.sum(min_count=1))

    for col in avg_cols:
        if col in df.columns:
            agg_dict[col] = (col, "mean")

    return df.groupby(group_keys, as_index=False).agg(**agg_dict)


df = pd.read_csv(INPUT)
print(f"Betöltött sorok: {len(df)}")

df_agg = aggregate_df(df, sum_cols, avg_cols, group_keys)
print(f"Aggregálás után: {len(df_agg)} sor ({len(df) - len(df_agg)} sor összevonva)")


TARGET_COLS = [
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
df_agg = df_agg[[c for c in TARGET_COLS if c in df_agg.columns]]

df_agg.to_csv(OUTPUT, index=False)
print(f"Mentve: {OUTPUT}")
