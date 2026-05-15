import pandas as pd
import difflib

INPUT = r"C:\Egyetem\Szakdolgozat\Szakdoga adatok\outfield_players.csv"
OUTPUT = r"C:\Egyetem\Szakdolgozat\Szakdoga adatok\outfield_players_with_xg.csv"
DATA_DIR = r"C:\Egyetem\Szakdolgozat\Szakdoga adatok"


Season_files = [
    (rf"{DATA_DIR}\epl2122.csv", "Premier League", "2021-2022"),
    (rf"{DATA_DIR}\epl2223.csv", "Premier League", "2022-2023"),
    (rf"{DATA_DIR}\epl2324.csv", "Premier League", "2023-2024"),
    (rf"{DATA_DIR}\epl2425.csv", "Premier League", "2024-2025"),
    (rf"{DATA_DIR}\ligue12122.csv", "Ligue 1", "2021-2022"),
    (rf"{DATA_DIR}\ligue12223.csv", "Ligue 1", "2022-2023"),
    (rf"{DATA_DIR}\ligue12324.csv", "Ligue 1", "2023-2024"),
    (rf"{DATA_DIR}\ligue12425.csv", "Ligue 1", "2024-2025"),
    (rf"{DATA_DIR}\seriaa2122.csv", "Serie A", "2021-2022"),
    (rf"{DATA_DIR}\seriaa2223.csv", "Serie A", "2022-2023"),
    (rf"{DATA_DIR}\seriaa2324.csv", "Serie A", "2023-2024"),
    (rf"{DATA_DIR}\seriaa2425.csv", "Serie A", "2024-2025"),
    (rf"{DATA_DIR}\bundesliga2122.csv", "Bundesliga", "2021-2022"),
    (rf"{DATA_DIR}\bundesliga2223.csv", "Bundesliga", "2022-2023"),
    (rf"{DATA_DIR}\bundesliga2324.csv", "Bundesliga", "2023-2024"),
    (rf"{DATA_DIR}\bundesliga2425.csv", "Bundesliga", "2024-2025"),
    (rf"{DATA_DIR}\laliga2122.csv", "La Liga", "2021-2022"),
    (rf"{DATA_DIR}\laliga2223.csv", "La Liga", "2022-2023"),
    (rf"{DATA_DIR}\laliga2324.csv", "La Liga", "2023-2024"),
    (rf"{DATA_DIR}\laliga2425.csv", "La Liga", "2024-2025"),
]


TEAM_MAP_XG_TO_CLEAN = {
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Sheffield United": "Sheffield Utd",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Norwich": "Norwich City",
    "Wolverhampton Wanderers": "Wolves",
    "West Bromwich Albion": "West Brom",
    "Nottingham Forest": "Nott'ham Forest",
    "Luton": "Luton Town",
    "Ipswich": "Ipswich Town",
    "Borussia M.Gladbach": "M'Gladbach",
    "Bayer Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "Eintracht Frankfurt": "Eint Frankfurt",
    "FC Cologne": "Köln",
    "Cologne": "Köln",
    "RasenBallsport Leipzig": "RB Leipzig",
    "VfB Stuttgart": "Stuttgart",
    "Hertha Berlin": "Hertha BSC",
    "Greuther Fuerth": "Greuther Fürth",
    "Darmstadt": "Darmstadt 98",
    "FC Heidenheim": "Heidenheim",
    "Holstein Kiel": "Holstein Kiel",
    "St. Pauli": "St. Pauli",
    "Internazionale": "Inter",
    "AC Milan": "Milan",
    "Hellas Verona": "Hellas Verona",
    "Verona": "Hellas Verona",
    "Paris Saint Germain": "Paris S-G",
    "Paris Saint-Germain": "Paris S-G",
    "PSG": "Paris S-G",
    "Saint-Étienne": "Saint-Étienne",
    "Atletico Madrid": "Atlético Madrid",
    "Athletic Bilbao": "Athletic Club",
    "Athletic Club Bilbao": "Athletic Club",
    "Real Betis": "Betis",
    "Almeria": "Almería",
    "Cadiz": "Cádiz",
    "Leganes": "Leganés",
    "Alaves": "Alavés",
    "Real Valladolid": "Valladolid",
}


SIMILARITY_THRESHOLD = 70


def best_match(name, candidates):

    best_name = None
    best_score = -1
    name_lower = name.lower().strip()

    for candidate in candidates:
        cand_lower = str(candidate).lower().strip()

        # SequenceMatcher arány
        seq_score = difflib.SequenceMatcher(None, name_lower, cand_lower).ratio()

        # Részleges egyezés ha az egyik név benne van a másikban
        if name_lower in cand_lower or cand_lower in name_lower:
            longer = max(len(name_lower), len(cand_lower))
            shorter = min(len(name_lower), len(cand_lower))
            partial_score = 0.85 + (shorter / longer) * 0.15
        else:
            partial_score = 0

        # Jaccard egyezés
        name_tokens = set(name_lower.split())
        cand_tokens = set(cand_lower.split())
        if name_tokens and cand_tokens:
            jaccard = len(name_tokens & cand_tokens) / len(name_tokens | cand_tokens)
        else:
            jaccard = 0

        final_score = max(seq_score, partial_score, jaccard)

        if final_score > best_score:
            best_score = final_score
            best_name = candidate

    return best_name, round(best_score * 100, 1)


def normalize_team(team_str):

    if pd.isna(team_str):
        return []
    parts = [t.strip() for t in str(team_str).split(",")]
    return [TEAM_MAP_XG_TO_CLEAN.get(p, p) for p in parts]


outfield = pd.read_csv(INPUT)

mask_missing = outfield["xG"].isna()
outfield_missing = outfield[mask_missing].copy()
outfield_complete = outfield[~mask_missing].copy()

print(f"Kitöltött sorok :   {len(outfield_complete)}")
print(f"Kitöltetlen sorok: {len(outfield_missing)}")
print(f"Hiányzó xG szezonok: {sorted(outfield_missing['season'].unique())}")


dfs = []
for fname, lg, season in Season_files:
    df = pd.read_csv(fname, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip().capitalize() for c in df.columns]
    df["_source_league"] = lg
    df["_source_season"] = season
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)


for col in ["Xg", "Xa", "Xg90", "Xa90", "Goals", "A"]:
    if col in combined.columns:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")


if "A" in combined.columns and "Assists" not in combined.columns:
    combined["Assists"] = combined["A"]


combined["_team_norm_list"] = combined["Team"].apply(
    lambda x: [t.lower().strip() for t in normalize_team(x)]
)


results = []

for _, row in outfield_missing.iterrows():
    player_name = str(row["player"])
    season = str(row["season"])
    league = row["league"]
    team = str(row["team"])
    gls = pd.to_numeric(row["goals"], errors="coerce")
    ast = pd.to_numeric(row["assists"], errors="coerce")

    own_teams = [t.strip().lower() for t in team.split(" / ")]

    subset = combined[
        (combined["_source_season"] == season)
        & (combined["_source_league"] == league)
        & (
            combined["_team_norm_list"].apply(
                lambda team_list: any(t in team_list for t in own_teams)
            )
        )
    ]
    team_matched = not subset.empty

    if subset.empty:
        subset = combined[
            (combined["_source_season"] == season)
            & (combined["_source_league"] == league)
        ]

    if subset.empty:
        results.append(
            {
                "player": player_name,
                "season": season,
                "league": league,
                "team": team,
                "best_match_player": None,
                "similarity_score": None,
                "goals_matched": False,
                "assists_matched": False,
                "team_matched": team_matched,
                "match_method": "no_match",
                "xG": None,
                "xA": None,
                "xG/90": None,
                "xA/90": None,
            }
        )
        continue

    goals_matched = False
    assists_matched = False

    if pd.notna(gls) and pd.notna(ast) and "Assists" in subset.columns:
        subset_ga = subset[(subset["Goals"] == gls) & (subset["Assists"] == ast)]
        if not subset_ga.empty:
            subset = subset_ga
            goals_matched = True
            assists_matched = True
        else:
            subset_g = subset[subset["Goals"] == gls]
            if not subset_g.empty:
                subset = subset_g
                goals_matched = True
            else:
                subset_a = subset[subset["Assists"] == ast]
                if not subset_a.empty:
                    subset = subset_a
                    assists_matched = True
    elif pd.notna(gls):
        subset_g = subset[subset["Goals"] == gls]
        if not subset_g.empty:
            subset = subset_g
            goals_matched = True
    elif pd.notna(ast) and "Assists" in subset.columns:
        subset_a = subset[subset["Assists"] == ast]
        if not subset_a.empty:
            subset = subset_a
            assists_matched = True

    candidates = subset["Player"].dropna().tolist()
    best_name, score = best_match(player_name, candidates)

    if (
        best_name
        and score >= SIMILARITY_THRESHOLD
        and not subset[subset["Player"] == best_name].empty
    ):
        matched_row = subset[subset["Player"] == best_name].iloc[0]
        xg = matched_row.get("Xg")
        xa = matched_row.get("Xa")
        xg90 = matched_row.get("Xg90")
        xa90 = matched_row.get("Xa90")
    else:
        xg = xa = xg90 = xa90 = None

    if team_matched and goals_matched and assists_matched:
        match_method = "team+goals+assists+name"
    elif team_matched and goals_matched:
        match_method = "team+goals+name"
    elif team_matched and assists_matched:
        match_method = "team+assists+name"
    elif team_matched:
        match_method = "team+name"
    elif goals_matched and assists_matched:
        match_method = "league+goals+assists+name"
    elif goals_matched:
        match_method = "league+goals+name"
    elif assists_matched:
        match_method = "league+assists+name"
    else:
        match_method = "league+name"

    results.append(
        {
            "player": player_name,
            "season": season,
            "league": league,
            "team": team,
            "best_match_player": best_name,
            "similarity_score": score,
            "goals_matched": goals_matched,
            "assists_matched": assists_matched,
            "team_matched": team_matched,
            "match_method": match_method,
            "xG": xg,
            "xA": xa,
            "xG/90": xg90,
            "xA/90": xa90,
        }
    )

results_df = pd.DataFrame(results)


xg_cols = results_df[["player", "season", "league", "xG", "xA", "xG/90", "xA/90"]]

outfield_missing_filled = outfield_missing.merge(
    xg_cols, on=["player", "season", "league"], how="left", suffixes=("_orig", "_new")
)

for col in ["xG", "xA", "xG/90", "xA/90"]:
    outfield_missing_filled[col] = outfield_missing_filled[f"{col}_new"]
    outfield_missing_filled = outfield_missing_filled.drop(
        columns=[f"{col}_orig", f"{col}_new"]
    )


outfield_final = pd.concat(
    [outfield_complete, outfield_missing_filled], ignore_index=True
)
outfield_final.to_csv(OUTPUT, index=False)


print(
    f" alacsony egyezés: {(results_df['similarity_score'] < SIMILARITY_THRESHOLD).sum()}"
)



low = results_df[results_df["similarity_score"] < SIMILARITY_THRESHOLD].copy()
if not low.empty:
    season_order = ["2021-2022", "2022-2023"]
    league_order = ["Premier League", "Ligue 1", "Serie A", "Bundesliga", "La Liga"]
    low["_s"] = pd.Categorical(low["season"], categories=season_order, ordered=True)
    low["_l"] = pd.Categorical(low["league"], categories=league_order, ordered=True)
    low = low.sort_values(["_s", "_l"]).drop(columns=["_s", "_l"])

    print()
    print("Alacsony egyezések:")
    print(
        low[
            [
                "player",
                "season",
                "league",
                "team",
                "best_match_player",
                "similarity_score",
            ]
        ].to_string()
    )

print(f"\nMentve: {OUTPUT}")
