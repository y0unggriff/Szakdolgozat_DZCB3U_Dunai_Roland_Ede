import pandas as pd
import re
import hashlib

INPUT = r"C:\Egyetem\Szakdolgozat\Szakdoga adatok\merged_database.csv"
OUTPUT = r"C:\Egyetem\Szakdolgozat\Szakdoga adatok\merged_database_clean.csv"


def fix_encoding(s):

    if not isinstance(s, str):
        return s
    if not re.search(r"[ĂÄÂÃ]", s):
        return s
    try:
        fixed = s.encode("cp1250").decode("utf-8")
        if not re.search(r"[ĂÄÂÃ]", fixed):
            return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    try:
        fixed = s.encode("latin-1").decode("utf-8")
        if not re.search(r"[ĂÄÂÃ]", fixed):
            return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return s


df = pd.read_csv(INPUT)


for col in ["player", "team", "nationality"]:
    df[col] = df[col].apply(fix_encoding)


df_out = df[df["minutes"] >= 750].copy()


def season_sort_key(season):

    try:
        start = int(str(season).split("-")[0])
        return -start
    except:
        return 0


df_out["_season_sort"] = df_out["season"].apply(season_sort_key)


df_out = df_out.sort_values(by=["league", "team", "_season_sort"]).drop(
    columns=["_season_sort"]
)


def normalize_position(pos):
    if pd.isna(pos):
        return pos
    s = str(pos).replace(",", "/")

    if "/" not in s and len(s) == 4 and s.isalpha():
        return s[:2]

    if "/" in s:
        return s.split("/")[0].strip()
    return s


df_out["position"] = df_out["position"].apply(normalize_position)


def make_player_id(name):
    return hashlib.md5(str(name).encode()).hexdigest()


df_out["player_id"] = df_out["player"].apply(make_player_id)

df_out.to_csv(OUTPUT, index=False)

print(f"Sorok a tisztítás után: {len(df_out)}")
print(f"Mentve: {OUTPUT}")