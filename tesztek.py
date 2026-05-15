import re
import hashlib
import difflib
import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class TestTisztitas:

    def test_min_minutes_filter(self):
        df = pd.DataFrame({
            "player": ["A", "B", "C", "D"],
            "minutes": [749, 750, 1500, 100],
        })
        result = df[df["minutes"] >= 750].copy()

        assert len(result) == 2
        assert set(result["player"]) == {"B", "C"}

    def test_nation_prefix_removal(self):
        def remove_prefix(x):
            if pd.isna(x):
                return x
            return re.sub(r"\b[a-z]+\b", "", str(x)).strip()

        assert remove_prefix("hr CRO") == "CRO"
        assert remove_prefix("fr FRA") == "FRA"
        assert remove_prefix("JPN") == "JPN"
        assert pd.isna(remove_prefix(np.nan))

    def test_position_separator_replacement(self):
        def normalize_position(pos):
            if pd.isna(pos):
                return pos
            s = str(pos).replace(",", "/")
            if "/" not in s and len(s) == 4 and s.isalpha():
                return s[:2]
            if "/" in s:
                return s.split("/")[0].strip()
            return s

        assert normalize_position("MF,FW") == "MF"
        assert normalize_position("MFFW") == "MF"
        assert normalize_position("DF") == "DF"

    def test_season_sort_key(self):
        def season_sort_key(season):
            try:
                start = int(str(season).split("-")[0])
                return -start
            except (ValueError, AttributeError):
                return 0

        assert season_sort_key("2024-2025") < season_sort_key("2017-2018")
        assert season_sort_key("2024-2025") == -2024
        assert season_sort_key("invalid") == 0


class TestKulcsGeneralas:

    def test_make_player_id_determinisztikus(self):
        def make_player_id(name):
            return hashlib.md5(str(name).encode()).hexdigest()

        h = make_player_id("Lionel Messi")
        assert h == make_player_id("Lionel Messi")
        assert len(h) == 32
        assert make_player_id("Messi") != make_player_id("Ronaldo")


class TestAggregacio:

    def test_teli_igazolas_osszevonas(self):
        df = pd.DataFrame({
            "player": ["Messi", "Messi"],
            "season": ["2022-2023", "2022-2023"],
            "team": ["Barcelona", "Getafe"],
            "minutes": [800, 600],
            "goals": [5, 3],
        })

        agg = df.groupby(["player", "season"], as_index=False).agg(
            team=("team", lambda x: " / ".join(sorted(set(x)))),
            minutes=("minutes", "sum"),
            goals=("goals", "sum"),
        )

        assert len(agg) == 1
        assert agg["goals"].iloc[0] == 8
        assert agg["minutes"].iloc[0] == 1400
        assert "Barcelona" in agg["team"].iloc[0]
        assert "Getafe" in agg["team"].iloc[0]
        assert " / " in agg["team"].iloc[0]

    def test_nincs_igazolas_valtoztatas(self):
        df = pd.DataFrame({
            "player": ["Lewandowski"],
            "season": ["2023-2024"],
            "team": ["Barcelona"],
            "minutes": [2700],
            "goals": [19],
        })

        agg = df.groupby(["player", "season"], as_index=False).agg(
            team=("team", lambda x: " / ".join(sorted(set(x)))),
            minutes=("minutes", "sum"),
            goals=("goals", "sum"),
        )

        assert len(agg) == 1
        assert agg["team"].iloc[0] == "Barcelona"
        assert agg["goals"].iloc[0] == 19

    def test_per90_atlagolas(self):
        df = pd.DataFrame({
            "player": ["X", "X"],
            "season": ["2022-2023", "2022-2023"],
            "xG/90": [0.4, 0.6],
        })

        agg = df.groupby(["player", "season"], as_index=False).agg(
            **{"xG/90": ("xG/90", "mean")}
        )
        assert agg["xG/90"].iloc[0] == pytest.approx(0.5)

    def test_pozicio_leggyakoribb(self):
        df = pd.DataFrame({
            "player": ["X", "X", "X"],
            "season": ["2022-2023"] * 3,
            "position": ["MF", "MF", "FW"],
        })

        agg = df.groupby(["player", "season"], as_index=False).agg(
            position=("position", lambda x: x.mode()[0] if not x.mode().empty else None)
        )
        assert agg["position"].iloc[0] == "MF"


class TestSzetvalasztas:

    def test_gk_es_mezonyjatekos_szetvalik(self):
        df = pd.DataFrame({
            "player": ["Kapus1", "Csatar1", "Kapus2", "Vedo1"],
            "position": ["GK", "FW", "GK", "DF"],
        })

        df_gk = df[df["position"] == "GK"].copy()
        df_of = df[df["position"] != "GK"].copy()

        assert len(df_gk) == 2
        assert len(df_of) == 2
        assert set(df_gk["player"]) == {"Kapus1", "Kapus2"}

    def test_specifikus_oszlopok_megmaradnak(self):
        df = pd.DataFrame({
            "player": ["Kapus", "Csatar"],
            "position": ["GK", "FW"],
            "goals": [0, 15],
            "saves": [80, 0],
        })

        outfield_cols = ["player", "position", "goals"]
        goalkeeper_cols = ["player", "position", "saves"]

        df_of = df[df["position"] != "GK"][outfield_cols].copy()
        df_gk = df[df["position"] == "GK"][goalkeeper_cols].copy()

        assert "saves" not in df_of.columns
        assert "goals" not in df_gk.columns


class TestBestMatch:

    @staticmethod
    def best_match(name, candidates):
        best_name = None
        best_score = -1
        name_lower = name.lower().strip()

        for candidate in candidates:
            cand_lower = str(candidate).lower().strip()
            seq_score = difflib.SequenceMatcher(None, name_lower, cand_lower).ratio()

            if name_lower in cand_lower or cand_lower in name_lower:
                longer = max(len(name_lower), len(cand_lower))
                shorter = min(len(name_lower), len(cand_lower))
                partial_score = 0.85 + (shorter / longer) * 0.15
            else:
                partial_score = 0

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

    def test_pontos_egyezes(self):
        name, score = self.best_match("Lionel Messi", ["Lionel Messi", "Cristiano Ronaldo"])
        assert name == "Lionel Messi"
        assert score == 100.0

    def test_ures_jelolt_lista(self):
        name, score = self.best_match("Bárki", [])
        assert name is None

    def test_reszleges_egyezes(self):
        name, score = self.best_match("Lucas Gaya", ["Jose Lucas Gaya", "Random Játékos"])
        assert name == "Jose Lucas Gaya"
        assert score >= 85.0

    def test_ekezet_kulonbseg(self):
        name, score = self.best_match("N'Golo Kante", ["N'Golo Kanté", "Random Játékos"])
        assert name == "N'Golo Kanté"
        assert score >= 70.0


class TestCsapatNevNormalizalas:

    def test_csapat_mappolas(self):
        TEAM_MAP = {
            "Manchester United": "Manchester Utd",
            "Wolverhampton Wanderers": "Wolves",
        }

        def normalize(team_str):
            if pd.isna(team_str):
                return []
            parts = [t.strip() for t in str(team_str).split(",")]
            return [TEAM_MAP.get(p, p) for p in parts]

        assert normalize("Manchester United") == ["Manchester Utd"]
        assert normalize("Liverpool") == ["Liverpool"]
        result = normalize("Manchester United, Wolverhampton Wanderers")
        assert result == ["Manchester Utd", "Wolves"]


class TestFeatureEngineering:

    def test_prev_shift_jatekosonkent(self):
        df = pd.DataFrame({
            "player_id": ["A", "A", "B", "B"],
            "season_start": [2022, 2023, 2022, 2023],
            "goals": [10, 12, 5, 7],
        }).sort_values(["player_id", "season_start"]).reset_index(drop=True)

        df["prev_goals"] = df.groupby("player_id")["goals"].shift(1)

        a_2023 = df[(df["player_id"] == "A") & (df["season_start"] == 2023)]
        assert a_2023["prev_goals"].iloc[0] == 10
        b_2023 = df[(df["player_id"] == "B") & (df["season_start"] == 2023)]
        assert b_2023["prev_goals"].iloc[0] == 5
        a_2022 = df[(df["player_id"] == "A") & (df["season_start"] == 2022)]
        assert pd.isna(a_2022["prev_goals"].iloc[0])

    def test_prev2_ket_szezonnal_korabbi(self):
        df = pd.DataFrame({
            "player_id": ["A"] * 4,
            "season_start": [2021, 2022, 2023, 2024],
            "goals": [5, 10, 15, 20],
        }).sort_values(["player_id", "season_start"]).reset_index(drop=True)

        df["prev2_goals"] = df.groupby("player_id")["goals"].shift(2)

        assert df[df["season_start"] == 2023]["prev2_goals"].iloc[0] == 5
        assert df[df["season_start"] == 2024]["prev2_goals"].iloc[0] == 10

    def test_age_next_szamitas(self):
        df = pd.DataFrame({"age": [18, 25, 33, 40]})
        df["age_next"] = df["age"] + 1
        assert (df["age_next"] - df["age"] == 1).all()
        assert df["age_next"].tolist() == [19, 26, 34, 41]

    def test_per90_szamolas(self):
        df = pd.DataFrame({"minutes": [900, 1800, 0], "goals": [10, 18, 0]})
        df["goals_p90"] = df["goals"] / (df["minutes"] / 90).replace(0, np.nan)
        df["goals_p90"] = df["goals_p90"].fillna(0)

        assert df["goals_p90"].iloc[0] == pytest.approx(1.0)
        assert df["goals_p90"].iloc[1] == pytest.approx(0.9)
        assert df["goals_p90"].iloc[2] == 0

    def test_age_pos_lookup_fallback(self):
        age_pos_goals = {(25, "FW"): 15.0}
        pos_avg = {"FW": 10.0, "MF": 5.0}

        def lookup(age, pos):
            return age_pos_goals.get((age, pos), pos_avg.get(pos, 0))

        assert lookup(25, "FW") == 15.0
        assert lookup(18, "FW") == 10.0
        assert lookup(25, "GK") == 0


class TestEvaluate:

    @staticmethod
    def evaluate(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R²": r2_score(y_true, y_pred),
        }

    def test_tokeletes_predikcio(self):
        m = self.evaluate([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert m["MAE"] == 0
        assert m["RMSE"] == 0
        assert m["R²"] == pytest.approx(1.0)

    def test_ismert_kezi_szamitas(self):
        y_true = [0, 1, 2, 3, 4]
        y_pred = [1, 1, 2, 3, 5]
        m = self.evaluate(y_true, y_pred)
        assert m["MAE"] == pytest.approx(0.4)
        assert m["RMSE"] == pytest.approx(np.sqrt(0.4))

    def test_konstans_predikcio_r2_nullaban(self):
        m = self.evaluate([1, 2, 3, 4, 5], [3, 3, 3, 3, 3])
        assert m["R²"] == pytest.approx(0.0)


class TestTrainTestSplit:

    def test_szezon_alapu_split(self):
        df = pd.DataFrame({
            "season_start": [2020, 2021, 2022, 2023, 2024],
        })
        train = df[df["season_start"] <= 2022]
        test = df[df["season_start"] >= 2023]

        assert len(train) == 3
        assert len(test) == 2
        assert len(set(train["season_start"]) & set(test["season_start"])) == 0


class TestEnsembleSulyozas:

    def test_sulyok_osszege_egy(self):
        cv_scores = {"Ridge": 0.5, "RF": 0.55, "XGBoost": 0.58}
        weights = {name: max(score, 0.01) for name, score in cv_scores.items()}
        total = sum(weights.values())
        weights = {name: w / total for name, w in weights.items()}

        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights["XGBoost"] > weights["Ridge"]

    def test_ensemble_predikcio_sulyozott_atlag(self):
        preds = {
            "M1": np.array([1.0, 2.0]),
            "M2": np.array([3.0, 4.0]),
        }
        weights = {"M1": 0.5, "M2": 0.5}

        ensemble = np.zeros_like(preds["M1"])
        for name, w in weights.items():
            ensemble += w * preds[name]

        assert ensemble[0] == pytest.approx(2.0)
        assert ensemble[1] == pytest.approx(3.0)


class TestPredikcioUtofeldolgozas:

    def test_log_visszatranszformacio(self):
        y_log = np.array([np.log1p(5), np.log1p(10)])
        y = np.expm1(y_log)
        assert y[0] == pytest.approx(5.0)
        assert y[1] == pytest.approx(10.0)

    def test_predikcio_clip_es_nan_kezeles(self):
        y_pred = np.array([-2.5, 0.5, 500.0, np.nan, np.inf])
        y_clipped = np.clip(y_pred, 0, 100)
        y_final = np.nan_to_num(y_clipped, nan=0.0, posinf=100.0, neginf=0.0)

        assert y_final[0] == 0
        assert y_final[1] == 0.5
        assert y_final[2] == 100
        assert y_final[3] == 0
        assert y_final[4] == 100