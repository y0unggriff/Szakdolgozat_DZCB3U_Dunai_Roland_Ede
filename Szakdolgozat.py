import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
from sklearn.ensemble import GradientBoostingRegressor

# Beallitasok

CSV_PATH = r"C:\Egyetem\Szakdolgozat\Szakdoga adatok\outfield_players.csv"
OUTPUT_DIR = os.path.dirname(CSV_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# Permutation importance
PERM_N_REPEATS = 10
PERM_MAX_SAMPLES = 2000
HAS_XGB = True


def out(filename):
    return os.path.join(OUTPUT_DIR, filename)


# Adatbetoltes es eloszűres

df = pd.read_csv(CSV_PATH)
df = df[df["matches_played"] >= 5].copy()

main_leagues = ["Premier League", "Bundesliga", "La Liga", "Ligue 1", "Serie A"]
df = df[df["league"].isin(main_leagues)].copy()

df["season_start"] = df["season"].str.split("-").str[0].astype(int)
df = df.sort_values(["player_id", "season_start"]).reset_index(drop=True)


#  Feature engineering


per90_cols = [
    "goals",
    "assists",
    "shots_total",
    "shots_on_target",
    "shot_creating_action",
    "goal_creating_action",
    "progressive_passes",
    "key_passes",
    "tackles_won",
    "interceptions",
    "take_ons_won",
]

for col in per90_cols:
    df[f"{col}_p90"] = df[col] / (df["minutes"] / 90).replace(0, np.nan)

df[[f"{c}_p90" for c in per90_cols]] = df[[f"{c}_p90" for c in per90_cols]].fillna(0)

# Hatekonysagi mutatok
df["shot_accuracy"] = df["shots_on_target"] / df["shots_total"].replace(0, np.nan)
df["pass_accuracy"] = df["passes_completed"] / df["total_passes"].replace(0, np.nan)
df["take_on_success"] = df["take_ons_won"] / df["total_take_ons"].replace(0, np.nan)
df["conversion"] = df["goals"] / df["shots_on_target"].replace(0, np.nan)
df["xG_overperformance"] = df["goals"] - df["xG"]
df["xA_overperformance"] = df["assists"] - df["xA"]
df = df.fillna(0)

# Csapatszintu kontextus
df["team_total_goals"] = df.groupby(["team", "season"])["goals"].transform("sum")
df["team_total_assists"] = df.groupby(["team", "season"])["assists"].transform("sum")

df["position_orig"] = df["position"].copy()
df = pd.get_dummies(df, columns=["position"], prefix="pos")


# Idobeli parok elozo szezon

feature_cols = [
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
    "goals_p90",
    "assists_p90",
    "shots_total_p90",
    "shots_on_target_p90",
    "shot_creating_action",
    "goal_creating_action",
    "shot_creating_action_p90",
    "goal_creating_action_p90",
    "progressive_passes",
    "key_passes",
    "progressive_passes_p90",
    "key_passes_p90",
    "tackles_won",
    "interceptions",
    "take_ons_won",
    "tackles_won_p90",
    "interceptions_p90",
    "take_ons_won_p90",
    "shot_accuracy",
    "pass_accuracy",
    "take_on_success",
    "conversion",
    "xG_overperformance",
    "xA_overperformance",
    "team_total_goals",
    "team_total_assists",
    "pos_DF",
    "pos_FW",
    "pos_MF",
]

# Elozo szezon (t-1)
for col in feature_cols:
    df[f"prev_{col}"] = df.groupby("player_id")[col].shift(1)

# Ket szezonnal ezelotti adatok (t-2)
prev2_only = ["goals", "assists", "xG", "xA", "minutes", "goals_p90", "assists_p90"]
for col in prev2_only:
    df[f"prev2_{col}"] = df.groupby("player_id")[col].shift(2)


train_for_age = df[(df["season_start"] <= 2022)].copy()
train_for_age["_pos"] = train_for_age["position_orig"]
df["_pos"] = df["position_orig"]

age_pos_goals = train_for_age.groupby(["age", "_pos"])["goals"].mean().to_dict()
age_pos_assist = train_for_age.groupby(["age", "_pos"])["assists"].mean().to_dict()
age_pos_xG = train_for_age.groupby(["age", "_pos"])["xG"].mean().to_dict()
age_pos_xA = train_for_age.groupby(["age", "_pos"])["xA"].mean().to_dict()

pos_goals_avg = train_for_age.groupby("_pos")["goals"].mean().to_dict()
pos_assist_avg = train_for_age.groupby("_pos")["assists"].mean().to_dict()
pos_xG_avg = train_for_age.groupby("_pos")["xG"].mean().to_dict()
pos_xA_avg = train_for_age.groupby("_pos")["xA"].mean().to_dict()


def age_pos_lookup(row, mapping, fallback):
    key = (row["age"], row["_pos"])
    if key in mapping:
        return mapping[key]
    return fallback.get(row["_pos"], 0)


df["age_pos_avg_goals"] = df.apply(
    lambda r: age_pos_lookup(r, age_pos_goals, pos_goals_avg), axis=1
)
df["age_pos_avg_assists"] = df.apply(
    lambda r: age_pos_lookup(r, age_pos_assist, pos_assist_avg), axis=1
)
df["age_pos_avg_xG"] = df.apply(
    lambda r: age_pos_lookup(r, age_pos_xG, pos_xG_avg), axis=1
)
df["age_pos_avg_xA"] = df.apply(
    lambda r: age_pos_lookup(r, age_pos_xA, pos_xA_avg), axis=1
)


df["prev_season_start"] = df.groupby("player_id")["season_start"].shift(1)
df["prev2_season_start"] = df.groupby("player_id")["season_start"].shift(2)

valid = (df["season_start"] - df["prev_season_start"] == 1)
df_model = df[valid].copy()

df_model["has_prev2"] = (
    df_model["season_start"] - df_model["prev2_season_start"] == 2
).astype(int)
prev2_cols = [
    c for c in df_model.columns if c.startswith("prev2_") and c != "prev2_season_start"
]
df_model[prev2_cols] = df_model[prev2_cols].fillna(0)

age_pos_cols = [
    "age_pos_avg_goals",
    "age_pos_avg_assists",
    "age_pos_avg_xG",
    "age_pos_avg_xA",
]
df_model[age_pos_cols] = df_model[age_pos_cols].fillna(0)

model_features = (
    [f"prev_{c}" for c in feature_cols]
    + prev2_cols
    + age_pos_cols
    + ["has_prev2", "age"]
)

X = df_model[model_features].fillna(0)

X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float64)

y_goals = df_model["goals"]
y_assists = df_model["assists"]

# Idobeli split
train_mask = df_model["season_start"] <= 2022
test_mask = df_model["season_start"] >= 2023

X_tr, X_te = X[train_mask], X[test_mask]
yg_tr, yg_te = y_goals[train_mask], y_goals[test_mask]
ya_tr, ya_te = y_assists[train_mask], y_assists[test_mask]


#  Modellek es hiperparameter kereses


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R²": r2_score(y_true, y_pred),
    }


def tune_and_fit(X_tr, y_tr, target_name):
    print(f"\nHiperparameter kereses - {target_name}")
    cv = KFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    y_tr_log = np.log1p(y_tr)

    ridge_alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    lr = RidgeCV(alphas=ridge_alphas)
    lr.fit(X_tr_sc, y_tr_log)
    lr_cv = cross_val_score(
        RidgeCV(alphas=ridge_alphas), X_tr_sc, y_tr_log, cv=cv, scoring="r2", n_jobs=-1
    ).mean()

    rf = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators": [200, 400, 600],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.5, 0.7],
        },
        n_iter=15,
        cv=cv,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr_log)

    if HAS_XGB:
        xgb = RandomizedSearchCV(
            XGBRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
                verbosity=0,
                objective="reg:tweedie",
                tweedie_variance_power=1.5,
            ),
            {
                "n_estimators": [300, 500, 800],
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "min_child_weight": [1, 3, 5],
                "reg_alpha": [0, 0.1, 1],
                "reg_lambda": [0.5, 1, 2],
            },
            n_iter=25,
            cv=cv,
            scoring="r2",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        xgb.fit(X_tr, y_tr)  # nyers y, nem log
        gbm_model = xgb.best_estimator_
        gbm_score = xgb.best_score_
        gbm_name = "XGBoost"
    else:
        gb = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "n_estimators": [200, 400, 600],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.03, 0.05, 0.1],
                "subsample": [0.7, 0.85, 1.0],
            },
            n_iter=15,
            cv=cv,
            scoring="r2",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        gb.fit(X_tr, y_tr_log)
        gbm_model = gb.best_estimator_
        gbm_score = gb.best_score_
        gbm_name = "Gradient Boosting"

    cv_scores = {
        "Linear Regression": lr_cv,
        "Random Forest": rf.best_score_,
        gbm_name: gbm_score,
    }

    # CV eredmenyek osszevontan
    print(
        f"  CV R²:  Ridge={lr_cv:.3f}  RF={rf.best_score_:.3f}  {gbm_name}={gbm_score:.3f}"
    )

    models = {
        "Linear Regression": (lr, scaler, True, "log"),
        "Random Forest": (rf.best_estimator_, None, False, "log"),
        gbm_name: (gbm_model, None, False, "raw"),
    }
    return models, cv_scores, gbm_name


def predict_all(models_dict, X_te):
    preds = {}
    for name, (model, scaler, needs_scale, fit_target) in models_dict.items():
        X_input = scaler.transform(X_te) if needs_scale else X_te
        y_pred = model.predict(X_input)

        if fit_target == "log":
            y_pred = np.clip(y_pred, -10, np.log(101))
            y_pred = np.expm1(y_pred)

        y_pred = np.clip(y_pred, 0, 100)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=100.0, neginf=0.0)
        preds[name] = y_pred
    return preds


def build_ensemble(preds_dict, cv_scores):

    weights = {name: max(score, 0.01) for name, score in cv_scores.items()}
    total = sum(weights.values())
    weights = {name: w / total for name, w in weights.items()}

    ensemble_pred = np.zeros_like(preds_dict["Linear Regression"])
    for name, w in weights.items():
        ensemble_pred += w * preds_dict[name]
    return ensemble_pred, weights


# Tanitas golok

models_g, cv_scores_g, gbm_name_g = tune_and_fit(X_tr, yg_tr, "Golok")
preds_g = predict_all(models_g, X_te)
ensemble_g, weights_g = build_ensemble(preds_g, cv_scores_g)
preds_g["Ensemble"] = ensemble_g
preds_g["BASELINE (tavalyi)"] = X_te["prev_goals"].values


# Tanitas golpasszok

models_a, cv_scores_a, gbm_name_a = tune_and_fit(X_tr, ya_tr, "Golpasszok")
preds_a = predict_all(models_a, X_te)
ensemble_a, weights_a = build_ensemble(preds_a, cv_scores_a)
preds_a["Ensemble"] = ensemble_a
preds_a["BASELINE (tavalyi)"] = X_te["prev_assists"].values


# Kiertekeles

print("\nEredmenyek - golok:")
results_g = {name: evaluate(yg_te, pred) for name, pred in preds_g.items()}
df_results_g = pd.DataFrame(results_g).T.round(3)
print(df_results_g.to_string())

print("\nEredmenyek - golpasszok:")
results_a = {name: evaluate(ya_te, pred) for name, pred in preds_a.items()}
df_results_a = pd.DataFrame(results_a).T.round(3)
print(df_results_a.to_string())

df_results_g.to_csv(out("results_goals.csv"))
df_results_a.to_csv(out("results_assists.csv"))

# Long formatum a Power BIhoz
results_long_g = (
    df_results_g.reset_index()
    .melt(id_vars="index", var_name="metric", value_name="value")
    .rename(columns={"index": "model"})
)
results_long_g["target"] = "Golok"

results_long_a = (
    df_results_a.reset_index()
    .melt(id_vars="index", var_name="metric", value_name="value")
    .rename(columns={"index": "model"})
)
results_long_a["target"] = "Golpasszok"

results_long = pd.concat([results_long_g, results_long_a], ignore_index=True)
results_long = results_long[["target", "model", "metric", "value"]]
results_long["value"] = results_long["value"].round(4)
results_long.to_csv(out("results_long.csv"), index=False)


# Reszletes predikcios CSV

test_info = df_model[test_mask][
    [
        "player_id",
        "season",
        "season_start",
        "team",
        "league",
        "position_orig",
        "age",
        "matches_played",
        "minutes",
    ]
].copy()
player_names = df[["player_id", "season", "player"]].drop_duplicates()
test_info = test_info.merge(player_names, on=["player_id", "season"], how="left")

predictions_df = pd.DataFrame(
    {
        "player_id": test_info["player_id"].values,
        "player": test_info["player"].values,
        "team": test_info["team"].values,
        "league": test_info["league"].values,
        "position": test_info["position_orig"].values,
        "season": test_info["season"].values,
        "age": test_info["age"].values,
        "matches_played": test_info["matches_played"].values,
        "minutes": test_info["minutes"].values,
        "actual_goals": yg_te.values,
        "actual_assists": ya_te.values,
        "pred_goals_LR": preds_g["Linear Regression"].round(2),
        "pred_goals_RF": preds_g["Random Forest"].round(2),
        f"pred_goals_{gbm_name_g}": preds_g[gbm_name_g].round(2),
        "pred_goals_Ensemble": preds_g["Ensemble"].round(2),
        "pred_assists_LR": preds_a["Linear Regression"].round(2),
        "pred_assists_RF": preds_a["Random Forest"].round(2),
        f"pred_assists_{gbm_name_a}": preds_a[gbm_name_a].round(2),
        "pred_assists_Ensemble": preds_a["Ensemble"].round(2),
    }
)

predictions_df["error_goals_Ensemble"] = (
    predictions_df["pred_goals_Ensemble"] - predictions_df["actual_goals"]
).round(2)
predictions_df["error_assists_Ensemble"] = (
    predictions_df["pred_assists_Ensemble"] - predictions_df["actual_assists"]
).round(2)

predictions_df = predictions_df.sort_values(
    ["season", "actual_goals", "actual_assists"],
    ascending=[True, False, False],
).reset_index(drop=True)

pred_2023 = predictions_df[predictions_df["season"].str.startswith("2023")]
pred_2024 = predictions_df[predictions_df["season"].str.startswith("2024")]

predictions_df.to_csv(out("predictions_test_all.csv"), index=False)
pred_2023.to_csv(out("predictions_2023_24.csv"), index=False)
pred_2024.to_csv(out("predictions_2024_25.csv"), index=False)

print("\nTop 20 golrugo 2024-25-ben:")
print(
    pred_2024.nlargest(20, "actual_goals")[
        [
            "player",
            "team",
            "position",
            "actual_goals",
            "pred_goals_Ensemble",
            "error_goals_Ensemble",
        ]
    ].to_string(index=False)
)

print("\nTop 20 golpassz ado 2024-25-ben:")
print(
    pred_2024.nlargest(20, "actual_assists")[
        [
            "player",
            "team",
            "position",
            "actual_assists",
            "pred_assists_Ensemble",
            "error_assists_Ensemble",
        ]
    ].to_string(index=False)
)


#  2025-26 pred

df_2024 = df[df["season_start"] == 2024].copy()
df_2023 = df[df["season_start"] == 2023].copy()


df_2024["age_next"] = df_2024["age"] + 1

future_features = pd.DataFrame()

# prev_ = 2024-25os ertekek
for col in feature_cols:
    future_features[f"prev_{col}"] = df_2024[col].values

# prev2_ = 2023-24 es ertekek
df_2023_lookup = df_2023.set_index("player_id")
for col in prev2_only:
    if col in df_2023_lookup.columns:
        future_features[f"prev2_{col}"] = (
            df_2024["player_id"].map(df_2023_lookup[col]).fillna(0).values
        )
    else:
        future_features[f"prev2_{col}"] = 0

future_features["has_prev2"] = (
    df_2024["player_id"].isin(df_2023["player_id"]).astype(int).values
)
future_features["age"] = df_2024["age_next"].values

# eletkor pozicio atlagok a kovetkezo szezonos eletkorra
df_2024_pos = df_2024["position_orig"].values
df_2024_age_next = df_2024["age_next"].values

for col_name, mapping, fallback in [
    ("age_pos_avg_goals", age_pos_goals, pos_goals_avg),
    ("age_pos_avg_assists", age_pos_assist, pos_assist_avg),
    ("age_pos_avg_xG", age_pos_xG, pos_xG_avg),
    ("age_pos_avg_xA", age_pos_xA, pos_xA_avg),
]:
    vals = []
    for a, p in zip(df_2024_age_next, df_2024_pos):
        vals.append(mapping.get((a, p), fallback.get(p, 0)))
    future_features[col_name] = vals

future_features = future_features[model_features].fillna(0)
future_features = (
    future_features.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float64)
)

future_preds_g = predict_all(models_g, future_features)
future_preds_g["Ensemble"] = sum(
    weights_g[name] * future_preds_g[name] for name in weights_g
)

future_preds_a = predict_all(models_a, future_features)
future_preds_a["Ensemble"] = sum(
    weights_a[name] * future_preds_a[name] for name in weights_a
)

future_df = pd.DataFrame(
    {
        "player_id": df_2024["player_id"].values,
        "player": df_2024["player"].values,
        "team": df_2024["team"].values,
        "league": df_2024["league"].values,
        "position": df_2024["position_orig"].values,
        "age_in_2025_26": df_2024["age_next"].values,
        "goals_2024_25": df_2024["goals"].values,
        "assists_2024_25": df_2024["assists"].values,
        "minutes_2024_25": df_2024["minutes"].values,
        "pred_goals_2025_26_LR": future_preds_g["Linear Regression"].round(2),
        "pred_goals_2025_26_RF": future_preds_g["Random Forest"].round(2),
        f"pred_goals_2025_26_{gbm_name_g}": future_preds_g[gbm_name_g].round(2),
        "pred_goals_2025_26_Ensemble": future_preds_g["Ensemble"].round(2),
        "pred_assists_2025_26_LR": future_preds_a["Linear Regression"].round(2),
        "pred_assists_2025_26_RF": future_preds_a["Random Forest"].round(2),
        f"pred_assists_2025_26_{gbm_name_a}": future_preds_a[gbm_name_a].round(2),
        "pred_assists_2025_26_Ensemble": future_preds_a["Ensemble"].round(2),
    }
)

future_df = future_df.sort_values(
    "pred_goals_2025_26_Ensemble", ascending=False
).reset_index(drop=True)
future_df.to_csv(out("predictions_2025_26_FUTURE.csv"), index=False)

print("\nTop 25 vart golrúgo 2025-26-ban:")
print(
    future_df.head(25)[
        [
            "player",
            "team",
            "position",
            "age_in_2025_26",
            "goals_2024_25",
            "pred_goals_2025_26_Ensemble",
        ]
    ].to_string(index=False)
)

print("\nTop 25 vart golpassz-ado 2025-26-ban:")
top25_a = future_df.sort_values("pred_assists_2025_26_Ensemble", ascending=False).head(
    25
)
print(
    top25_a[
        [
            "player",
            "team",
            "position",
            "age_in_2025_26",
            "assists_2024_25",
            "pred_assists_2025_26_Ensemble",
        ]
    ].to_string(index=False)
)


# Feature fontossag

# Beepitett RF fontossag
rf_g = models_g["Random Forest"][0]
rf_a = models_a["Random Forest"][0]

fi_g = (
    pd.DataFrame({"feature": model_features, "fontossag": rf_g.feature_importances_})
    .sort_values("fontossag", ascending=False)
    .head(15)
)
fi_a = (
    pd.DataFrame({"feature": model_features, "fontossag": rf_a.feature_importances_})
    .sort_values("fontossag", ascending=False)
    .head(15)
)

fi_g.to_csv(out("feature_importance_goals.csv"), index=False)
fi_a.to_csv(out("feature_importance_assists.csv"), index=False)

fi_g_long = fi_g.copy()
fi_g_long["target"] = "Golok"
fi_a_long = fi_a.copy()
fi_a_long["target"] = "Golpasszok"
fi_combined = pd.concat([fi_g_long, fi_a_long], ignore_index=True)
fi_combined = fi_combined[["target", "feature", "fontossag"]]
fi_combined.to_csv(out("feature_importance_combined.csv"), index=False)


# Permutation importance

if PERM_MAX_SAMPLES is not None and len(X_te) > PERM_MAX_SAMPLES:
    rng = np.random.RandomState(RANDOM_STATE)
    perm_idx = rng.choice(len(X_te), PERM_MAX_SAMPLES, replace=False)
    X_te_perm = X_te.iloc[perm_idx]
    yg_te_perm = yg_te.iloc[perm_idx]
    ya_te_perm = ya_te.iloc[perm_idx]
else:
    X_te_perm = X_te
    yg_te_perm = yg_te
    ya_te_perm = ya_te


def compute_perm_importance(model, X, y, fit_target, scaler=None, needs_scale=False):

    y_eval = np.log1p(y) if fit_target == "log" else y

    if needs_scale:
        X_scaled = scaler.transform(X)
        X_input_df = pd.DataFrame(
            X_scaled, columns=X.columns if hasattr(X, "columns") else None
        )
        result = permutation_importance(
            model,
            X_input_df,
            y_eval,
            n_repeats=PERM_N_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            scoring="r2",
        )
    else:
        result = permutation_importance(
            model,
            X,
            y_eval,
            n_repeats=PERM_N_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            scoring="r2",
        )

    return pd.DataFrame(
        {
            "feature": (
                X.columns
                if hasattr(X, "columns")
                else [f"f{i}" for i in range(X.shape[1])]
            ),
            "perm_importance_mean": result.importances_mean,
            "perm_importance_std": result.importances_std,
        }
    ).sort_values("perm_importance_mean", ascending=False)


rf_model_g, rf_scaler_g, rf_scale_g, rf_target_g = models_g["Random Forest"]
perm_rf_g = compute_perm_importance(
    rf_model_g, X_te_perm, yg_te_perm, rf_target_g, rf_scaler_g, rf_scale_g
)
perm_rf_g["model"] = "Random Forest"
perm_rf_g["target"] = "Golok"


xgb_model_g, xgb_scaler_g, xgb_scale_g, xgb_target_g = models_g[gbm_name_g]
perm_xgb_g = compute_perm_importance(
    xgb_model_g, X_te_perm, yg_te_perm, xgb_target_g, xgb_scaler_g, xgb_scale_g
)
perm_xgb_g["model"] = gbm_name_g
perm_xgb_g["target"] = "Golok"


rf_model_a, rf_scaler_a, rf_scale_a, rf_target_a = models_a["Random Forest"]
perm_rf_a = compute_perm_importance(
    rf_model_a, X_te_perm, ya_te_perm, rf_target_a, rf_scaler_a, rf_scale_a
)
perm_rf_a["model"] = "Random Forest"
perm_rf_a["target"] = "Golpasszok"


xgb_model_a, xgb_scaler_a, xgb_scale_a, xgb_target_a = models_a[gbm_name_a]
perm_xgb_a = compute_perm_importance(
    xgb_model_a, X_te_perm, ya_te_perm, xgb_target_a, xgb_scaler_a, xgb_scale_a
)
perm_xgb_a["model"] = gbm_name_a
perm_xgb_a["target"] = "Golpasszok"

perm_all = pd.concat([perm_rf_g, perm_xgb_g, perm_rf_a, perm_xgb_a], ignore_index=True)
perm_all["perm_importance_mean"] = perm_all["perm_importance_mean"].round(5)
perm_all["perm_importance_std"] = perm_all["perm_importance_std"].round(5)
perm_all = perm_all[
    ["target", "model", "feature", "perm_importance_mean", "perm_importance_std"]
]
perm_all.to_csv(out("permutation_importance.csv"), index=False)


# Vizualizaciok

# Fo abra R² es MAE osszehasonlitas + scatter plotok
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

model_names = [n for n in preds_g.keys() if n != "BASELINE (tavalyi)"]
x_pos = np.arange(len(model_names))

# R² osszehasonlitas
ax = axes[0, 0]
r2_g_list = [results_g[n]["R²"] for n in model_names]
r2_a_list = [results_a[n]["R²"] for n in model_names]
ax.bar(x_pos - 0.2, r2_g_list, 0.4, label="Golok", color="#e74c3c")
ax.bar(x_pos + 0.2, r2_a_list, 0.4, label="Golpasszok", color="#2ecc71")
ax.axhline(
    results_g["BASELINE (tavalyi)"]["R²"],
    color="red",
    linestyle="--",
    alpha=0.5,
    label="Gol baseline",
)
ax.axhline(
    results_a["BASELINE (tavalyi)"]["R²"],
    color="green",
    linestyle="--",
    alpha=0.5,
    label="Passz baseline",
)
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation=30)
ax.set_ylabel("R²")
ax.set_title("Modellek R² osszehasonlitasa", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)

# MAE osszehasonlitas
ax = axes[0, 1]
mae_g_list = [results_g[n]["MAE"] for n in model_names]
mae_a_list = [results_a[n]["MAE"] for n in model_names]
ax.bar(x_pos - 0.2, mae_g_list, 0.4, label="Golok", color="#e74c3c")
ax.bar(x_pos + 0.2, mae_a_list, 0.4, label="Golpasszok", color="#2ecc71")
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation=30)
ax.set_ylabel("MAE")
ax.set_title("Modellek MAE osszehasonlitasa", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)

# Scatter valos, predikcio
ax = axes[1, 0]
ax.scatter(yg_te, preds_g["Ensemble"], alpha=0.5, s=20, color="#e74c3c")
mx = max(yg_te.max(), preds_g["Ensemble"].max())
ax.plot([0, mx], [0, mx], "k--", lw=2)
ax.set_xlabel("Valos golok")
ax.set_ylabel("Elorejelzett golok")
ax.set_title(
    f"Golok: Ensemble (R²={results_g['Ensemble']['R²']:.3f}, "
    f"MAE={results_g['Ensemble']['MAE']:.3f})",
    fontweight="bold",
)
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.scatter(ya_te, preds_a["Ensemble"], alpha=0.5, s=20, color="#2ecc71")
mx = max(ya_te.max(), preds_a["Ensemble"].max())
ax.plot([0, mx], [0, mx], "k--", lw=2)
ax.set_xlabel("Valos golpasszok")
ax.set_ylabel("Elorejelzett golpasszok")
ax.set_title(
    f"Golpasszok: Ensemble (R²={results_a['Ensemble']['R²']:.3f}, "
    f"MAE={results_a['Ensemble']['MAE']:.3f})",
    fontweight="bold",
)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(out("ml_results_improved.png"), dpi=100, bbox_inches="tight")
plt.show()

# Permutation importance abra
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
for ax, df_imp, title, color in [
    (axes[0, 0], perm_rf_g.head(15), "Random Forest - Golok", "#e74c3c"),
    (axes[0, 1], perm_xgb_g.head(15), f"{gbm_name_g} - Golok", "#c0392b"),
    (axes[1, 0], perm_rf_a.head(15), "Random Forest - Golpasszok", "#2ecc71"),
    (axes[1, 1], perm_xgb_a.head(15), f"{gbm_name_a} - Golpasszok", "#27ae60"),
]:
    df_plot = df_imp.iloc[::-1]
    ax.barh(
        df_plot["feature"],
        df_plot["perm_importance_mean"],
        xerr=df_plot["perm_importance_std"],
        color=color,
        alpha=0.85,
        error_kw={"ecolor": "black", "lw": 0.8, "alpha": 0.5},
    )
    ax.set_title(f"Permutation Importance - {title}", fontweight="bold", fontsize=11)
    ax.set_xlabel("R² csokkenes kevereskor")
    ax.grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(out("permutation_importance.png"), dpi=100, bbox_inches="tight")
plt.show()




