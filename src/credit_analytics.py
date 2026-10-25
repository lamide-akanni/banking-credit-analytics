"""Credit analytics functions used by the notebooks and the dashboard."""

import numpy as np
import pandas as pd

def prepare_applications(app: pd.DataFrame) -> pd.DataFrame:
    """Clean application_train.csv and add affordability features."""
    df = app.copy()
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    # 365243 is Home Credit's code for "not employed" (mostly pensioners)
    df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"].replace(365243, np.nan) / 365.25
    df["CREDIT_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_TO_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TO_ANNUITY"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    return df.replace([np.inf, -np.inf], np.nan)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Explainable feature set. Gender and family status left out on purpose (fair lending).
NUMERIC_FEATURES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AGE_YEARS", "YEARS_EMPLOYED",
                    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
                    "CREDIT_TO_INCOME", "ANNUITY_TO_INCOME", "CREDIT_TO_ANNUITY"]
CATEGORICAL_FEATURES = ["NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE"]


def train_pd_model(df, test_size=0.25, seed=42):
    """Logistic regression PD model. Returns (model, scored hold-out set, metrics)."""
    X, y = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df["TARGET"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)

    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01))]),
         CATEGORICAL_FEATURES),
    ])
    model = Pipeline([("prep", prep), ("logit", LogisticRegression(max_iter=1000))])
    model.fit(X_train, y_train)

    test = df.loc[X_test.index].copy()
    test["pd"] = model.predict_proba(X_test)[:, 1]
    return model, test, model_metrics(y_test, test["pd"])


def model_metrics(y_true, pd_hat):
    """AUC, Gini and KS: the three numbers credit teams quote for a scorecard."""
    auc = roc_auc_score(y_true, pd_hat)
    fpr, tpr, _ = roc_curve(y_true, pd_hat)
    return {"auc": auc, "gini": 2 * auc - 1, "ks": float(np.max(tpr - fpr))}


def coefficient_table(model):
    """Coefficients on standardised features: bigger size = stronger effect."""
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = model.named_steps["logit"].coef_[0]
    out = pd.DataFrame({"feature": names, "coefficient": coefs})
    out["feature"] = out["feature"].str.replace(r"^(num|cat)__", "", regex=True)
    out["direction"] = np.where(out["coefficient"] > 0, "raises risk", "lowers risk")
    return out.reindex(out["coefficient"].abs().sort_values(ascending=False).index)

def pd_to_score(pd_hat, base_score=600, base_odds=50, pdo=20):
    """Scorecard points: 600 = 50:1 good:bad odds, +20 points doubles the odds."""
    p = np.clip(pd_hat, 1e-6, 1 - 1e-6)
    factor = pdo / np.log(2)
    return np.round(base_score + factor * (np.log((1 - p) / p) - np.log(base_odds)))


def assign_risk_tier(pd_hat, n_tiers=5):
    """Five equal-sized tiers by PD: A = lowest risk, E = highest."""
    labels = [chr(ord("A") + i) for i in range(n_tiers)]
    return pd.qcut(pd_hat, n_tiers, labels=labels)

def cutoff_table(scored, score_col="score", bad_col="TARGET", exposure_col="AMT_CREDIT",
                 n_points=20, margin_rate=0.15, lgd=0.60):
    """Approval rate, bad rate and expected profit at each candidate cut-off.
    profit = margin_rate x good exposure - lgd x bad exposure (both are assumptions)."""
    s = scored[score_col]
    cutoffs = np.unique(np.quantile(s, np.linspace(0.01, 0.95, n_points)).round())
    rows = []
    for c in cutoffs:
        approved = scored[s >= c]
        good = approved.loc[approved[bad_col] == 0, exposure_col].sum()
        bad = approved.loc[approved[bad_col] == 1, exposure_col].sum()
        rows.append({"cutoff_score": int(c),
                     "approval_rate": len(approved) / len(scored),
                     "bad_rate": approved[bad_col].mean(),
                     "expected_profit": margin_rate * good - lgd * bad})
    out = pd.DataFrame(rows)
    out["best"] = out["expected_profit"] == out["expected_profit"].max()
    return out 

def expected_loss_by_tier(scored, lgd=0.60):
    """Expected loss = PD x LGD x EAD, totalled by risk tier."""
    df = scored.assign(expected_loss=scored["pd"] * lgd * scored["AMT_CREDIT"])
    t = df.groupby("risk_tier", observed=True).agg(
        avg_pd=("pd", "mean"), ead=("AMT_CREDIT", "sum"), expected_loss=("expected_loss", "sum"))
    t["share_of_book"] = t["ead"] / t["ead"].sum()
    t["share_of_loss"] = t["expected_loss"] / t["expected_loss"].sum()
    return t.reset_index()

def affordability_table(df):
    """Bad rate by repayment-to-income band."""
    bins = [0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, np.inf]
    labels = ["<10%", "10-15%", "15-20%", "20-25%", "25-30%", "30-40%", "40%+"]
    band = pd.cut(df["ANNUITY_TO_INCOME"], bins=bins, labels=labels)
    t = df.groupby(band, observed=True)["TARGET"].agg(loans="size", bad_rate="mean")
    t["share_of_loans"] = t["loans"] / t["loans"].sum()
    return t.reset_index().rename(columns={"ANNUITY_TO_INCOME": "repayment_to_income"})

def prepare_card_panel(cc, anchor="2019-01-01"):
    """Turn credit_card_balance.csv into a monthly panel with real-looking dates.
    MONTHS_BALANCE is relative, so it is mapped onto a calendar from an anchor date
    (labels are illustrative). Accounts already open at the start of the data
    window are dropped because their true start month is unknown."""
    df = cc[["SK_ID_PREV", "MONTHS_BALANCE", "SK_DPD", "AMT_BALANCE"]].copy()
    first = df.groupby("SK_ID_PREV")["MONTHS_BALANCE"].transform("min")
    keep = first > df["MONTHS_BALANCE"].min()
    df, first = df[keep], first[keep]

    base = pd.Period(anchor, freq="M").ordinal
    to_date = lambda m: pd.PeriodIndex.from_ordinals(base + np.asarray(m, dtype="int64"), freq="M").to_timestamp()

    return pd.DataFrame({
        "loan_id": df["SK_ID_PREV"].values,
        "origination_date": to_date(first),
        "snapshot_date": to_date(df["MONTHS_BALANCE"]),
        "days_past_due": df["SK_DPD"].clip(lower=0).values,
        "balance": df["AMT_BALANCE"].clip(lower=0).values,
    })

DPD_BUCKETS = [-1, 0, 30, 60, 90, np.inf]
BUCKET_LABELS = ["Current", "1-30", "31-60", "61-90", "90+"]


def add_bucket_and_mob(panel, vintage_freq="Q"):
    """Add delinquency bucket, months on book (MOB) and vintage (origination quarter)."""
    df = panel.copy()
    df["bucket"] = pd.cut(df["days_past_due"], bins=DPD_BUCKETS, labels=BUCKET_LABELS)
    df["mob"] = ((df["snapshot_date"].dt.year - df["origination_date"].dt.year) * 12
                 + (df["snapshot_date"].dt.month - df["origination_date"].dt.month))
    df["vintage"] = df["origination_date"].dt.to_period(vintage_freq).astype(str)
    return df


def vintage_curves(panel, bad_dpd=90, max_mob=24, min_accounts=200):
    """Cumulative share of a vintage's accounts that have EVER been bad_dpd+ days late.
    Denominator = all accounts opened in the vintage (closed accounts stay in),
    so curves can only rise. Each vintage is shown only up to the month every
    one of its accounts could have been observed."""
    df = add_bucket_and_mob(panel)
    data_end = df["snapshot_date"].max()
    acc = df.groupby("loan_id").agg(vintage=("vintage", "first"), orig=("origination_date", "first"))
    acc["first_bad_mob"] = df[df["days_past_due"] >= bad_dpd].groupby("loan_id")["mob"].min()
    acc["observable"] = ((data_end.year - acc["orig"].dt.year) * 12
                         + (data_end.month - acc["orig"].dt.month))
    curves = {}
    for v, g in acc.groupby("vintage"):
        if len(g) < min_accounts:
            continue
        horizon = int(min(g["observable"].min(), max_mob))
        curves[v] = pd.Series([(g["first_bad_mob"] <= m).mean() for m in range(horizon + 1)])
    out = pd.DataFrame(curves).reindex(range(max_mob + 1))
    return out.loc[:, out.notna().sum() >= 3]

def roll_rates(panel, by="balance"):
    """Month-to-month transition matrix between delinquency buckets.
    by='balance' weights by amount owed; by='count' counts accounts."""
    df = add_bucket_and_mob(panel).sort_values(["loan_id", "snapshot_date"])
    df["next_bucket"] = df.groupby("loan_id")["bucket"].shift(-1)
    df = df.dropna(subset=["next_bucket"])
    if by == "balance":
        m = pd.crosstab(df["bucket"], df["next_bucket"], values=df["balance"],
                        aggfunc="sum", normalize="index")
    else:
        m = pd.crosstab(df["bucket"], df["next_bucket"], normalize="index")
    return m.reindex(index=BUCKET_LABELS, columns=BUCKET_LABELS).fillna(0)

def annualised_loss(panel, chargeoff_dpd=90):
    """Balance on accounts reaching chargeoff_dpd+ for the first time,
    divided by average monthly balance outstanding, scaled to a year."""
    df = panel.sort_values(["loan_id", "snapshot_date"])
    first_co = df[df["days_past_due"] >= chargeoff_dpd].groupby("loan_id").head(1)
    months = df["snapshot_date"].dt.to_period("M")
    avg_balance = df.groupby(months)["balance"].sum().mean()
    return float(first_co["balance"].sum() / avg_balance * 12 / months.nunique())