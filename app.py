"""Credit Risk Analytics dashboard. Run with:  streamlit run app.py"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))
import credit_analytics as ca  # noqa: E402

NAVY, BLUE, LIGHT, RED = "#1F4E79", "#2E6FB7", "#A9C8E8", "#C4312B"
st.set_page_config(page_title="Credit Risk Analytics", layout="wide")


def banner(title):
    """Very faint blue band holding the page title."""
    st.markdown(
        f"""<div style="background:#F4F8FD;border:1px solid #E4EEF9;border-radius:10px;
        padding:18px 24px;margin-bottom:18px;">
        <span style="font-size:1.9rem;font-weight:700;color:{NAVY};">{title}</span></div>""",
        unsafe_allow_html=True)


@st.cache_data
def load():
    read = lambda f, **k: pd.read_csv(DATA / f, **k)
    return {
        "scored": read("scored_holdout.csv.gz"),
        "el": read("expected_loss_by_tier.csv"),
        "aff": read("affordability.csv"),
        "coefs": read("pd_model_coefficients.csv"),
        "vintage": read("vintage_curves.csv", index_col=0),
        "rr_bal": read("roll_rates_balance.csv", index_col=0),
        "rr_cnt": read("roll_rates_count.csv", index_col=0),
        "model": json.loads((DATA / "model_summary.json").read_text()),
        "book": json.loads((DATA / "portfolio_summary.json").read_text()),
    }


d = load()
m, book = d["model"], d["book"]
pct = lambda x, n=1: f"{x:.{n}%}"

st.sidebar.title("Credit Risk Analytics")
page = st.sidebar.radio("Section", ["Overview", "Cut-off simulator", "Expected loss by tier",
                                     "Affordability", "Vintage curves", "Roll rates", "Model"])
st.sidebar.caption("Data: Home Credit Default Risk (Kaggle). Amounts in the dataset's own "
                   "currency units. Default assumptions: LGD 60%, margin 15%.")
st.sidebar.markdown("[lamide-akanni](https://github.com/lamide-akanni)")

# ------------------------------------------------------------------ Overview
if page == "Overview":
    banner("Credit Risk Analytics")
    st.write("Who to approve, where the risk sits, and how the book performs after lending.")
    c = st.columns(4)
    c[0].metric("Applications", f"{m['applications']:,}")
    c[1].metric("Default rate", pct(m["default_rate"], 2))
    c[2].metric("Model Gini", f"{m['gini']:.2f}", help=f"AUC {m['auc']:.3f}, KS {m['ks']:.3f}")
    c[3].metric("Best cut-off", f"{m['best_cutoff']} pts",
                help=f"Approves {pct(m['best_approval_rate'])}, bad rate {pct(m['best_bad_rate'], 2)}")
    el = d["el"]
    c = st.columns(4)
    c[0].metric("Expected loss rate", pct(el["expected_loss"].sum() / el["ead"].sum(), 2))
    c[1].metric("Tier E share of loss", pct(el["share_of_loss"].iloc[-1], 0),
                help=f"vs {pct(el['share_of_book'].iloc[-1], 0)} of the book")
    c[2].metric("Card annualised loss", pct(book["annualised_loss"], 2))
    c[3].metric("Card accounts tracked", f"{book['accounts']:,}")
    st.subheader("What each section answers")
    st.table(pd.DataFrame({
        "Section": ["Cut-off simulator", "Expected loss by tier", "Affordability", "Vintage curves", "Roll rates"],
        "Question": ["Who do we approve?", "Where does the loss come from?", "Can customers afford it?",
                     "Is new lending performing as expected?", "Where do late customers stop recovering?"],
    }).set_index("Section"))

# ------------------------------------------------------------------ Cut-off simulator
elif page == "Cut-off simulator":
    banner("Score cut-off simulator")
    st.write("Move the sliders to see the trade-off between approving more and losing more.")
    sc = d["scored"]
    c1, c2, c3 = st.columns(3)
    lgd = c1.slider("LGD (share lost on a default)", 0.2, 1.0, float(m["lgd"]), 0.05)
    margin = c2.slider("Margin on good loans", 0.02, 0.40, float(m["margin_rate"]), 0.01)
    table = ca.cutoff_table(sc, margin_rate=margin, lgd=lgd, n_points=30)
    best = table.loc[table["best"]].iloc[0]
    lo, hi = int(table["cutoff_score"].min()), int(table["cutoff_score"].max())
    cutoff = c3.slider("Cut-off score", lo, hi, int(best["cutoff_score"]))

    approved = sc[sc["score"] >= cutoff]
    good = approved.loc[approved["TARGET"] == 0, "AMT_CREDIT"].sum()
    bad = approved.loc[approved["TARGET"] == 1, "AMT_CREDIT"].sum()
    k = st.columns(4)
    k[0].metric("Approval rate", pct(len(approved) / len(sc)))
    k[1].metric("Bad rate of approved", pct(approved["TARGET"].mean(), 2),
                delta=f"{(approved['TARGET'].mean() - sc['TARGET'].mean()) * 100:+.2f} pts vs approve all",
                delta_color="inverse")
    k[2].metric("Expected profit", f"{(margin * good - lgd * bad) / 1e6:,.0f}m")
    k[3].metric("Best cut-off", f"{best['cutoff_score']:.0f}",
                help=f"Break-even PD = margin / (margin + LGD) = {margin / (margin + lgd):.1%}")

    fig = go.Figure()
    fig.add_scatter(x=table["cutoff_score"], y=table["approval_rate"], name="Approval rate", line_color=NAVY)
    fig.add_scatter(x=table["cutoff_score"], y=table["bad_rate"], name="Bad rate", yaxis="y2",
                    line=dict(color=RED, dash="dash"))
    fig.add_vline(x=cutoff, line_dash="dot", line_color=BLUE)
    fig.update_layout(height=380, xaxis_title="Cut-off score", legend=dict(orientation="h", y=1.12),
                      yaxis=dict(title="Approval rate", tickformat=".0%"),
                      yaxis2=dict(title="Bad rate", tickformat=".1%", overlaying="y", side="right"))
    st.plotly_chart(fig, width="stretch")
    fig2 = px.line(table, x="cutoff_score", y=table["expected_profit"] / 1e6, markers=True,
                   labels={"cutoff_score": "Cut-off score", "y": "Expected profit (m)"})
    fig2.update_traces(line_color=BLUE)
    fig2.add_vline(x=cutoff, line_dash="dot", line_color=BLUE)
    fig2.update_layout(height=280)
    st.plotly_chart(fig2, width="stretch")

# ------------------------------------------------------------------ Expected loss
elif page == "Expected loss by tier":
    banner("Expected loss by risk tier")
    st.latex(r"\text{Expected loss} = PD \times LGD \times EAD")
    el = d["el"]
    fig = go.Figure([go.Bar(x=el["risk_tier"], y=el["share_of_book"], name="Share of book", marker_color=LIGHT),
                     go.Bar(x=el["risk_tier"], y=el["share_of_loss"], name="Share of expected loss", marker_color=RED)])
    fig.update_layout(barmode="group", height=380, yaxis_tickformat=".0%",
                      xaxis_title="Risk tier (A = safest)", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
    show = el.copy()
    for col in ["avg_pd", "share_of_book", "share_of_loss"]:
        show[col] = show[col].map(lambda v: f"{v:.1%}")
    for col in ["ead", "expected_loss"]:
        show[col] = show[col].map(lambda v: f"{v:,.0f}")
    st.dataframe(show, hide_index=True, width="stretch")

# ------------------------------------------------------------------ Affordability
elif page == "Affordability":
    banner("Affordability: repayment-to-income")
    aff = d["aff"]
    fig = px.bar(aff, x="repayment_to_income", y="bad_rate", text=aff["bad_rate"].map(lambda v: f"{v:.1%}"),
                 labels={"repayment_to_income": "Repayment ÷ income", "bad_rate": "Bad rate"})
    fig.update_traces(marker_color=BLUE)
    fig.add_hline(y=m["default_rate"], line_dash="dash", line_color="grey", annotation_text="Average")
    fig.update_layout(height=400, yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    st.write("The bad rate rises only gently with repayment burden and dips above 30%. These are "
             "approved loans, so the most stretched applicants were likely already declined "
             "(selection bias). A hard affordability cap would need testing before adoption.")

# ------------------------------------------------------------------ Vintage curves
elif page == "Vintage curves":
    banner("Vintage curves")
    st.write(f"Share of each quarter's card accounts that have ever been "
             f"{book.get('vintage_bad_definition', 'ever 30+ DPD').replace('ever ', '')}, by months on book. "
             "Dates are mapped from relative months, so labels are illustrative.")
    vc = d["vintage"]
    options = list(vc.columns)
    chosen = st.multiselect("Vintages", options, default=[v for v in options if "2015Q3" <= v <= "2017Q4"] or options[-8:])
    long = vc[chosen].reset_index(names="mob").melt("mob", var_name="Vintage", value_name="rate").dropna()
    fig = px.line(long, x="mob", y="rate", color="Vintage", markers=True,
                  labels={"mob": "Months on book", "rate": "Cumulative share late"},
                  color_discrete_sequence=px.colors.sample_colorscale("Blues", [0.35 + 0.65 * i / max(len(chosen) - 1, 1) for i in range(len(chosen))]))
    fig.update_layout(height=450, yaxis_tickformat=".1%")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ Roll rates
elif page == "Roll rates":
    banner("Roll rates")
    basis = st.radio("Weighted by", ["Balance", "Account count"], horizontal=True)
    rr = d["rr_bal"] if basis == "Balance" else d["rr_cnt"]
    fig = px.imshow(rr, text_auto=".0%", color_continuous_scale="Blues", zmin=0, zmax=1,
                    labels=dict(x="Next month", y="This month", color="Share"))
    fig.update_layout(height=450)
    st.plotly_chart(fig, width="stretch")
    cure = rr.iloc[:, 0]
    st.write("**Cure rate** (back to Current next month): " +
             " · ".join(f"{b} {v:.0%}" for b, v in cure.items() if b != "Current"))
    st.caption("Light-touch contact early; focus collections effort on 31–90 days, "
               "before accounts reach 90+ where few recover.")

# ------------------------------------------------------------------ Model
elif page == "Model":
    banner("Default (PD) model")
    c = st.columns(3)
    c[0].metric("AUC", f"{m['auc']:.3f}")
    c[1].metric("Gini", f"{m['gini']:.3f}")
    c[2].metric("KS", f"{m['ks']:.3f}")
    st.write("Logistic regression on application data, tested on a 25% hold-out set. "
             "Gender and family status were excluded on purpose.")
    co = d["coefs"].head(12).iloc[::-1]
    fig = px.bar(co, x="coefficient", y="feature", orientation="h", color="direction",
                 color_discrete_map={"raises risk": RED, "lowers risk": BLUE},
                 labels={"coefficient": "Standardised coefficient", "feature": "", "direction": ""})
    fig.update_layout(height=450)
    st.plotly_chart(fig, width="stretch")
    st.caption("Category effects (contract, income and education type) compare only within their own group.")