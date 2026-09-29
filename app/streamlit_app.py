"""FY27 Recompete Cliff: explore which expiring federal contracts are most likely to change hands."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA = Path(__file__).parent / "data"
NAVY, TEAL, BLUE, SLATE = "#1e3a5f", "#0f766e", "#2563eb", "#64748b"
REPO = "https://github.com/jasonpellerin-aisolutionist/federal-recompete-cliff"

st.set_page_config(
    page_title="FY27 Recompete Cliff", page_icon=":material/timeline:", layout="wide"
)


@st.cache_data
def load() -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(DATA / "watchlist_fy27.parquet")
    df["end_date"] = pd.to_datetime(df["end_date"])
    meta = json.loads((DATA / "meta.json").read_text())
    return df, meta


def money(v: float) -> str:
    return f"${v / 1e9:,.1f}B" if v >= 1e9 else f"${v / 1e6:,.1f}M"


df, meta = load()
m = meta["test_metrics"]

st.title("The FY27 Recompete Cliff")
st.markdown(
    f"**{len(df):,} federal contracts worth {money(df['obligated'].sum())}** reach their end date "
    "between October 1, 2026 and September 30, 2027. If the work is bought again as a new award, "
    "how likely is it to go to a different vendor? "
    f"Data as of **{meta['asof']}**, refreshed weekly. Construction projects are excluded."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Contracts expiring in FY27", f"{len(df):,}")
c2.metric("Est. vendor changes", f"{df['p_change'].sum():,.0f}",
          help="Sum of calibrated probabilities, assuming each requirement is recompeted as a new "
               "award. Label noise in the training data leans this number high; see How it was tested.")  # fmt: skip
c3.metric(
    "Top 5% dollars",
    money(df.loc[df["risk_band"] == "Top 5%", "obligated"].sum()),
    help="Obligated to date on the 5% of contracts the model ranks most likely to change hands.",
)
c4.metric("Held-out AUC", f"{m['auc']:.2f}",
          help="Tested on recompetes the model never saw. 0.5 is a coin flip, 1.0 is perfect.")  # fmt: skip

tab_list, tab_lookup, tab_agency, tab_method = st.tabs(
    ["Watchlist", "Look up a contract", "By agency", "How it was tested"]
)

with tab_list:
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
    agencies = f1.multiselect("Agency", sorted(df["agency"].dropna().unique()))
    bands = f2.multiselect(
        "Risk band", ["Top 5%", "Elevated", "Typical", "Low"], default=["Top 5%", "Elevated"]
    )
    query = f3.text_input("Search vendor, office, or description")
    min_m = f4.number_input("Min obligated ($M)", min_value=0, value=5, step=5)
    view = df[df["obligated"] >= min_m * 1e6]
    if agencies:
        view = view[view["agency"].isin(agencies)]
    if bands:
        view = view[view["risk_band"].isin(bands)]
    if query:
        text = (view["recipient_name"].fillna("") + " " + view["office"].fillna("") + " "
                + view["description"].fillna("")).str.lower()  # fmt: skip
        view = view[text.str.contains(query.lower(), regex=False)]
    st.caption(f"{len(view):,} contracts, {money(view['obligated'].sum())} obligated")
    st.dataframe(
        view[["rank", "piid", "agency", "recipient_name", "end_date", "obligated", "offers", "p_change",
              "risk_band", "reason_1", "reason_2", "heuristic_score", "contract_protests", "permalink"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", format="%d"),
            "piid": "Contract (PIID)",
            "agency": "Agency",
            "contract_protests": st.column_config.NumberColumn("Open GAO protests", format="%d",
                help="GAO protests on the same solicitation in Fed-Spend's live docket."),
            "recipient_name": "Incumbent",
            "end_date": st.column_config.DateColumn("Ends", format="MMM D, YYYY"),
            "obligated": st.column_config.NumberColumn("Obligated", format="dollar"),
            "offers": st.column_config.NumberColumn("Offers", format="%d"),
            "p_change": st.column_config.ProgressColumn("P(incumbent loses)", min_value=0, max_value=1, format="percent"),
            "risk_band": "Band",
            "reason_1": "Top driver",
            "reason_2": "Second driver",
            "heuristic_score": st.column_config.NumberColumn("Fed-Spend heuristic", format="%d"),
            "permalink": st.column_config.LinkColumn("USAspending", display_text="open"),
        },
    )  # fmt: skip

with tab_lookup:
    piid = (
        st.text_input("Contract number (PIID)", placeholder="e.g. " + df.iloc[0]["piid"])
        .strip()
        .upper()
    )
    if piid:
        hit = df[df["piid"].str.upper() == piid]
        if hit.empty:
            st.info(
                "That PIID is not on the FY27 list: it may end in another year, be under $5M, or be newer than the data."
            )
        else:
            r = hit.iloc[0]
            a, b = st.columns([1, 2])
            a.metric(
                "Chance the incumbent loses", f"{r['p_change']:.0%}", help=f"Band: {r['risk_band']}"
            )
            a.metric("Fed-Spend heuristic score", f"{r['heuristic_score']:.0f} / 100")
            b.markdown(
                f"**{r['recipient_name']}** holds **{r['piid']}** at {r['office']} ({r['agency']}).  \n"
                f"Ends **{r['end_date']:%B %d, %Y}** ({r['days_to_end']} days). "
                f"Obligated {money(r['obligated'])} of a {money(r['potential_value'] or r['obligated'])} ceiling. "
                f"Offers on the last award: {'unknown' if pd.isna(r['offers']) else int(r['offers'])}.  \n"
                f"NAICS {r['naics']} {r['naics_description'] or ''}"
            )
            drivers = [x for x in (r["reason_1"], r["reason_2"], r["reason_3"]) if x]
            if drivers:
                b.markdown("**What pushes the estimate up:** " + "; ".join(drivers) + ".")
            b.markdown(
                f"[Open on USAspending]({r['permalink']}) · [Search on Fed-Spend](https://fed-spend.com/search?q={r['piid']})"
            )

with tab_agency:
    g = (df.groupby("agency", dropna=True)
           .agg(contracts=("piid", "size"), obligated=("obligated", "sum"), expected_changes=("p_change", "sum"))
           .assign(change_rate=lambda x: x["expected_changes"] / x["contracts"])
           .sort_values("expected_changes", ascending=False).head(15).reset_index())  # fmt: skip
    fig = px.bar(g.iloc[::-1], x="expected_changes", y="agency", orientation="h",
                 color_discrete_sequence=[NAVY], hover_data={"contracts": True, "change_rate": ":.0%"},
                 labels={"expected_changes": "Expected vendor changes in FY27", "agency": ""})  # fmt: skip
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
    monthly = df.assign(month=df["end_date"].dt.to_period("M").dt.to_timestamp()).groupby("month").agg(
        contracts=("piid", "size"), expected_changes=("p_change", "sum")).reset_index()  # fmt: skip
    fig2 = px.bar(monthly, x="month", y="contracts", color_discrete_sequence=[TEAL],
                  labels={"month": "", "contracts": "Contracts reaching end date"})  # fmt: skip
    fig2.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)

with tab_method:
    h, lg, hc, au = (meta["heuristic_metrics"], meta["logistic_metrics"],
                     meta["high_confidence_metrics"], meta["audit"])  # fmt: skip
    st.markdown(
        f"""
**Question.** When a federal contract ends and the work is bought again as a new award, does the
incumbent keep it?

**Labels.** {meta["n_labeled"]:,} contracts of $5M+ that ended in FY2019 to FY2025 were matched to
their follow-on award: same office, same kind of work, a similar description, similar size and
timing. The matcher never looks at the vendor. {meta["change_rate"]:.0%} of the training set went to
a different vendor.

**Label audit.** A blind review of 100 random matched pairs found {au.get("same_requirement", 0)}
clearly the same requirement, {au.get("partial", 0)} plausibly related, and
{au.get("wrong_match", 0)} wrong matches. Wrong matches mostly show up as false "changed" labels,
so change rates here lean high.

**Test.** Trained on FY2019 to FY2023, calibrated on FY2024, scored once on FY2025.

| | AUC | Brier | Top-10% hit rate |
|---|---|---|---|
| Fed-Spend heuristic (as shipped) | {h["auc"]:.3f} | {h["brier"]:.3f} | {h["top_decile_precision"]:.0%} |
| Logistic regression (baseline) | {lg["auc"]:.3f} | {lg["brier"]:.3f} | {lg["top_decile_precision"]:.0%} |
| This model (LightGBM, calibrated) | {m["auc"]:.3f} | {m["brier"]:.3f} | {m["top_decile_precision"]:.0%} |
| This model, high-confidence labels only | {hc["auc"]:.3f} | {hc["brier"]:.3f} | {hc["top_decile_precision"]:.0%} |

Model card, label audit, and code: [{REPO.split("/")[-1]}]({REPO}).
"""
    )
    st.caption(
        "Estimates, not predictions of any specific agency decision. Public data only (USAspending, GAO)."
    )
