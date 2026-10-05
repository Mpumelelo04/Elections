from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="eThekwini Election Analytics",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    history = pd.read_csv(ROOT / "data/historical_ward_votes.csv")
    turnout = pd.read_csv(ROOT / "data/historical_turnout.csv")
    metro = pd.read_csv(ROOT / "results/metro_forecast.csv", index_col=0)
    wards = pd.read_csv(ROOT / "results/ward_forecast.csv", index_col=0)
    shares = pd.read_csv(
        ROOT / "results/ward_forecast_shares.csv", index_col=0
    )
    scenarios = pd.read_csv(ROOT / "results/turnout_scenarios.csv")
    vote_metrics = pd.read_csv(
        ROOT / "results/vote_model_validation.csv"
    )
    turnout_metrics = pd.read_csv(
        ROOT / "results/turnout_model_validation.csv"
    )
    return (
        history, turnout, metro, wards, shares,
        scenarios, vote_metrics, turnout_metrics
    )

(
    history, turnout, metro, wards, shares,
    scenarios, vote_metrics, turnout_metrics
) = load_data()

st.title("eThekwini Election Analytics")
st.caption("Historical evidence: 2016 and 2021 | Conditional 2026 scenarios")
st.warning(
    "2026 outputs are conditional model estimates using historical "
    "party categories and 2021 geographic footprints."
)

st.sidebar.header("Display settings")
top_n = st.sidebar.slider("Number of party categories", 5, 20, 10)

tabs = st.tabs([
    "Historical results", "Metro estimates", "Selected wards",
    "Model evaluation", "Sources and limitations"
])

with tabs[0]:
    st.subheader("Observed historical results")
    year = st.selectbox("Election year", [2016, 2021], index=1)
    historical = history.loc[history["ElectionYear"].eq(year)]

    totals = historical.groupby("PartyName")["TotalValidVotes"].sum()
    historical_shares = totals / totals.sum() * 100
    displayed = historical_shares.nlargest(top_n)

    turnout_year = turnout.loc[turnout["ElectionYear"].eq(year)]
    voters = turnout_year["VoterTurnout"].sum()
    denominator = (
        turnout_year["RegisteredVoters"] + turnout_year["MEC7Votes"]
    ).sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Valid Ward votes", f"{totals.sum():,.0f}")
    c2.metric("Reported voter turnout", f"{voters:,.0f}")
    c3.metric("Official turnout rate", f"{voters / denominator:.1%}")

    st.bar_chart(displayed.rename("Ward vote share (%)"))
    st.dataframe(displayed.to_frame("Ward vote share (%)").round(2))
    st.caption(
        "Shares use all valid Ward votes as the denominator. "
        "INDEPENDENT combines candidates."
    )

with tabs[1]:
    st.subheader("Conditional metro estimates for 2026")
    central = scenarios.loc[
        scenarios["Scenario"].eq("Central estimate")
    ].iloc[0]

    named = metro.loc[metro.index != "INDEPENDENT"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Largest estimated named party", named.index[0])
    c2.metric("Estimated voters", f"{central['EstimatedVoters']:,.0f}")
    c3.metric(
        "Estimated turnout",
        f"{central['EstimatedTurnoutPercent']:.1f}%"
    )

    st.bar_chart(
        metro.head(top_n)["EstimatedSharePercent"]
        .rename("Estimated Ward vote share (%)")
    )
    st.dataframe(metro.round(1), use_container_width=True)
    st.subheader("Turnout sensitivity scenarios")
    st.dataframe(scenarios.round(1), hide_index=True)
    st.caption(
        "Sensitivity ranges are not confidence intervals. Electorate "
        "and MEC7 counts are held at 2021 levels. Party totals are "
        "valid Ward ballots, not voter counts or council seats."
    )
    st.info(
        "Vote shares alone cannot determine council seat allocation, "
        "coalition formation or which party will govern."
    )
    st.download_button(
        "Download metro estimates",
        metro.to_csv(),
        file_name="metro_forecast.csv",
        mime="text/csv"
    )

with tabs[2]:
    st.subheader("Three selected ward footprints")
    selected = ["Ward 59500001", "Ward 59500004", "Ward 59500005"]
    ward = st.selectbox("Select ward", selected)

    st.dataframe(wards.loc[[ward]].round(1), use_container_width=True)
    ward_shares = shares.loc[ward].nlargest(top_n) * 100
    st.bar_chart(ward_shares.rename("Estimated vote share (%)"))

    st.caption(
        "Wards 1, 4 and 5 are the first three qualifying wards in "
        "numerical order. Their voting-district ID sets match across "
        "2016 and 2021. This does not verify unchanged boundaries "
        "or establish alignment with 2026 wards."
    )

with tabs[3]:
    st.subheader("Held-out ward validation")
    st.write(
        "39 training wards and 14 validation wards were used for "
        "the 2016–2021 transition. Models were subsequently refitted "
        "on all 53 qualifying wards for scenario generation."
    )
    st.write("Vote-share models")
    st.dataframe(vote_metrics.round(2), hide_index=True)
    st.write("Turnout models")
    st.dataframe(turnout_metrics.round(2), hide_index=True)

    st.write(
        "Both leading-party methods identified 13 of 14 validation "
        "wards correctly (92.9%). Random Forest misclassified one "
        "ANC-leading ward as DA-leading. Actual validation leaders "
        "included only ANC and DA."
    )
    st.caption(
        "pp means percentage points. Validation measures transfer "
        "to held-out wards, not forecasting accuracy for 2026."
    )

with tabs[4]:
    st.subheader("Data sources")
    st.markdown(
        "[IEC municipal results downloads]"
        "(https://results.elections.org.za/home/Downloads/ME-Results)"
    )
    st.write(
        "2016 and 2021 eThekwini detailed election results and "
        "Percentage Voter Turnout reports, accessed 5 October 2026."
    )

    st.subheader("Important limitations")
    st.markdown("""
- Only one historical election transition is available.
- Turnout estimates assume the earlier relationship continues.
- New parties and changing candidates are not adequately represented.
- Metro estimates apply models beyond the 53 qualifying training wards.
- Matching district IDs do not prove unchanged geographic boundaries.
- 2026 estimates refer to 2021 footprints with a fixed electorate.
- INDEPENDENT is a pooled category, not one party or candidate.
- Sensitivity scenarios are not statistical confidence intervals.
- Ward votes cannot establish council seats or governing coalitions.
""")
    st.write(
        "The notebook contains data preparation, model evaluation "
        "and analytical explanations. The submitted AI prompt "
        "record documents assistance used during the assessment."
    )
