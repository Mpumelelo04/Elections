from pathlib import Path
import pandas as pd
import io
import numpy as np
from IPython.display import display

project_dir = Path.cwd()
if not (project_dir / "data").exists():
    project_dir = project_dir / "election_analytics"
df_2021 = pd.read_csv(project_dir / "data/elections_2021_ward.csv")
df_2021["ElectionYear"] = 2021


raw_2016 = pd.read_csv(project_dir / "data/elections_2016_all_ballots.csv")
df_2016 = raw_2016.loc[raw_2016["BallotType"].str.strip().eq("Ward")].copy()
df_2016["ElectionYear"] = 2016


# Combine the two election years.
data = pd.concat([df_2016, df_2021], ignore_index=True)

# Remove accidental spaces from text fields.
for column in ["Ward", "PartyName", "BallotType"]:
    data[column] = data[column].astype("string").str.strip()

# Check that counts are numeric and non-negative.
count_columns = ["RegisteredVoters", "SpoiltVotes", "TotalValidVotes"]

for column in count_columns:
    data[column] = pd.to_numeric(data[column], errors="raise")
    assert data[column].notna().all(), f"Missing values in {column}"
    assert data[column].ge(0).all(), f"Negative values in {column}"

# Each party should appear once per district and election.
keys = ["ElectionYear", "VotingDistrict", "BallotType", "PartyName"]
print("Duplicate party records:", data.duplicated(keys).sum())
print("Missing values:", data.isna().sum().sum())

# Registered voters and spoilt votes repeat on party rows.
# Check they are consistent within each district.
district_keys = ["ElectionYear", "Ward", "VotingDistrict"]
consistency = data.groupby(district_keys)[
    ["RegisteredVoters", "SpoiltVotes"]
].nunique()

print("Districts with inconsistent counts:",
      consistency.gt(1).any(axis=1).sum())

# Sum party votes, but count registration and spoilt votes only once.
districts = data.groupby(district_keys, as_index=False).agg(
    ValidVotes=("TotalValidVotes", "sum"),
    RegisteredVoters=("RegisteredVoters", "first"),
    SpoiltVotes=("SpoiltVotes", "first")
)

display(districts.head())

import matplotlib.pyplot as plt

# Historical metro totals for Ward ballots only.
metro_summary = districts.groupby("ElectionYear").agg(
    RegisteredVoters=("RegisteredVoters", "sum"),
    ValidWardVotes=("ValidVotes", "sum"),
    SpoiltWardVotes=("SpoiltVotes", "sum")
)
display(metro_summary)

# Calculate each party's share of valid Ward votes.
party_totals = data.groupby(
    ["ElectionYear", "PartyName"]
)["TotalValidVotes"].sum().unstack("ElectionYear", fill_value=0)

party_shares = party_totals.div(
    party_totals.sum(axis=0), axis=1
) * 100

# Display the eight largest parties by 2021 Ward votes.
top_parties = party_totals[2021].nlargest(8).index

display(party_shares.loc[top_parties].round(2))

party_shares.loc[top_parties].plot(
    kind="barh", figsize=(10, 6)
)
plt.title("eThekwini: historical Ward-ballot vote shares")
plt.xlabel("Share of all valid Ward votes (%)")
plt.ylabel("Party")
plt.legend(title="Election year")
plt.tight_layout()
plt.show()

# Compare the voting-district IDs assigned to each ward.
ward_sets = {
    year: districts.loc[districts["ElectionYear"].eq(year)]
        .groupby("Ward")["VotingDistrict"].apply(set)
    for year in [2016, 2021]
}

common_wards = sorted(
    set(ward_sets[2016].index) & set(ward_sets[2021].index)
)

comparison = []
for ward in common_wards:
    old = ward_sets[2016][ward]
    new = ward_sets[2021][ward]
    comparison.append({
        "Ward": ward,
        "Districts2016": len(old),
        "Districts2021": len(new),
        "SameDistrictIDs": old == new,
        "DistrictOverlap": len(old & new) / len(old | new)
    })

ward_comparison = pd.DataFrame(comparison)

print("Wards appearing in both years:", len(ward_comparison))
print("Wards with identical district ID sets:",
      ward_comparison["SameDistrictIDs"].sum())

display(
    ward_comparison.sort_values(
        ["SameDistrictIDs", "DistrictOverlap"],
        ascending=False
    ).head(10)
)

# Historical party totals within each ward.
ward_votes = data.groupby(
    ["ElectionYear", "Ward", "PartyName"], as_index=False
)["TotalValidVotes"].sum()

# Exclude the combined INDEPENDENT category when naming a leading party.
party_only = ward_votes.loc[
    ~ward_votes["PartyName"].str.upper().eq("INDEPENDENT")
]

leaders_2021 = (
    party_only.loc[party_only["ElectionYear"].eq(2021)]
    .sort_values("TotalValidVotes", ascending=False)
    .drop_duplicates("Ward")
    .rename(columns={"PartyName": "LeadingParty2021"})
)

candidate_wards = ward_comparison.loc[
    ward_comparison["SameDistrictIDs"]
].merge(
    leaders_2021[["Ward", "LeadingParty2021", "TotalValidVotes"]],
    on="Ward"
)

display(candidate_wards[
    ["Ward", "Districts2021", "LeadingParty2021", "TotalValidVotes"]
].head(20))

selected_wards = (
    candidate_wards.sort_values("Ward")["Ward"].head(3).tolist()
)

print("Selected wards:", selected_wards)

# Keep all 53 qualifying wards for later evaluation.
eligible_wards = candidate_wards["Ward"].tolist()

# One row per ward, one column per party/category.
votes_2016 = ward_votes.loc[
    ward_votes["ElectionYear"].eq(2016)
].pivot(index="Ward", columns="PartyName", values="TotalValidVotes")

votes_2021 = ward_votes.loc[
    ward_votes["ElectionYear"].eq(2021)
].pivot(index="Ward", columns="PartyName", values="TotalValidVotes")

# Align categories across years.
# Missing categories get zero recorded votes, not zero forecast uncertainty.
all_categories = sorted(
    set(votes_2016.columns) | set(votes_2021.columns)
)

votes_2016 = votes_2016.reindex(
    index=eligible_wards, columns=all_categories
).fillna(0)

votes_2021 = votes_2021.reindex(
    index=eligible_wards, columns=all_categories
).fillna(0)

shares_2016 = votes_2016.div(votes_2016.sum(axis=1), axis=0)
shares_2021 = votes_2021.div(votes_2021.sum(axis=1), axis=0)

# Baseline prediction made from the earlier election only.
baseline_prediction = shares_2016.copy()

# Error measured in percentage points.
absolute_errors = (
    baseline_prediction - shares_2021
).abs() * 100

print("Wards evaluated:", len(absolute_errors))
print("Mean absolute error across all categories:",
      round(absolute_errors.to_numpy().mean(), 2),
      "percentage points")

print("\nErrors for the four major named parties:")
display(
    absolute_errors[
        ["AFRICAN NATIONAL CONGRESS",
         "DEMOCRATIC ALLIANCE",
         "ECONOMIC FREEDOM FIGHTERS",
         "INKATHA FREEDOM PARTY"]
    ].mean().rename("MAE_percentage_points").to_frame().round(2)
)

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

major_parties = [
    "AFRICAN NATIONAL CONGRESS",
    "DEMOCRATIC ALLIANCE",
    "ECONOMIC FREEDOM FIGHTERS",
    "INKATHA FREEDOM PARTY"
]

train_wards, validation_wards = train_test_split(
    eligible_wards, test_size=0.25, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=3,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

model.fit(
    shares_2016.loc[train_wards],
    shares_2021.loc[train_wards]
)

rf_prediction = pd.DataFrame(
    model.predict(shares_2016.loc[validation_wards]),
    index=validation_wards,
    columns=all_categories
)

# Ensure non-negative shares that sum to one.
rf_prediction = rf_prediction.clip(lower=0)
rf_prediction = rf_prediction.div(rf_prediction.sum(axis=1), axis=0)

actual = shares_2021.loc[validation_wards]
baseline_valid = shares_2016.loc[validation_wards]

results = []
for name, prediction in [
    ("Persistence baseline", baseline_valid),
    ("Random Forest", rf_prediction)
]:
    errors = (prediction - actual) * 100
    results.append({
        "Model": name,
        "All-category MAE (pp)": errors.abs().to_numpy().mean(),
        "Major-party MAE (pp)": errors[major_parties].abs().to_numpy().mean(),
        "All-category RMSE (pp)": np.sqrt((errors ** 2).to_numpy().mean())
    })

model_comparison = pd.DataFrame(results)
print("Training wards:", len(train_wards))
print("Validation wards:", len(validation_wards))
display(model_comparison.round(2))

from sklearn.metrics import (
    accuracy_score, classification_report, ConfusionMatrixDisplay
)

def leading_party(shares):
    named = shares.drop(columns=["INDEPENDENT"], errors="ignore")
    leaders = named.idxmax(axis=1)
    largest = named.max(axis=1)

    # A pooled independent category cannot identify one candidate.
    independent = (
        shares["INDEPENDENT"] if "INDEPENDENT" in shares
        else pd.Series(0.0, index=shares.index)
    )

    ties = named.eq(largest, axis=0).sum(axis=1).gt(1)
    uncertain = independent.ge(largest) | ties

    return leaders.mask(uncertain, "UNCERTAIN")

actual_leaders = leading_party(actual)
baseline_leaders = leading_party(baseline_valid)
rf_leaders = leading_party(rf_prediction)

# Evaluate only cases where the actual leading party is identifiable.
identifiable = actual_leaders.ne("UNCERTAIN")
print("Identifiable validation wards:", identifiable.sum(),
      "out of", len(actual_leaders))

for name, predictions in [
    ("Baseline", baseline_leaders),
    ("Random Forest", rf_leaders)
]:
    print(name, "accuracy:",
          round(accuracy_score(
              actual_leaders[identifiable],
              predictions[identifiable]
          ), 3))

print("\nRandom Forest classification report:")
print(classification_report(
    actual_leaders[identifiable],
    rf_leaders[identifiable],
    zero_division=0
))

ConfusionMatrixDisplay.from_predictions(
    actual_leaders[identifiable],
    rf_leaders[identifiable],
    xticks_rotation=90
)
plt.title("Leading-party validation: Random Forest")
plt.tight_layout()
plt.show()


from pathlib import Path
import pandas as pd
import io

turnout_uploads = {
    name: (project_dir / "data" / name).read_bytes()
    for name in ["turnout_2016.xls", "turnout_2021.xls"]
}

turnout_tables = []

for filename, content in turnout_uploads.items():
    raw = pd.read_excel(
        io.BytesIO(content), header=None, engine="xlrd"
    )

    report_text = " ".join(
        raw.fillna("").astype(str).to_numpy().ravel()
    )

    if "LOCAL GOVERNMENT ELECTION 2021" in report_text:
        year = 2021
        positions = [1, 10, 12, 13, 16]
    elif "LOCAL GOVERNMENT ELECTION 2016" in report_text:
        year = 2016
        positions = [1, 6, 8, 9, 12]
    else:
        raise ValueError(f"Unrecognised election report: {filename}")

    table = raw.iloc[:, positions].copy()
    table.columns = [
        "Ward", "RegisteredVoters", "MEC7Votes",
        "VoterTurnout", "TurnoutRate"
    ]

    # Keep ward records, excluding headings and the Total row.
    ward_ids = table["Ward"].astype(str).str.strip()
    table = table.loc[
        ward_ids.str.fullmatch(r"\d{8}", na=False)
    ].copy()

    table["Ward"] = "Ward " + ward_ids.loc[table.index]

    for column in table.columns[1:]:
        table[column] = pd.to_numeric(
            table[column], errors="raise"
        )

    table["ElectionYear"] = year
    turnout_tables.append(table)

turnout_data = pd.concat(turnout_tables, ignore_index=True)

assert set(turnout_data["ElectionYear"]) == {2016, 2021}
assert not turnout_data.duplicated(
    ["ElectionYear", "Ward"]
).any()

print("Wards loaded per year:")
print(turnout_data.groupby("ElectionYear").size())

display(turnout_data.head())

# Confirm the reported rate matches the IEC denominator.
calculated_rate = turnout_data["VoterTurnout"] / (
    turnout_data["RegisteredVoters"] + turnout_data["MEC7Votes"]
)
assert np.allclose(calculated_rate, turnout_data["TurnoutRate"])

turnout_2016 = (
    turnout_data.loc[turnout_data["ElectionYear"].eq(2016)]
    .set_index("Ward")
)
turnout_2021 = (
    turnout_data.loc[turnout_data["ElectionYear"].eq(2021)]
    .set_index("Ward")
)

# Inputs come only from the earlier election.
turnout_features = shares_2016.copy()
turnout_features["PreviousTurnoutRate"] = (
    turnout_2016.loc[eligible_wards, "TurnoutRate"]
)

turnout_target = turnout_2021.loc[eligible_wards, "TurnoutRate"]

turnout_model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=3,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

turnout_model.fit(
    turnout_features.loc[train_wards],
    turnout_target.loc[train_wards]
)

turnout_rf_valid = pd.Series(
    turnout_model.predict(turnout_features.loc[validation_wards]),
    index=validation_wards
).clip(0, 1)

turnout_actual = turnout_target.loc[validation_wards]
turnout_baseline = turnout_2016.loc[validation_wards, "TurnoutRate"]

turnout_results = []
for name, prediction in [
    ("Persistence baseline", turnout_baseline),
    ("Random Forest", turnout_rf_valid)
]:
    error_pp = (prediction - turnout_actual) * 100
    turnout_results.append({
        "Model": name,
        "Turnout MAE (pp)": error_pp.abs().mean(),
        "Turnout RMSE (pp)": np.sqrt((error_pp ** 2).mean())
    })

turnout_comparison = pd.DataFrame(turnout_results)
display(turnout_comparison.round(2))


# Refit using all 53 qualifying historical ward pairs.
model.fit(shares_2016, shares_2021)
turnout_model.fit(turnout_features, turnout_target)

# Prepare 2021 inputs for every ward in the metro.
all_votes_2021 = ward_votes.loc[
    ward_votes["ElectionYear"].eq(2021)
].pivot(
    index="Ward", columns="PartyName", values="TotalValidVotes"
).reindex(columns=all_categories).fillna(0)

all_shares_2021 = all_votes_2021.div(
    all_votes_2021.sum(axis=1), axis=0
)

forecast_shares = pd.DataFrame(
    model.predict(all_shares_2021),
    index=all_shares_2021.index,
    columns=all_categories
).clip(lower=0)

forecast_shares = forecast_shares.div(
    forecast_shares.sum(axis=1), axis=0
)

future_turnout_features = all_shares_2021.copy()
future_turnout_features["PreviousTurnoutRate"] = (
    turnout_2021["TurnoutRate"].reindex(all_shares_2021.index)
)

assert future_turnout_features.notna().all().all()

forecast_turnout_rates = pd.Series(
    turnout_model.predict(future_turnout_features),
    index=all_shares_2021.index,
    name="EstimatedTurnoutRate"
).clip(0, 1)

ward_forecast = pd.DataFrame({
    "ProjectedLeadingParty": leading_party(forecast_shares),
    "EstimatedTurnoutPercent": forecast_turnout_rates * 100,
    "LargestCategorySharePercent": forecast_shares.max(axis=1) * 100
})

print("Conditional 2026 estimates for selected 2021 ward footprints:")
display(ward_forecast.loc[selected_wards].round(1))

# Scenario assumption: retain the 2021 electorate and MEC7 counts.
electorate = (
    turnout_2021["RegisteredVoters"] + turnout_2021["MEC7Votes"]
).reindex(forecast_shares.index)

forecast_voters = forecast_turnout_rates * electorate

# Convert turnout to valid Ward ballots using the observed 2021 ratio.
# Official turnout can differ from the number of Ward ballots.
valid_ward_ratio = (
    all_votes_2021.sum(axis=1)
    / turnout_2021["VoterTurnout"].reindex(forecast_shares.index)
)

assert valid_ward_ratio.between(0, 1).all()

forecast_valid_ward_votes = forecast_voters * valid_ward_ratio

forecast_party_votes = forecast_shares.mul(
    forecast_valid_ward_votes, axis=0
)

metro_forecast = forecast_party_votes.sum().to_frame(
    "EstimatedValidWardVotes"
)
metro_forecast["EstimatedSharePercent"] = (
    metro_forecast["EstimatedValidWardVotes"]
    / metro_forecast["EstimatedValidWardVotes"].sum() * 100
)
metro_forecast = metro_forecast.sort_values(
    "EstimatedValidWardVotes", ascending=False
)

# Sensitivity range based on historical validation MAE.
# This is NOT a statistical confidence interval.
turnout_error = float(
    turnout_comparison.loc[
        turnout_comparison["Model"].eq("Random Forest"),
        "Turnout MAE (pp)"
    ].iloc[0]
) / 100

turnout_summary = pd.DataFrame({
    "Scenario": ["Lower turnout", "Central estimate", "Higher turnout"],
    "EstimatedVoters": [
        (forecast_turnout_rates.sub(turnout_error).clip(0, 1)
         * electorate).sum(),
        forecast_voters.sum(),
        (forecast_turnout_rates.add(turnout_error).clip(0, 1)
         * electorate).sum()
    ]
})
turnout_summary["EstimatedTurnoutPercent"] = (
    turnout_summary["EstimatedVoters"] / electorate.sum() * 100
)

assert np.isclose(
    metro_forecast["EstimatedValidWardVotes"].sum(),
    forecast_valid_ward_votes.sum()
)

print("Conditional metro turnout scenarios:")
display(turnout_summary.round(1))

print("Conditional party/category estimates — top 10:")
display(metro_forecast.head(10).round(1))

from pathlib import Path
import shutil

 # project_dir was set in the loading cell
data_dir = project_dir / "data"
results_dir = project_dir / "results"

data_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# Preserve the source data used in the analysis.
raw_2016.to_csv(data_dir / "elections_2016_all_ballots.csv", index=False)
df_2021.drop(columns="ElectionYear").to_csv(
    data_dir / "elections_2021_ward.csv", index=False
)




# Export prepared historical data.
data.to_csv(data_dir / "historical_ward_votes.csv", index=False)
turnout_data.to_csv(data_dir / "historical_turnout.csv", index=False)

# Export forecasts and validation results.
metro_forecast.to_csv(results_dir / "metro_forecast.csv")
ward_forecast.to_csv(results_dir / "ward_forecast.csv")
forecast_shares.to_csv(results_dir / "ward_forecast_shares.csv")
turnout_summary.to_csv(results_dir / "turnout_scenarios.csv", index=False)
model_comparison.to_csv(results_dir / "vote_model_validation.csv", index=False)
turnout_comparison.to_csv(
    results_dir / "turnout_model_validation.csv", index=False
)

print("Exported files:")
for path in sorted(project_dir.rglob("*")):
    if path.is_file():
        print(path.relative_to(project_dir))