from pathlib import Path

import pandas as pd
import streamlit as st


DATA_ANALYSIS_ROOT = Path(__file__).resolve().parents[4]
AUTO_INSURANCE_DIR = DATA_ANALYSIS_ROOT / "Python" / "Auto Insurance Analysis"
DATA_PATH = (
    AUTO_INSURANCE_DIR
    / "data"
    / "WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv"
)


@st.cache_data
def load_auto_insurance_data():
    data = pd.read_csv(DATA_PATH)
    for column in (
        "Customer Lifetime Value",
        "Income",
        "Monthly Premium Auto",
        "Total Claim Amount",
    ):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


st.title("Auto Insurance Analysis")
st.caption(
    "Explore customer lifetime value, policy-advertisement responses, "
    "and the project’s churn and predictive analyses."
)

data = load_auto_insurance_data()
response_filter = st.sidebar.selectbox(
    "Campaign response",
    ("All customers", "Responded", "Did not respond"),
)
if response_filter == "Responded":
    filtered_data = data[data["Response"] == "Yes"]
elif response_filter == "Did not respond":
    filtered_data = data[data["Response"] == "No"]
else:
    filtered_data = data

response_rate = data["Response"].eq("Yes").mean()
metrics = st.columns(4)
metrics[0].metric("Customers", f"{len(filtered_data):,}")
metrics[1].metric(
    "Average customer lifetime value",
    f"${filtered_data['Customer Lifetime Value'].mean():,.0f}",
)
metrics[2].metric("Campaign response rate", f"{response_rate:.1%}")
metrics[3].metric(
    "Average monthly premium",
    f"${filtered_data['Monthly Premium Auto'].mean():,.0f}",
)

clv_tab, response_tab, retention_tab, modeling_tab = st.tabs(
    ["Customer lifetime value", "Campaign response", "Churn and retention", "Predictive analysis"]
)

with clv_tab:
    st.subheader("Customer lifetime value")
    clv_by_policy = (
        filtered_data.groupby("Policy Type")["Customer Lifetime Value"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(clv_by_policy.rename("Average CLV"))
    st.image(
        str(AUTO_INSURANCE_DIR / "CLV_contribution.png"),
        caption="CLV contribution by policy type, vehicle size, and vehicle class",
        use_container_width=True,
    )
    st.image(
        str(AUTO_INSURANCE_DIR / "CLV_hist.jpg"),
        caption="Customer lifetime value across customer categories",
        use_container_width=True,
    )

with response_tab:
    st.subheader("Response to policy offers")
    response_by_offer = (
        data.groupby("Renew Offer Type")["Response"]
        .apply(lambda responses: responses.eq("Yes").mean())
        .sort_values(ascending=False)
        .rename("Response rate")
        .to_frame()
    )
    st.bar_chart(response_by_offer)
    st.image(
        str(AUTO_INSURANCE_DIR / "Response.png"),
        caption="Response analysis from the project notebook",
        use_container_width=True,
    )

with retention_tab:
    st.subheader("Hazard and survival analysis")
    st.image(
        str(AUTO_INSURANCE_DIR / "hazard_survival_churn_analysis.png"),
        caption="Exploratory hazard and survival analysis",
        use_container_width=True,
    )
    st.info(
        "The notebook estimates hazard and survival patterns from policy-tenure "
        "records. Treat these as exploratory patterns rather than observed "
        "customer-level churn labels."
    )

with modeling_tab:
    st.subheader("Predictive modeling")
    st.write(
        "The project notebook explores regression models for customer lifetime "
        "value and classification models for campaign response. The models are "
        "not saved as deployable artifacts, so this page presents the analysis "
        "rather than generating live predictions."
    )
    st.link_button(
        "Open the full analysis notebook",
        "https://github.com/Shivani-Chanaliya/Indian-Air-Quality-Analysis/blob/main/"
        "Data-Analysis/Python/Auto%20Insurance%20Analysis/AutoInsuranceAnalysis.ipynb",
    )

with st.expander("Preview the source data"):
    st.dataframe(filtered_data.head(100), use_container_width=True, hide_index=True)
