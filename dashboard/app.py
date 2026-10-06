import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = BASE_DIR / "models" / "shipment_risk_model.pkl"
FINANCIAL_FILE = BASE_DIR / "data" / "processed" / "financial_risk_dataset.csv"
NETWORK_NODES = BASE_DIR / "data" / "processed" / "network_nodes.csv"
NETWORK_LINKS = BASE_DIR / "data" / "processed" / "network_links.csv"
SCENARIOS_FILE = BASE_DIR / "data" / "processed" / "disruption_scenarios.csv"
LAB_FILE = BASE_DIR / "data" / "processed" / "logischain_lab_results.csv"


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="LogisChain AI",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 LogisChain AI")
st.caption(
    "Supply Chain Intelligence for Trade Finance, "
    "Working Capital and Credit Risk"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_data():

    financial = pd.read_csv(FINANCIAL_FILE)
    nodes = pd.read_csv(NETWORK_NODES)
    links = pd.read_csv(NETWORK_LINKS)
    scenarios = pd.read_csv(SCENARIOS_FILE)
    lab = pd.read_csv(LAB_FILE)

    return financial, nodes, links, scenarios, lab


model = load_model()

financial, nodes, links, scenarios, lab = load_data()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("LogisChain Lab")

page = st.sidebar.radio(
    "Select Module",
    [
        "Executive Overview",
        "Shipment Risk",
        "Financial Risk",
        "Supply Chain Network",
        "Disruption Scenarios",
        "Trade Finance Portfolio",
        "SCF Pricing"
    ]
)


# =========================================================
# EXECUTIVE OVERVIEW
# =========================================================

if page == "Executive Overview":

    st.header("Executive Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Shipments",
        len(financial)
    )

    c2.metric(
        "Network Nodes",
        len(nodes)
    )

    c3.metric(
        "Network Links",
        len(links)
    )

    c4.metric(
        "Disruption Scenarios",
        len(scenarios)
    )

    st.divider()

    st.subheader("Financial Risk Distribution")

    risk_counts = financial[
        "Financial_Risk_Level"
    ].value_counts()

    st.bar_chart(risk_counts)

    st.subheader("Key Financial Indicators")

    metrics = pd.DataFrame({
        "Metric": [
            "Average PD",
            "Average LGD",
            "Average EAD",
            "Average Expected Loss",
            "Average CCC",
            "Average WCVI",
            "Average Credit Score"
        ],
        "Value": [
            financial["PD"].mean(),
            financial["LGD"].mean(),
            financial["EAD"].mean(),
            financial["Expected_Loss"].mean(),
            financial["CCC"].mean(),
            financial["WCVI"].mean(),
            financial["Supply_Chain_Credit_Score"].mean()
        ]
    })

    st.dataframe(
        metrics,
        use_container_width=True
    )


# =========================================================
# SHIPMENT RISK
# =========================================================

elif page == "Shipment Risk":

    st.header("Shipment Risk Prediction")

    st.write(
        "Analyze the operational risk of an individual shipment."
    )

    col1, col2 = st.columns(2)

    with col1:

        distance = st.number_input(
            "Distance (km)",
            min_value=100.0,
            value=2500.0
        )

        freight = st.number_input(
            "Freight Rate",
            min_value=10.0,
            value=500.0
        )

        congestion = st.slider(
            "Port Congestion",
            0.0,
            1.0,
            0.30
        )

        delay = st.slider(
            "Carrier Delay",
            0.0,
            1.0,
            0.20
        )

    with col2:

        inventory = st.number_input(
            "Inventory Days",
            min_value=1.0,
            value=30.0
        )

        supplier = st.slider(
            "Supplier Risk",
            0.0,
            1.0,
            0.25
        )

        route = st.slider(
            "Route Risk",
            0.0,
            1.0,
            0.25
        )

        demand = st.slider(
            "Demand Change",
            -1.0,
            1.0,
            0.0
        )

    if st.button(
        "Analyze Shipment Risk",
        type="primary"
    ):

        data = pd.DataFrame([{
            "distance_km": distance,
            "freight_rate": freight,
            "port_congestion": congestion,
            "carrier_delay": delay,
            "inventory_days": inventory,
            "supplier_risk": supplier,
            "route_risk": route,
            "demand_change": demand
        }])

        probability = model.predict_proba(
            data
        )[0][1]

        if probability < 0.35:
            risk = "LOW"
        elif probability < 0.65:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        exposure = freight * (
            1 + congestion
        )

        expected_loss = (
            exposure * probability
        )

        st.divider()

        a, b, c = st.columns(3)

        a.metric(
            "Disruption Probability",
            f"{probability:.2%}"
        )

        b.metric(
            "Risk Level",
            risk
        )

        c.metric(
            "Expected Loss",
            f"{expected_loss:.2f}"
        )

        st.subheader("Recommendation")

        if risk == "HIGH":
            st.error(
                "High shipment risk. Review carrier, route "
                "and supplier exposure before financing."
            )
        elif risk == "MEDIUM":
            st.warning(
                "Moderate risk. Consider additional monitoring "
                "and contingency planning."
            )
        else:
            st.success(
                "Low operational risk under the current inputs."
            )


# =========================================================
# FINANCIAL RISK
# =========================================================

elif page == "Financial Risk":

    st.header("Financial Risk Analytics")

    st.write(
        "Supply-chain risk translated into financial exposure."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Average PD",
        f"{financial['PD'].mean():.2%}"
    )

    c2.metric(
        "Average LGD",
        f"{financial['LGD'].mean():.2%}"
    )

    c3.metric(
        "Average EAD",
        f"{financial['EAD'].mean():.2f}"
    )

    c4.metric(
        "Average Expected Loss",
        f"{financial['Expected_Loss'].mean():.2f}"
    )

    st.divider()

    st.subheader("Supply Chain Credit Score")

    st.line_chart(
        financial[
            "Supply_Chain_Credit_Score"
        ].head(100)
    )

    st.subheader("Financial Risk Dataset")

    st.dataframe(
        financial[
            [
                "shipment_id",
                "PD",
                "LGD",
                "EAD",
                "Expected_Loss",
                "CCC",
                "WCVI",
                "TRFSI",
                "Supply_Chain_Credit_Score",
                "Financial_Risk_Level"
            ]
        ].head(100),
        use_container_width=True
    )


# =========================================================
# NETWORK
# =========================================================

elif page == "Supply Chain Network":

    st.header("Supply Chain Network")

    c1, c2 = st.columns(2)

    c1.metric(
        "Nodes",
        len(nodes)
    )

    c2.metric(
        "Links",
        len(links)
    )

    st.subheader("Network Nodes")

    st.dataframe(
        nodes,
        use_container_width=True
    )

    st.subheader("Network Links")

    st.dataframe(
        links,
        use_container_width=True
    )

    st.subheader("Node Risk")

    st.bar_chart(
        nodes.set_index("name")[
            "risk_score"
        ].head(20)
    )


# =========================================================
# SCENARIOS
# =========================================================

elif page == "Disruption Scenarios":

    st.header("Disruption Scenario Analysis")

    selected = st.selectbox(
        "Select Scenario",
        scenarios["scenario"].tolist()
    )

    scenario = scenarios[
        scenarios["scenario"] == selected
    ].iloc[0]

    st.subheader(
        scenario["scenario"]
    )

    st.write(
        scenario["description"]
    )

    a, b, c = st.columns(3)

    a.metric(
        "Delay Shock",
        f"+{scenario['delay_change']:.0%}"
    )

    b.metric(
        "Freight Shock",
        f"+{scenario['freight_change']:.0%}"
    )

    c.metric(
        "Inventory Shock",
        f"+{scenario['inventory_change']} days"
    )

    matching = lab[
        lab["scenario"] == selected
    ]

    if not matching.empty:

        row = matching.iloc[0]

        st.divider()

        x, y, z = st.columns(3)

        x.metric(
            "Portfolio Expected Loss",
            f"{row['portfolio_expected_loss']:.2f}"
        )

        y.metric(
            "Financial Impact",
            f"{row['financial_impact']:.2f}"
        )

        z.metric(
            "Average CCC",
            f"{row['average_CCC']:.1f} days"
        )


# =========================================================
# TRADE FINANCE PORTFOLIO
# =========================================================

elif page == "Trade Finance Portfolio":

    st.header(
        "Trade Finance Portfolio Management"
    )

    st.write(
        "Evaluate logistics-dependent trade finance exposures "
        "under disruption scenarios."
    )

    portfolio = financial.sample(
        n=min(20, len(financial)),
        random_state=42
    )

    total_ead = portfolio["EAD"].sum()
    total_loss = portfolio["Expected_Loss"].sum()

    a, b, c = st.columns(3)

    a.metric(
        "Portfolio Exposures",
        len(portfolio)
    )

    b.metric(
        "Total EAD",
        f"{total_ead:.2f}"
    )

    c.metric(
        "Expected Loss",
        f"{total_loss:.2f}"
    )

    st.subheader("Portfolio Risk")

    st.dataframe(
        portfolio[
            [
                "shipment_id",
                "EAD",
                "PD",
                "LGD",
                "Expected_Loss",
                "Supply_Chain_Credit_Score",
                "Financial_Risk_Level"
            ]
        ],
        use_container_width=True
    )


# =========================================================
# SCF PRICING
# =========================================================

elif page == "SCF Pricing":

    st.header(
        "Supply Chain Finance Pricing"
    )

    st.write(
        "Illustrative pricing support based on "
        "supply-chain financial risk."
    )

    score = st.slider(
        "Supply Chain Credit Score",
        0,
        100,
        70
    )

    pd_value = st.slider(
        "Estimated PD",
        0.0,
        1.0,
        0.10
    )

    base_rate = st.number_input(
        "Base Financing Rate (%)",
        min_value=1.0,
        value=8.0
    )

    risk_premium = (
        pd_value * 10
        + (100 - score) * 0.03
    )

    recommended_rate = (
        base_rate + risk_premium
    )

    a, b = st.columns(2)

    a.metric(
        "Risk Premium",
        f"{risk_premium:.2f}%"
    )

    b.metric(
        "Illustrative Financing Rate",
        f"{recommended_rate:.2f}%"
    )

    if score >= 75:
        st.success(
            "Lower relative supply-chain risk."
        )
    elif score >= 50:
        st.warning(
            "Moderate supply-chain financing risk."
        )
    else:
        st.error(
            "High relative financing risk."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "LogisChain AI | Supply Chain Intelligence + Financial Risk"
)