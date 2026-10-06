\# LogisChain AI



\## Supply Chain Intelligence for Financial Risk



LogisChain AI is a dual-domain supply-chain intelligence prototype that connects logistics risk with financial risk analysis.



The system is designed to support:



\- Shipment risk prediction

\- Supply-chain disruption analysis

\- Financial exposure assessment

\- Expected-loss estimation

\- Working-capital and Cash Conversion Cycle analysis

\- Trade finance portfolio analysis

\- Supply Chain Finance pricing support

\- Disruption scenario simulation



\## Architecture



\### Physical Layer

Represents supply-chain entities and relationships including countries, ports and logistics links.



\### Intelligence Layer

Analyzes shipment, supplier, route, carrier and port risk using machine-learning and engineered supply-chain indicators.



\### Financial Layer

Translates operational disruption into:



\- Probability of Default (PD)

\- Loss Given Default (LGD)

\- Exposure at Default (EAD)

\- Expected Loss

\- Cash Conversion Cycle (CCC)

\- Working Capital Vulnerability Index (WCVI)

\- Trade Risk Financing Stress Index (TRFSI)

\- Supply Chain Credit Score



\## Models



The current implementation includes:



\- Random Forest shipment-risk model

\- XGBoost shipment-risk model

\- Feature engineering

\- SHAP explainability

\- Financial risk scoring

\- Scenario-based simulation



Advanced research models described in the original project specification are treated as future extensions unless implemented and evaluated in this repository.



\## LogisChain Lab



The prototype includes two financial simulation modes:



1\. Trade Finance Portfolio Management

2\. Supply Chain Finance Pricing



The simulation evaluates disruption scenarios and their effects on expected loss, financing exposure and working capital.



\## Disruption Scenarios



The prototype includes scenarios such as:



\- Port Congestion

\- Supplier Failure

\- Geopolitical Disruption

\- Demand Shock

\- Carrier Disruption

\- Fuel Cost Shock

\- Working Capital Stress

\- Multi-node Disruption



\## Running the Project



Install dependencies:



```bash

pip install -r requirements.txt

