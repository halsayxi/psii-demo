# PSII Interactive Demo

This directory contains a standalone demo webpage for **PSII**.

## What It Shows

- Background: LLM synthetic agents for public opinion simulation.
- Diversity Collapse: why prompt-only persona conditioning can flatten social differences.
- PSII Framework: representation-level injection of demographic and value vectors into intermediate hidden states.
- Interactive Simulation: choose a WVS question, LLM model, and up to seven methods; compare each simulated distribution against Human ground truth.
- Main Results: paper figures for KL divergence, distributional fidelity, and diversity.

Use the top navigation bar to show all sections or focus on one part of the demo at a time.

## Data Sources

The app reads from:

- `demo/data/questions.json`
- `demo/data/WVS_Cross-National_Wave_7_csv_v6_0_100.csv`
- `demo/outputs/{model_name}/{method}/*.json`
- `demo/outputs/simvbg/results_{model_name}_100.xlsx`
- `demo/agent_profile/descriptions/wvs_demographic_descriptions_100.json`
- `demo/assets/*.png`

## Install

From the project root:

```bash
python3 -m venv demo/.venv
source demo/.venv/bin/activate
pip install -r demo/requirements.txt
```

## Run

From the project root:

```bash
streamlit run demo/app.py
```

From inside a standalone copied `demo/` directory:

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```
