from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import (
    available_methods,
    available_models,
    human_distribution,
    qid_from_choice,
    question_choices,
    question_options,
    question_text,
    sample_profiles,
    simulation_distribution,
)
from utils.metrics import metric_summary
from utils.plotting import distribution_figure


DEMO_ASSET_DIR = Path(__file__).resolve().parent / "assets"
PSII_SOURCE_METHOD = "".join(("P", "S", "I", "I"))
PSII_DISPLAY_METHOD = "PSII"
METHOD_DISPLAY_NAMES = {PSII_SOURCE_METHOD: PSII_DISPLAY_METHOD}
NAV_OPTIONS = [
    "All Sections",
    "Background",
    "Framework",
    "Interactive Simulation",
    "Main Results",
]


st.set_page_config(
    page_title="PSII Interactive Demo",
    page_icon="PW",
    layout="wide",
    initial_sidebar_state="expanded",
)


def asset(name: str) -> str:
    return str(DEMO_ASSET_DIR / name)


def method_label(method_name: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method_name, method_name)


def method_sort_key(method_name: str) -> tuple[int, str]:
    if method_label(method_name) == PSII_DISPLAY_METHOD:
        return (0, method_label(method_name).lower())
    return (1, method_label(method_name).lower())


def order_methods(methods: list[str]) -> list[str]:
    return sorted(methods, key=method_sort_key)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #172033;
            --muted: #667085;
            --line: #dbe3ec;
            --panel: #f7f9fc;
            --accent: #1b4d89;
            --warm: #b85c38;
        }
        .stApp {
            background: #ffffff;
            color: var(--ink);
        }
        .block-container {
            max-width: 1240px;
            padding-top: 2.25rem;
            padding-bottom: 4rem;
        }
        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }
        h1 {
            font-size: 2.85rem;
            line-height: 1.05;
            margin-bottom: 0.45rem;
        }
        h2 {
            border-top: 1px solid var(--line);
            padding-top: 2rem;
            margin-top: 2.5rem;
        }
        .lede {
            color: var(--muted);
            font-size: 1.13rem;
            line-height: 1.7;
            max-width: 900px;
        }
        .section-note {
            color: var(--muted);
            line-height: 1.65;
            font-size: 1rem;
        }
        .small-note {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.5;
        }
        .concept-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 1rem 0 1.25rem;
        }
        .concept {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 120px;
        }
        .concept strong {
            display: block;
            color: var(--accent);
            margin-bottom: 6px;
        }
        .method-compare {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin: 1rem 0 1.25rem;
        }
        .method-box {
            border: 1px solid var(--line);
            background: #fff;
            border-radius: 8px;
            padding: 16px;
        }
        .method-box strong {
            color: var(--ink);
        }
        .token {
            display: inline-block;
            border: 1px solid var(--line);
            background: #fff;
            border-radius: 999px;
            padding: 5px 10px;
            margin: 3px 4px 3px 0;
            font-size: 0.9rem;
            color: #344054;
        }
        .callout {
            border-left: 4px solid var(--warm);
            background: #fff8f4;
            padding: 12px 14px;
            border-radius: 6px;
            color: #45342d;
        }
        div[data-testid="stRadio"] {
            background: #ffffff;
            border: 1px solid #c8d3df;
            border-radius: 8px;
            padding: 8px;
            margin: 1.35rem 0 0.7rem;
            box-shadow: 0 8px 22px rgba(23, 32, 51, 0.08);
        }
        div[data-testid="stRadio"] div[role="radiogroup"] {
            background: #eef3f8;
            border: 1px solid #d7e0ea;
            border-radius: 7px;
            padding: 6px;
            gap: 6px;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label {
            background: #ffffff;
            border: 1px solid #d7e0ea;
            border-radius: 6px;
            padding: 8px 12px;
            min-height: 38px;
            box-shadow: 0 1px 2px rgba(23, 32, 51, 0.05);
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
            background: #e6eff8;
            border-color: #a9bed4;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
            background: #ffffff;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
            background: #1b4d89;
            border-color: #1b4d89;
            color: #ffffff;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) span,
        div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) div {
            color: #ffffff;
        }
        @media (max-width: 780px) {
            h1 { font-size: 2.1rem; }
            .concept-grid, .method-compare {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_header() -> None:
    st.markdown("# PSII")
    st.markdown(
        """
        <p class="lede">
        An interactive demo for PSII: a representation-level framework for
        injecting demographic attributes and value orientations into synthetic LLM agents.
        </p>
        """,
        unsafe_allow_html=True,
    )


def display_top_navigation() -> str:
    return st.radio(
        "Choose a demo section",
        NAV_OPTIONS,
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )


def display_metric_table(metric_df: pd.DataFrame) -> None:
    min_kl = metric_df["KL divergence"].min()
    min_ed = metric_df["ED"].min()

    def highlight_min(value: float, min_value: float) -> str:
        if np.isclose(value, min_value, rtol=1e-9, atol=1e-12):
            return "font-weight: 800; background-color: #fff3d8; color: #172033;"
        return ""

    styled = metric_df.style.format(
        {
            "KL divergence": "{:.4f}",
            "ED": "{:.4f}",
            "Normalized entropy": "{:.4f}",
        }
    ).map(lambda value: highlight_min(value, min_kl), subset=["KL divergence"])
    styled = styled.map(lambda value: highlight_min(value, min_ed), subset=["ED"])
    styled = styled.set_table_styles(
        [
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("width", "100%"),
                    ("font-size", "0.94rem"),
                ],
            },
            {
                "selector": "th",
                "props": [
                    ("text-align", "left"),
                    ("background-color", "#f3f6fa"),
                    ("border-bottom", "1px solid #dbe3ec"),
                    ("padding", "9px 10px"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border-bottom", "1px solid #edf1f5"),
                    ("padding", "8px 10px"),
                ],
            },
        ]
    ).hide(axis="index")
    st.markdown(styled.to_html(), unsafe_allow_html=True)


def display_background() -> None:
    st.markdown("## Background")
    st.markdown(
        """
        <div class="concept-grid">
          <div class="concept"><strong>LLM Agents</strong>
          Large language models can simulate survey respondents at scale, making public
          opinion experiments faster and cheaper than repeated human surveys.</div>
          <div class="concept"><strong>Prompt Conditioning</strong>
          Prompt-only personas describe social identity in text, but the model may still
          compress different groups into similar response patterns.</div>
          <div class="concept"><strong>Diversity Collapse</strong>
          PSII frames this failure as hidden-state convergence: distinct social identities
          become less distinguishable in deeper model layers.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.image(asset("diversity-collapse.png"), use_container_width=True)


def display_framework() -> None:
    st.markdown("## PSII Framework")
    st.markdown(
        """
        <p class="section-note">
        PSII moves identity conditioning from prompt text into intermediate hidden states.
        Each synthetic respondent has demographic and value descriptors such as gender,
        age, education, income, political orientation, and value orientation. These signals
        are encoded as parametric identity/value vectors and injected into the model's
        internal representation during generation.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <span class="token">gender</span>
        <span class="token">age</span>
        <span class="token">education</span>
        <span class="token">income</span>
        <span class="token">political orientation</span>
        <span class="token">value orientation</span>
        <span class="token">country</span>
        <span class="token">religion</span>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="method-compare">
          <div class="method-box"><strong>Prompt-only persona conditioning</strong><br>
          Identity appears as natural-language context. The model may attend to it weakly
          or wash it out as generation proceeds.</div>
          <div class="method-box"><strong>PSII representation-level injection</strong><br>
          Identity/value vectors directly modulate hidden states, enabling finer control
          over simulated group differences.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.image(asset("framework.png"), use_container_width=True)

    profiles = sample_profiles(3)
    if profiles:
        with st.expander("Example respondent profiles used by the experiment", expanded=False):
            for idx, profile in enumerate(profiles, start=1):
                st.markdown(f"**Agent {idx}**")
                st.write(profile)


def display_interactive_simulation() -> None:
    st.markdown("## Interactive Simulation")

    models = available_models()
    if not models:
        st.error("No local model outputs were found under demo/outputs/ or demo/output/.")
        return

    with st.sidebar:
        st.header("Simulation Controls")
        default_model_idx = models.index("qwen2.5-14b") if "qwen2.5-14b" in models else 0
        model = st.selectbox("LLM model", models, index=default_model_idx, key="model_selector_v2")
        methods = order_methods(available_methods(model))
        selected_methods = st.multiselect(
            "Methods",
            methods,
            default=methods,
            format_func=method_label,
            key="method_selector_v2",
        )
        choices = question_choices()
        default_q = next((idx for idx, choice in enumerate(choices) if choice.startswith("Q106 - ")), 0)
        question_choice = st.selectbox(
            "World Values Survey question",
            choices,
            index=default_q,
            key="question_selector_v2",
        )

    if not selected_methods:
        st.info("Select at least one method in the sidebar.")
        return

    qid = qid_from_choice(question_choice)
    options = question_options(qid)
    human = human_distribution(qid)
    ordered_selected_methods = order_methods(selected_methods)
    method_distributions = {
        method: simulation_distribution(model, method, qid) for method in selected_methods
    }

    st.markdown(f"### {qid}: {question_text(qid)}")
    st.markdown(
        " ".join([f"<span class='token'>{code}: {label}</span>" for code, label in options.items()]),
        unsafe_allow_html=True,
    )

    fig = distribution_figure(
        list(options.keys()),
        list(options.values()),
        human["probs"],
        {method_label(method): method_distributions[method]["probs"] for method in ordered_selected_methods},
    )
    st.plotly_chart(fig, use_container_width=True)

    rows = []
    for method in ordered_selected_methods:
        dist = method_distributions[method]
        metrics = metric_summary(dist["probs"], human["probs"])
        rows.append(
            {
                "Method": method,
                "Display name": method_label(method),
                "KL divergence": metrics["KL divergence"],
                "ED": metrics["ED"],
                "Normalized entropy": metrics["Normalized entropy"],
                "Valid responses": dist["valid_n"],
                "Ignored responses": dist["invalid_n"],
            }
        )
    metric_df = pd.DataFrame(rows).sort_values("KL divergence", ascending=True)
    metric_df = metric_df.drop(columns=["Method"]).rename(columns={"Display name": "Method"})

    display_metric_table(metric_df)
    st.markdown(
        f"<p class='small-note'>Human valid responses for this question: {human['valid_n']}."
        " Metrics compare each model-method probability vector against the WVS vector."
        " Lower KL indicates closer distributional fit; lower ED means the simulated normalized entropy is closer to the human normalized entropy.</p>",
        unsafe_allow_html=True,
    )


def display_main_results() -> None:
    st.markdown("## Main Results")
    st.markdown(
        """
        <div class="callout">
        Across the reported WVS experiments, PSII reduces KL divergence relative to
        prompt-only and other baseline methods, improves alignment with real survey
        distributions, and increases response diversity across social groups.
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(asset("main-result.png"), caption="Main distributional fidelity results", use_container_width=True)
    with col2:
        st.image(asset("distribution.png"), caption="Response distribution comparison", use_container_width=True)


def main() -> None:
    inject_css()
    display_header()
    selected_part = display_top_navigation()

    if selected_part in ("All Sections", "Background"):
        display_background()
    if selected_part in ("All Sections", "Framework"):
        display_framework()
    if selected_part in ("All Sections", "Interactive Simulation"):
        display_interactive_simulation()
    if selected_part in ("All Sections", "Main Results"):
        display_main_results()


if __name__ == "__main__":
    main()
