import os
import json
import traceback
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Research Pipeline",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI Research Pipeline")
st.caption("Turn any topic into a fully cited research report — powered by AI")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Planner** — breaks your topic into focused search angles  \n"
        "2. **Search** — finds fresh sources from across the web  \n"
        "3. **Validator** — picks the most credible sources  \n"
        "4. **Extractor** — reads and pulls out facts, metrics, and quotes  \n"
        "5. **Writer** — builds the final report with inline citations"
    )
    st.markdown("---")
    st.markdown(
        "**Pro tip:** Use specific topics for richer reports.  \n"
        "*Example:* \"RAG evaluation benchmarks and failure modes in enterprise settings\""
    )
    st.markdown("---")
    show_debug = st.checkbox("Show behind-the-scenes details (debug)", value=False)

# Check keys before we even try to import the backend
missing = []
for k in ["GROQ_API_KEY", "TAVILY_API_KEY"]:
    if not os.getenv(k):
        missing.append(k)

if missing:
    st.error(f"Missing environment variables: {', '.join(missing)}")
    st.stop()

# ---- DEFERRED IMPORT: if agents/crewai is broken, we still see the UI above ----
try:
    from agents import run_research_pipeline
except Exception as import_err:
    st.error(f"Backend import failed: {import_err}")
    st.caption("Full traceback:")
    st.code(traceback.format_exc(), language="python")
    st.stop()

# ---- UI ----
topic = st.text_input(
    "Enter research topic",
    placeholder="e.g. Retrieval-Augmented Generation evaluation benchmarks and failure modes"
)

run_btn = st.button("Run Research Pipeline", type="primary")

progress_bar = st.progress(0)
status_text = st.empty()

def status_cb(msg, pct):
    status_text.info(msg)
    progress_bar.progress(min(pct, 100))

if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    with st.spinner("Running research pipeline..."):
        try:
            report, debug = run_research_pipeline(topic.strip(), status_cb=status_cb)
            st.success("Pipeline completed successfully!")

            st.subheader("Research Report")
            st.markdown(report)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download Report (.md)",
                    data=report,
                    file_name="research_report.md",
                    mime="text/markdown"
                )

            with col2:
                st.download_button(
                    "Download Debug JSON (.json)",
                    data=json.dumps(debug, indent=2, ensure_ascii=False),
                    file_name="pipeline_debug.json",
                    mime="application/json"
                )

            if show_debug:
                st.subheader("Intermediate Pipeline Data")
                st.json(debug)

        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")
            if show_debug:
                st.code(traceback.format_exc(), language="python")
