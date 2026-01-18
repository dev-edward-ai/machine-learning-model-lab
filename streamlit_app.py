from __future__ import annotations

import base64
from typing import Optional

import pandas as pd
import streamlit as st

from backend.api.services.auto_model import recommend_and_run_best_model

# Streamlit UI only; model selection comes from backend.api.services.auto_model

def main() -> None:
    st.set_page_config(page_title="Insight ML Workbench", layout="wide")
    st.title("Insight ML Workbench")
    st.write("Auto-detect task type, run a quick tournament, and recommend the best model.")

    user_intent = st.text_input("What is your goal?", placeholder="e.g., predict churn or forecast sales")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Source dataset")

    if uploaded is None:
        st.info("Upload a CSV to begin.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    if df.empty:
        st.warning("The uploaded file has no rows after loading.")
        return

    st.write("### Columns detected")
    st.dataframe(df.head(), use_container_width=True)

    target_options = ["<None - clustering>"] + list(df.columns)
    target_choice = st.selectbox("Target column (label)", target_options, index=1 if len(target_options) > 1 else 0)
    target_col = None if target_choice == "<None - clustering>" else target_choice

    if st.button("Run Auto Model Selection", type="primary"):
        try:
            result = recommend_and_run_best_model(df, target_col, user_intent)
            best_model = result["model_object"]
            task_type = result["task_type"]

            st.success(result["reasoning"])
            st.write(f"**Recommended:** {result['recommended_model_name']} ({task_type})")

            # Build predictions for download/preview
            features = df if target_col is None else df.drop(columns=[target_col])
            preds = best_model.predict(features)
            output_df = df.copy()
            pred_col = "prediction" if target_col is None else f"predicted_{target_col}"
            output_df[pred_col] = preds

            st.write("### Preview of predictions")
            st.dataframe(output_df.head(), use_container_width=True)

            csv_bytes = output_df.to_csv(index=False).encode("utf-8")
            b64 = base64.b64encode(csv_bytes).decode("utf-8")
            st.download_button(
                label="Download Predictions CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Failed to recommend or run model: {exc}")


if __name__ == "__main__":
    main()
