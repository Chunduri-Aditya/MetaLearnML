import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

# Add current directory to path
sys.path.append(str(Path(__file__).parent))


# Legacy import - update to use engine.orchestrator.run_automl() in future
from engine.evaluator import accuracy_metric, confusion_matrix_for_model, f1_macro_metric
from engine.legacy.search_controller_v0 import run_model_search
from engine.predictor import BestModelPredictor
from utils.datasets import load_csv_classification, load_text_csv_classification

st.set_page_config(
    page_title="AutoML Pro",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for "Premium" feel
st.markdown(
    """
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #4CAF50; 
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #34495e;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)


def save_uploaded_file(uploaded_file):
    try:
        Path("data").mkdir(exist_ok=True)
        with open(os.path.join("data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return os.path.join("data", uploaded_file.name)
    except Exception:
        return None


def main():
    st.title("🔥 AutoML Pro")
    st.markdown("### Intelligent Model Selection & Training System")

    # System Capabilities Section
    with st.expander("ℹ️ System Capabilities"):
        st.markdown("""
        - **🤖 Smart Model Selection**: Automatically trains and evaluates multiple models (Gradient Boosting, Random Forest, MLP, Logistic Regression) to find the best one.
        - **🧠 Intelligent Auto-Detection**: Automatically detects if your dataset is numeric or text-based and switches processing modes.
        - **⚡ Demo Mode**: Includes built-in datasets (Iris, Titanic, Coursera) for instant testing without uploads.
        - **📊 Robust Evaluation**: Optimizes for Accuracy or F1 Score and provides detailed confusion matrices.
        - **🔮 Meta-Learning Ready**: Designed to learn from past experiments to predict model performance.
        """)

    with st.sidebar:
        st.header("Configuration")
        st.info("Upload your dataset to get started.")

        data_source = st.radio("Data Source", ["Upload CSV", "Use Demo Dataset"])

        file_path = None
        dataset_name = None

        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file:
                file_path = save_uploaded_file(uploaded_file)
                dataset_name = uploaded_file.name
        else:
            # Dynamic Demo Mode: List all CSVs in data/
            data_dir = "data"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

            if not csv_files:
                st.error("No CSV files found in data/ folder.")
                file_path = None
            else:
                # Create a friendly map
                demo_map = {f: os.path.join(data_dir, f) for f in csv_files}

                # Sort for consistent order
                sorted_names = sorted(demo_map.keys())

                demo_choice = st.selectbox("Select Dataset", sorted_names)
                selected_path = demo_map[demo_choice]

                if os.path.exists(selected_path):
                    file_path = selected_path
                    dataset_name = demo_choice
                else:
                    st.error(f"⚠️ File not found: {selected_path}")

        if file_path is not None:
            st.success(f"Loaded {dataset_name}")

            # Load preview
            df = pd.read_csv(file_path)
            st.write("Preview:", df.head(3))

            all_cols = list(df.columns)
            label_col = st.selectbox("Select Target Column", all_cols, index=len(all_cols) - 1)

            metric_choice = st.selectbox("Optimization Metric", ["Accuracy", "F1 Score"])
            metric_name = "accuracy" if metric_choice == "Accuracy" else "f1"

            # Smart Mode Detection
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            # Remove label from numeric candidates if it's selected
            if label_col in numeric_cols:
                numeric_cols.remove(label_col)

            default_mode_index = 0
            if len(numeric_cols) == 0:
                default_mode_index = 1
                st.info("ℹ️ No numeric features detected. Switched to Text mode.")

            feature_mode = st.radio(
                "Feature Mode", ["Auto-detect Numeric", "Text (TF-IDF)"], index=default_mode_index
            )

            text_cols = []
            if feature_mode == "Text (TF-IDF)":
                # Default to all non-label object columns
                candidate_text_cols = [c for c in all_cols if c != label_col]
                text_cols = st.multiselect(
                    "Select Text Columns", candidate_text_cols, default=candidate_text_cols[:1]
                )

            start_btn = st.button("🚀 Start AutoML Search")
        else:
            st.warning("Please select or upload a dataset.")
            start_btn = False

    if start_btn and file_path:
        st.divider()
        st.header("Experiment Progress")

        status_text = st.empty()
        progress_bar = st.progress(0)

        try:
            # Setup
            status_text.text("Loading dataset...")
            progress_bar.progress(10)

            if feature_mode == "Auto-detect Numeric":
                try:
                    dataset, in_dim, out_dim = load_csv_classification(
                        path=file_path, feature_cols=None, label_col=label_col
                    )
                except ValueError as e:
                    if "No numeric feature columns found" in str(e):
                        st.error("❌ Error: No numeric columns found for training.")
                        st.info(
                            "💡 Tip: Your dataset seems to contain text. Try switching 'Feature Mode' to 'Text (TF-IDF)' in the sidebar."
                        )
                        return
                    raise e
            else:
                if not text_cols:
                    st.error("Please select at least one text column.")
                    return
                dataset, in_dim, out_dim = load_text_csv_classification(
                    path=file_path, text_cols=text_cols, label_col=label_col, max_features=5000
                )

            st.write(
                f"**Dataset Info:** {len(dataset)} samples, {in_dim} features, {out_dim} classes"
            )

            # Split
            from torch.utils.data import DataLoader, random_split

            n_train = int(0.8 * len(dataset))
            n_val = len(dataset) - n_train
            train_ds, val_ds = random_split(dataset, [n_train, n_val])

            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=64)

            # Config
            metric_fn = accuracy_metric if metric_name == "accuracy" else f1_macro_metric
            base_config = {
                "lr": 1e-3,
                "epochs": 20,
                "loss_fn": torch.nn.CrossEntropyLoss(),
                "metric_fn": metric_fn,
                "metric_name": metric_name,
                "patience": 5,
            }

            # Meta-features
            status_text.text("Computing meta-features...")
            progress_bar.progress(20)
            from meta.meta_features import compute_meta_features_from_dataset

            try:
                meta_features = compute_meta_features_from_dataset(
                    dataset, in_dim, out_dim, len(dataset)
                )
            except Exception:
                meta_features = None

            # Run Search
            status_text.text("Running Model Search (this may take a moment)...")
            progress_bar.progress(40)

            # Capture stdout to show logs
            from io import StringIO

            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            results_csv = "experiments/results.csv"

            best_model_info = run_model_search(
                train_loader=train_loader,
                val_loader=val_loader,
                in_dim=in_dim,
                out_dim=out_dim,
                device=device,
                base_config=base_config,
                results_csv=results_csv,
                dataset_name=dataset_name,
                meta_features=meta_features,
            )

            sys.stdout = old_stdout
            logs = mystdout.getvalue()

            progress_bar.progress(100)
            status_text.text("Search Complete!")

            # Results
            st.divider()
            st.header("🏆 Results")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Best Model")
                st.markdown(
                    f"""
                <div class="metric-card">
                    <h2>{best_model_info["model_name"]}</h2>
                    <p>Metric ({metric_name}): <strong>{best_model_info["metric"]:.4f}</strong></p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.subheader("Leaderboard")
                if os.path.exists(results_csv):
                    res_df = pd.read_csv(results_csv)
                    st.dataframe(
                        res_df[["model_name", "metric", "family"]].sort_values(
                            "metric", ascending=False
                        )
                    )

            # Visualizations
            st.subheader("Model Performance Analysis")

            predictor = BestModelPredictor(
                results_csv=results_csv,
                in_dim=in_dim,
                out_dim=out_dim,
                device=device,
            )

            cm, metrics = confusion_matrix_for_model(predictor.model, val_loader, device)

            st.write("Confusion Matrix:")
            st.write(cm)

            if metrics:
                st.write("Detailed Metrics:", metrics)

            with st.expander("View Training Logs"):
                st.code(logs)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback

            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
