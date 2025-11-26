# Examples & Demos

This folder contains example scripts and demos that demonstrate how to use the AutoML v2 engine.

**Important:** These files are **not used by the core AutoML pipeline**. They are safe to delete without breaking the main system.

---

## Demo Scripts

### `online_demo.py`
Example script demonstrating online learning with the Iris dataset.

**Usage:**
```bash
python examples/online_demo.py
```

### `online_titanic_demo.py`
Example script demonstrating online learning with the Titanic dataset.

**Usage:**
```bash
python examples/online_titanic_demo.py
```

### `stress_test.py`
Stress test suite for testing the AutoML system on various datasets.

**Usage:**
```bash
python examples/stress_test.py
```

### `app.py`
Streamlit web application for interactive AutoML.

**Usage:**
```bash
streamlit run examples/app.py
```

---

## Utilities

Utility scripts for data preparation, meta-model training, etc.

---

## Notes

- These scripts may use legacy imports (e.g., `from engine.legacy.search_controller_v0`)
- They are kept for reference and demonstration purposes
- For production use, use `main.py` with the dataset-agnostic intelligent mode

