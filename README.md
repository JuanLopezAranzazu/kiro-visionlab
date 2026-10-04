# VisionLab

Interactive computer vision playground built with **Python, Streamlit, and OpenCV**.

VisionLab allows users to upload an image, apply image processing transformations interactively, build processing pipelines, visualize the results, and download the processed image.

## Stack

* **Python 3.12+**
* **Streamlit** — Web interface
* **OpenCV** — Computer vision and image processing
* **NumPy** — Image data manipulation

## Project Structure

```text
kiro-visionlab/
├── app.py
├── transforms.py
├── ui.py
├── utils.py
├── requirements.txt
└── README.md
```

### Architecture

* `app.py` — Streamlit entry point and application orchestration.
* `transforms.py` — Image processing functions and pipeline execution.
* `ui.py` — Reusable Streamlit UI components.
* `utils.py` — Image encoding, decoding, and conversion utilities.
* `requirements.txt` — Project dependencies.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/JuanLopezAranzazu/kiro-visionlab.git
cd kiro-visionlab
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit development server:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```