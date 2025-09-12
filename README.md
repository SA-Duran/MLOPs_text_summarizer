# 📚 MLOps Text Summarizer with Transformers

## 🧠 Overview  
This project implements a full MLOps pipeline for abstractive text summarization using Transformer-based models (e.g., T5 or Pegasus). It ingests dialogue-style datasets (like SAMSum), preprocesses them, fine-tunes a seq2seq model, and evaluates the output using ROUGE metrics.

---

## 🧱 Pipeline Structure

The training pipeline is modular and separated into five orchestrated stages:

### 1. **Data Ingestion**  
- Downloads a zipped dataset from Google Drive (via `gdown`)  
- Extracts and stores it into the artifacts folder:contentReference[oaicite:0]{index=0}

### 2. **Data Validation**  
- Verifies that all expected files are present  
- Writes a validation status report:contentReference[oaicite:1]{index=1}

### 3. **Data Transformation**  
- Loads and tokenizes the dataset using a HuggingFace tokenizer  
- Applies `summarize:` prefix, truncation, and label creation  
- Saves the transformed dataset to disk:contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}

### 4. **Model Training**  
- Loads a pre-trained model (e.g. T5 or Pegasus)  
- Uses HuggingFace `Trainer` with `TrainingArguments` for fine-tuning  
- Saves both the fine-tuned model and tokenizer:contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}

### 5. **Model Evaluation**  
- Loads the model and validation set  
- Computes ROUGE scores (rouge1, rouge2, rougeL, rougeLsum)  
- Processes evaluation in batches for memory efficiency:contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}

Each stage is orchestrated by `main.py`, which sequentially triggers and logs the execution of each step:contentReference[oaicite:8]{index=8}.

---

## 🛠 Technologies & Libraries

| Layer              | Libraries / Tools                          |
|--------------------|---------------------------------------------|
| Transformers       | HuggingFace Transformers (`AutoModel`, `Trainer`)  
| Dataset Loading    | HuggingFace Datasets (`load_dataset`, `load_from_disk`)  
| Tokenization       | HuggingFace Tokenizers (`AutoTokenizer`)  
| Evaluation         | `evaluate` library with ROUGE metrics  
| Workflow           | Modular pipeline with `stage_XX_*.py` scripts  
| Data Downloading   | `gdown`, `zipfile`, `os`  
| Logging            | Custom logger module with timestamped logs  
| Configuration Mgmt | Custom `ConfigurationManager` & `entities.py`  
| Hardware Support   | CUDA/CPU adaptive training  
| Batch Management   | `DataCollatorForSeq2Seq` + manual batching  
| CLI (optional)     | `argparse`-driven script runner  
| Evaluation Output  | `.csv` with ROUGE scores (optional)

---

## 📦 Installation

```bash
git clone https://github.com/SA-Duran/MLOPs_text_summarizer.git
cd MLOPs_text_summarizer
pip install -r requirements.txt
```

Or if using `pyproject.toml` (not shown here):

```bash
pip install .
```

---

## 🚀 Usage

### ▶️ Run Full Training Pipeline

```bash
python main.py
```

This will:

1. Download and unzip the dataset  
2. Validate required files  
3. Preprocess into tokenized features  
4. Train the model  
5. Evaluate with ROUGE metrics

### 🧪 Evaluate Model (standalone)

```bash
python stage_05_model_evaluation.py
```

---

## 📁 Output Artifacts

| File / Folder                             | Description                           |
|-------------------------------------------|---------------------------------------|
| `artifacts/data_ingestion/`               | Raw and unzipped dataset              |
| `artifacts/samsum_dataset/`               | Tokenized dataset (input_ids, labels) |
| `artifacts/model/`                        | Trained seq2seq model                 |
| `artifacts/tokenizer/`                    | Tokenizer used for preprocessing      |
| `metrics.csv` or `metrics.json` (optional)| ROUGE evaluation metrics              |

---

## 📄 License

MIT

---

Let me know if you want a visual architecture diagram, example curl commands, or documentation for serving the model via FastAPI or Flask.
