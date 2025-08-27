# Brain MRI Analysis System

An AI-powered system for brain tumor detection using the **Vision Transformer (ViT)** architecture.
This project includes both **model training** and a **clinical web interface** for analyzing brain MRI scans.

---

## Overview

This system uses a fine-tuned Vision Transformer model to classify brain MRI images into four categories:

* **Glioma** – Aggressive brain tumors requiring urgent attention
* **Meningioma** – Typically benign, slow-growing tumors
* **Pituitary Adenoma** – Tumors affecting hormone production
* **No Tumor** – Normal healthy brain tissue

---

## Project Structure

```
brain-mri-analysis/
├── finetuning_vit.ipynb    # Model training notebook
├── app.py                  # Clinical web interface
├── README.md               # Project documentation
```

---

## Features

### Dataset

This project uses the publicly available **Brain Tumor MRI dataset** on Kaggle:
🔗 [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

### Model Training (`finetuning_vit.ipynb`)

* Fine-tunes Vision Transformer on the MRI dataset
* Stratified data splitting (train/validation/test)
* Real-time metrics tracking and visualization
* Confusion matrix and classification reports
* Attention visualization for model interpretability

### Clinical Interface (`app.py`)

* Professional medical-style interface
* Upload MRI scans for analysis
* Confidence scores for each tumor type
* Visual explanation using attention maps
* Downloadable clinical analysis reports
* Built-in disclaimers and safety warnings

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/BasmaZouaoui/Brain-MRI-Analysis-System.git
cd Brain-MRI-Analysis-System
```

2. **Install dependencies**

```bash
pip install torch torchvision transformers streamlit
pip install scikit-learn matplotlib seaborn pandas numpy
pip install plotly opencv-python pillow huggingface-hub
```

---

## Usage

### Training the Model

1. **Download the dataset** from Kaggle.
2. **Set training parameters** in the notebook:

   ```python
   MODEL_NAME = "google/vit-base-patch16-224-in21k"
   DATA_DIR = "/path/to/brain-tumor-mri-dataset"
   OUTPUT_DIR = "./vit-brain-tumor-classifier"
   ```
3. **Run all cells** in `finetuning_vit.ipynb`.

### Running the Clinical Interface

1. Update the model path in `app.py`.
2. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```
3. Open your browser at **[http://localhost:8501](http://localhost:8501)** to use the interface.

---

## Model Performance

**Final Results (after 15 epochs with early stopping):**

* **Validation Accuracy:** 98.16%
* **Test Accuracy:** 98.04%

### Class-wise Test Performance

| Class      | Precision | Recall | F1-Score | Support |
| ---------- | --------- | ------ | -------- | ------- |
| Glioma     | 98.21%    | 97.52% | 97.86%   | 564     |
| Meningioma | 95.70%    | 97.04% | 96.37%   | 574     |
| No Tumor   | 99.04%    | 99.45% | 99.24%   | 724     |
| Pituitary  | 98.97%    | 97.80% | 98.39%   | 592     |

---

## Key Features

### Training

* Automatic stratified dataset splitting
* Real-time accuracy/loss monitoring
* Visual training plots
* Classification reports & confusion matrices
* Saves model in Hugging Face format

### Clinical Web App

* Intuitive medical-grade interface
* Attention heatmaps for interpretability
* Class confidence scoring
* Medical info and recommendations
* Report generation with disclaimers

---

## Technical Details

### Model

* **Base Model:** ViT-Base-Patch16-224
* **Fine-tuning:** Custom classification head (4 classes)
* **Input Size:** 224×224 pixels
* **Patch Size:** 16×16
* **Architecture:** Multi-head self-attention

### Training

* **Epochs:** 15 (with early stopping)
* **Batch Size:** 64
* **Learning Rate:** 3e-5
* **Optimizer:** AdamW (with weight decay)
* **Evaluation:** Every 10 steps

---

## Data Requirements

* **Accepted formats:** JPEG, PNG, BMP, TIFF
* **Recommended:** T1/T2-weighted MRI scans
* **Resolution:** Auto-resized to 224×224
* **Dataset Size:** ≥100 images per class recommended

---

## Medical Disclaimer ⚠️

This system is **not a substitute for clinical judgment**.

* Always correlate results with medical context
* Confirm findings with professional evaluation
* Use for **educational and assistive purposes only**
* Final diagnosis and treatment remain the responsibility of qualified healthcare providers

---

## Deployment

### Hugging Face Hub

To push your model:

```python
# In training script
push_to_hub=True
```

Configure your Hugging Face token.

### Production Considerations

* Add authentication in clinical environments
* Enable audit logging for compliance
* Set access controls
* Ensure HIPAA/GDPR compliance if applicable

---

## Contributing

1. Fork this repo
2. Create a feature branch
3. Make your changes
4. Add tests (if applicable)
5. Submit a PR

---

## License

This project is for **educational and research purposes only**.
Please ensure compliance with local medical software regulations.

---

## Acknowledgments

* Vision Transformer (Google Research)
* Hugging Face Transformers library
* Brain MRI dataset contributors
* Open-source computer vision community