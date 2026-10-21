# ChangeInsight 🛰️

**Change-Aware Image Captioning for Bi-Temporal Remote Sensing Imagery**

ChangeInsight is a deep learning project that analyzes satellite images captured at different times to detect geographical changes, classify their semantic categories, and generate natural-language descriptions.

## Features

- Pixel-level change localization using a Siamese ResNet18 and U-Net-style decoder.
- Semantic change classification using a Siamese ResNet18 classifier.
- Natural-language caption generation using a ResNet18 encoder and GRU decoder.
- Change probability maps and binary mask visualization.
- Interactive Streamlit web application.
- Individual model execution and complete three-stage analysis.

## System Architecture

The system consists of three models:

1. **Model 1 — Change Localization:** Identifies where changes occur between two satellite images.
2. **Model 2 — Semantic Classification:** Predicts the semantic category of detected changes.
3. **Model 3 — Caption Generation:** Generates a natural-language description of the observed changes.

## Datasets

This project uses two complementary remote sensing datasets:

- **LEVIR-CC:** Bi-temporal satellite image pairs with natural-language change captions.
- **LEVIR-CD:** Bi-temporal satellite image pairs with pixel-level change annotations.

LEVIR-CC supports caption generation and caption-derived semantic supervision, while LEVIR-CD supports change-localization training and evaluation.

## Models and Technologies

| Component | Architecture / Technology |
|---|---|
| Change localization | Siamese ResNet18 + U-Net-style decoder |
| Semantic classification | Siamese ResNet18 classifier |
| Caption generation | ResNet18 encoder + GRU decoder |
| Deep learning | PyTorch, Torchvision |
| Image processing | Pillow, NumPy |
| Data processing | Pandas |
| Evaluation | IoU, Dice, Precision, Recall, BLEU, METEOR, ROUGE-L |
| Web interface | Streamlit |
| Development | Python, JupyterLab, Git |

## Results

### Model 1 — Change Localization

Test-set results using a prediction threshold of 0.55:

| Metric | Score |
|---|---:|
| IoU | 0.8128 |
| Dice / F1 Score | 0.8967 |
| Precision | 0.8995 |
| Recall | 0.8939 |
| Pixel Accuracy | 0.9895 |

### Model 3 — Caption Generation

Validation-set results:

| Metric | Score |
|---|---:|
| BLEU-1 | 0.6959 |
| BLEU-2 | 0.5823 |
| BLEU-3 | 0.4853 |
| BLEU-4 | 0.3920 |
| METEOR | 0.6472 |
| ROUGE-L | 0.6607 |

*Model 2's final validated evaluation results are pending. Caption-generation scores are from the validation set and should not be interpreted as test-set results.*

## Installation

### Prerequisites

- Python 3.11
- Git
- PyTorch and Torchvision
- CUDA-compatible NVIDIA GPU for GPU acceleration (optional)

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd ChangeInsight
```

Replace `<YOUR_REPOSITORY_URL>` with your repository's Git URL.

### 2. Create a virtual environment

```bash
conda create -n changeinsight python=3.11
conda activate changeinsight
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure model checkpoints

Place the trained model checkpoints and caption vocabulary in the paths configured in `app.py`. Ensure that each checkpoint matches its model architecture and label or vocabulary mapping.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## Usage

1. Upload the earlier satellite image.
2. Upload the later image of the same geographical area.
3. Select an individual model or the complete pipeline.
4. Configure the prediction threshold and visualization options.
5. Run the analysis.
6. View the predicted change mask, probability map, semantic category, and generated caption.

For reliable results, use corresponding image pairs with compatible dimensions and preprocessing.

## Limitations

- Small or subtle changes may be difficult to detect.
- Caption-derived semantic labels may contain ambiguity.
- Predictions depend on image quality, alignment, and training-data coverage.
- Generated captions may require manual verification.
- Model outputs should not be treated as authoritative geospatial ground truth without independent validation.

## Future Work

- Improve semantic classification and label quality.
- Explore attention-based and transformer-based captioning.
- Evaluate performance across additional geographical regions.
- Improve handling of subtle changes and unchanged areas.
- Develop joint evaluation methods for localization, classification, and caption generation.

## Acknowledgements

This project uses the LEVIR-CC and LEVIR-CD datasets for research on remote sensing change understanding. Please consult the original dataset publications for attribution and citation requirements.

## License

Add an appropriate `LICENSE` file before distributing the repository. Verify dataset and pretrained-weight licenses separately.

## Author

**Aishwarya Gilhotra**  
BCA (Honours), Artificial Intelligence and Machine Learning  
Symbiosis Institute of Computer Studies and Research (SICSR), Pune
# DL-Temporal-Satellite-data-Change-Captioning
