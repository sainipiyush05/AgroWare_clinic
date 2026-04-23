# 🌱 AgroWare — Plant Disease Prediction System

![AgroWare Header](https://img.shields.io/badge/AgroWare-Plant%20Disease%20Prediction-green?style=for-the-badge&logo=leaf)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange?style=for-the-badge&logo=tensorflow)

AgroWare is a powerful AI-driven utility designed to help farmers and agricultural researchers detect plant diseases early and provide actionable remedies. Leveraging state-of-the-art Deep Learning models (MobileNetV2), it provides high-accuracy diagnosis across various crops.

## ✨ Key Features

- **🚀 High-Performance Inference**: Supports both Keras (`.h5`/`.keras`) and TFLite formats for desktop and mobile performance.
- **🌐 Multilingual Support**: Provides remedy recommendations in **English**, **Hindi (हिन्दी)**, and **Punjabi (ਪੰਜਾਬੀ)**.
- **💊 Remedy Database**: Comprehensive treatment suggestions including chemical control, organic remedies, and cultural practices.
- **📦 Batch Processing**: Analyze entire directories of images and generate CSV/JSON reports.
- **📊 Confidence Scoring**: Detailed probability breakdowns for every prediction.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AgroWare_dIseaSE
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

> [!TIP]
> The project is now configured to automatically detect the pre-trained models (`agroware_model.keras`) and labels (`class_labels.json`) in the root directory. You no longer need to create a `model_output` folder.

## 🚀 How to Run

### 1. Basic Prediction
To run a simple prediction on a single image using the default model:
```bash
python predict.py --image "your_image.jpg"
```

### 2. Prediction with Remedy Suggestions (Recommended)
This is the main entry point for the full system. It provides the diagnosis along with detailed treatment advice.

**Available Flags:**
- `--image`: Path to the leaf image (Required)
- `--lang`: Output language (`en`, `hi`, `pa`)
- `--threshold`: Minimum confidence level (0-1, default 0.3)
- `--top-k`: Number of top results to show

**Run Examples:**
```bash
# Run in Hindi with higher confidence threshold
python predict_with_remedies.py --image "leaf.jpg" --lang hi --threshold 0.5

# Run in Punjabi and save output to a file
python predict_with_remedies.py --image "leaf.jpg" --lang pa --output results.txt
```

### 3. Batch Mode (Processing Folders)
Process all images in a folder and generate a summary report:
```bash
python predict_with_remedies.py --batch --images "data/*.jpg" --format csv --output diagnosis_report.csv
```

## 📂 Project Structure

```text
├── predict.py                # Core inference script (Keras/TFLite)
├── predict_with_remedies.py   # Advanced script with multilingual remedies
├── disease_remedies.json     # Main database for treatments
├── agroware_model.keras      # Pre-trained model weights
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 🧠 Supported Crops & Diseases
The system is trained on the PlantVillage dataset and custom regional data, covering crops like:
- **Wheat**, **Rice**, **Potato**, **Tomato**, **Corn**, **Apple**, and more.

## ⚠️ Disclaimer
*AgroWare is an AI-assisted diagnostic tool. For critical agricultural decisions, always consult with a certified agricultural expert or local extension office.*

---
Made with ❤️ for Sustainable Agriculture
