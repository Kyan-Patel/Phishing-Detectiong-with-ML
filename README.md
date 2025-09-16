# 🛡️ Phishing Email Detection Using Machine Learning

A comprehensive machine learning project that detects phishing emails through behavioral and textual analysis, achieving **99.45% accuracy** using iterative feature engineering and natural language processing techniques.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Project Overview

Phishing attacks remain one of the most prevalent cybersecurity threats, often bypassing traditional rule-based email filters. This project demonstrates how machine learning can significantly improve phishing detection by analyzing email content and behavioral patterns.

### Key Achievements
- 📈 **99.45% accuracy** on 82,500+ email dataset
- 🔄 **Iterative improvement** from 66% baseline to 99%+ final models
- 🧠 **Multiple ML approaches** including Logistic Regression and Random Forest
- 📊 **Comprehensive evaluation** with detailed performance metrics

## 📋 Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Versions](#model-versions)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Advanced Feature Engineering**: URL detection, sender domain analysis, capitalization patterns
- **Text Processing**: TF-IDF vectorization with customizable n-grams
- **Multiple Model Versions**: Progressive enhancement from baseline to production-ready
- **Comprehensive Evaluation**: Accuracy, F1-score, confusion matrices, and classification reports
- **Modular Design**: Reusable components for different model configurations
- **Performance Tracking**: Built-in version comparison and performance logging

## 📊 Dataset

The project uses the **CEAS-08 Phishing Email Dataset** containing approximately 82,500 email records with the following features:

| Feature | Description | Type |
|---------|-------------|------|
| `sender` | Email sender address | String |
| `receiver` | Email receiver address | String |
| `subject` | Email subject line | String |
| `body` | Email body content | String |
| `label` | Phishing (1) or Legitimate (0) | Binary |
| `urls` | Number of URLs in email | Integer |

### Dataset Requirements
- Place the `CEAS_08.csv` file in the project root directory
- Ensure the dataset contains all required columns
- Total dataset size: ~82,500 emails with balanced class distribution

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/phishing-detection.git
cd phishing-detection
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
   - Obtain the CEAS-08 dataset
   - Place `CEAS_08.csv` in the project root directory

## 🎯 Usage

### Quick Start

```python
# Load and run all model versions
from phishing_detection import load_and_preprocess, run_version, print_performance

# Load and preprocess data
df = load_and_preprocess()

# Run different model versions
run_version('1.1', df)  # Sender domain features
run_version('1.2', df)  # URL features  
run_version('2.0', df)  # Random Forest with all features

# Compare performance
print_performance()
```

### Running Individual Models

```python
# Run specific version
result = run_version('1.2', df, use_previous=True)
print(f"Accuracy: {result['Accuracy']:.4f}")
print(f"F1-Score: {result['F1 Score']:.4f}")
```

### Jupyter Notebook

For interactive analysis, open the provided Jupyter notebook:
```bash
jupyter notebook Phishing_Detection_Program.ipynb
```

## 🔄 Model Versions

### Version 1.0 - Baseline
- **Features**: Basic behavioral metrics (email length, capitalization, keywords)
- **Model**: Logistic Regression
- **Performance**: 66.1% accuracy
- **Purpose**: Establish baseline performance

### Version 1.1 - Sender Domain Enhancement
- **New Features**: Sender domain frequency analysis
- **Enhancement**: TF-IDF vectorization of email body and subject
- **Performance**: 99.40% accuracy, 99.46% F1-score
- **Key Insight**: Domain reputation significantly improves detection

### Version 1.2 - URL Analysis (Best Performing)
- **New Features**: URL count and presence detection
- **Performance**: **99.45% accuracy, 99.50% F1-score**
- **Key Insight**: URL patterns are strong phishing indicators

### Version 2.0 - Ensemble Approach
- **Model**: Random Forest with all features
- **Performance**: 99.20% accuracy, 99.27% F1-score
- **Trade-off**: Slightly lower performance but captures complex feature interactions

## 📈 Results

### Performance Comparison

| Version | Model | Accuracy | F1-Score | Key Features |
|---------|-------|----------|----------|--------------|
| 1.0 | Logistic Regression | 66.10% | 66.00% | Basic behavioral |
| 1.1 | Logistic Regression | 99.40% | 99.46% | + Sender domain |
| 1.2 | Logistic Regression | **99.45%** | **99.50%** | + URL features |
| 2.0 | Random Forest | 99.20% | 99.27% | All features |

### Best Model Performance (Version 1.2)
```
Confusion Matrix:
[[3468   22]
 [  21 4320]]

Classification Metrics:
- Accuracy: 99.45%
- Precision: 99.49%
- Recall: 99.50%
- F1-Score: 99.50%
```

## 📁 Project Structure

```
phishing-detection/
├── README.md
├── requirements.txt
├── Phishing_Detection_Program.ipynb
├── CEAS_08.csv                    # Dataset (not included)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Data cleaning functions
│   ├── feature_engineering.py     # Feature extraction
│   ├── models.py                  # Model implementations
│   └── evaluation.py              # Performance metrics
├── results/
│   ├── model_comparison.csv
│   └── confusion_matrices.png
├── docs/
│   └── technical_report.pdf
└── examples/
    └── quick_start_example.py
```

## 🔧 Technical Details

### Feature Engineering
- **Text Processing**: TF-IDF vectorization with unigram and bigram features
- **Behavioral Features**: Email length, capitalization ratios, keyword presence
- **Domain Analysis**: Sender domain frequency and reputation scoring
- **URL Features**: Count and presence of hyperlinks

### Preprocessing Pipeline
1. Text normalization (lowercase, special character removal)
2. URL standardization and placeholder replacement
3. Missing value handling
4. Feature scaling and encoding

### Model Architecture
- **Primary**: Logistic Regression with L2 regularization
- **Comparison**: Random Forest ensemble
- **Evaluation**: Stratified train-test split (80/20)
- **Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix

### Performance Optimization
- Sparse matrix operations for memory efficiency
- Vectorizer reuse across model versions
- Modular design for easy experimentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- CEAS-08 Dataset contributors for providing the phishing email dataset
- scikit-learn community for excellent machine learning tools
- Cybersecurity research community for phishing detection insights

## 📧 Contact

**Author**: Kyan Patel
- LinkedIn: [patelkyan](https://www.linkedin.com/in/patelkyan/)
- Email: patelkyan@gmail.co,
- GitHub: [@Kyan-Patel](https://github.com/Kyan-Patel)

---

*This project demonstrates practical application of machine learning in cybersecurity, showcasing the effectiveness of feature engineering and iterative model improvement in solving real-world security challenges.*

