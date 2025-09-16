# 🛡️ Phishing Email Detection Using Machine Learning

An exploratory machine learning project investigating the application of ML techniques to cybersecurity, specifically phishing email detection. This project explores how behavioral and textual analysis can identify phishing attempts, achieving **99.45% accuracy** through iterative feature engineering and natural language processing experimentation.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Project Overview

This project serves as an exploration into the intersection of machine learning and cybersecurity, specifically examining how ML techniques can be applied to phishing detection. Through systematic experimentation with different feature engineering approaches and model architectures, this work investigates the effectiveness of automated phishing detection compared to traditional rule-based systems.

### Research Questions Explored
- 🔍 **How effective are ML techniques** compared to traditional rule-based phishing filters?
- 📈 **What impact does feature engineering** have on detection accuracy?
- ⚖️ **How do different algorithms** (linear vs. ensemble) perform on cybersecurity datasets?
- 🧠 **Can behavioral patterns** complement content-based detection methods?

## 📋 Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Versions](#model-versions)
- [Research Methodology](#research-methodology)
- [Experimental Results](#experimental-results)
- [License](#license)

## ✨ Research Focus

This project explores several key areas in ML-powered cybersecurity:

- **Feature Engineering Impact**: Investigating how different feature combinations affect detection performance
- **Algorithm Comparison**: Comparing linear models (Logistic Regression) vs. ensemble methods (Random Forest)
- **Incremental Improvement**: Examining the effectiveness of iterative feature enhancement
- **Behavioral Analysis**: Exploring non-textual indicators like sender patterns and URL characteristics
- **Performance Trade-offs**: Analyzing accuracy vs. interpretability in cybersecurity applications

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

### Experimental Approach
- **Baseline Establishment**: Simple feature set to establish performance floor
- **Incremental Enhancement**: Systematic addition of behavioral and textual features
- **Algorithm Investigation**: Comparison of linear vs. ensemble approaches
- **Performance Analysis**: Comprehensive evaluation using multiple metrics

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

4. **Download dataset for experimentation**
   - Obtain the CEAS-08 dataset
   - Place `CEAS_08.csv` in the project root directory

### Dataset Requirements
- Place the `CEAS_08.csv` file in the project root directory
- Ensure the dataset contains all required columns
- Total dataset size: ~82,500 emails with balanced class distribution

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

## 🔬 Research Methodology

This project follows a systematic experimental approach to explore ML applications in cybersecurity:

### Experimental Design
- **Baseline Model**: Establish performance floor using basic features
- **Feature Engineering**: Iteratively add behavioral and textual features
- **Algorithm Comparison**: Test different ML approaches on the same feature sets
- **Performance Evaluation**: Use comprehensive metrics to assess detection effectiveness

### Version Evolution Strategy

### Experiment 1.0 - Baseline Investigation
- **Hypothesis**: Basic behavioral features can distinguish phishing emails
- **Features**: Email length, capitalization patterns, keyword presence
- **Result**: 66.1% accuracy - validates need for advanced techniques
- **Learning**: Simple features alone insufficient for reliable detection

### Experiment 1.1 - Domain Reputation Analysis
- **Hypothesis**: Sender domain patterns can improve detection
- **Enhancement**: Added domain frequency analysis + TF-IDF vectorization
- **Result**: 99.40% accuracy - dramatic improvement validates hypothesis
- **Learning**: Domain reputation is a powerful phishing indicator

### Experiment 1.2 - URL Behavioral Analysis (Best Performing)
- **Hypothesis**: URL patterns provide additional discriminative power
- **Enhancement**: Added URL count and presence detection
- **Result**: **99.45% accuracy** - marginal but consistent improvement
- **Learning**: URL analysis complements domain-based detection

### Experiment 2.0 - Algorithm Comparison
- **Hypothesis**: Ensemble methods may capture complex feature interactions
- **Change**: Random Forest instead of Logistic Regression
- **Result**: 99.20% accuracy - slight decrease from linear model
- **Learning**: Linear models can be as effective with proper feature engineering

## 📈 Experimental Results

### Performance Investigation Summary

| Experiment | Model | Accuracy | F1-Score | Key Findings |
|------------|-------|----------|----------|--------------|
| 1.0 | Logistic Regression | 66.10% | 66.00% | Baseline validation |
| 1.1 | Logistic Regression | 99.40% | 99.46% | Domain analysis breakthrough |
| 1.2 | Logistic Regression | **99.45%** | **99.50%** | URL features add value |
| 2.0 | Random Forest | 99.20% | 99.27% | Linear models competitive |

### Research Insights

**1. Feature Engineering Impact**
- Basic features: 66% accuracy
- + Domain analysis: 99.40% (+33.3 percentage points)
- + URL features: 99.45% (+0.05 percentage points)

**2. Algorithm Comparison**
- Logistic Regression: 99.45% (more interpretable)
- Random Forest: 99.20% (captures interactions but complex)

**3. Key Discovery**
The most significant performance jump occurred when adding domain reputation and TF-IDF features, suggesting that sender credibility and content analysis are the primary discriminators in phishing detection.

### Optimal Model Performance (Experiment 1.2)
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

## 🔧 Technical Implementation

### Feature Engineering Experiments
- **Text Processing**: TF-IDF vectorization exploring unigram and bigram features
- **Behavioral Features**: Email length, capitalization ratios, keyword presence analysis
- **Domain Analysis**: Sender domain frequency and reputation scoring investigation
- **URL Features**: Count and presence pattern analysis

### Experimental Pipeline
1. **Data Preprocessing**: Text normalization and feature extraction
2. **Feature Engineering**: Systematic addition of behavioral indicators  
3. **Model Training**: Comparative analysis of different algorithms
4. **Performance Evaluation**: Multi-metric assessment framework

### Technical Considerations
- **Memory Efficiency**: Sparse matrix operations for large-scale text processing
- **Reproducibility**: Fixed random seeds and consistent train-test splits
- **Modularity**: Reusable components for experiment iteration
- **Evaluation**: Comprehensive metrics including precision, recall, and F1-score

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- CEAS-08 Dataset contributors for providing the phishing email research dataset
- scikit-learn community for comprehensive machine learning tools
- Cybersecurity research community for foundational phishing detection research

---

⭐ **Star this repository if you found this exploration helpful!**

*This project demonstrates the application of machine learning techniques to cybersecurity challenges, showcasing systematic experimentation and the importance of feature engineering in achieving effective phishing detection.*
