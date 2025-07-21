# Fraud Detection System for E-commerce and Banking Transactions

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Advanced fraud detection system using XGBoost and SHAP explainability for financial transactions.

## Features

- Real-time fraud scoring (15ms latency)
- Geolocation intelligence
- Transaction pattern recognition
- SHAP explainable AI
- Automated model monitoring

## Project Structure
fraud-detection-fintech/
├── data/ # Data directories (see setup)
├── models/ # Trained model binaries
├── results/ # Evaluation metrics and SHAP visuals
├── src/ # Source code
│ ├── preprocessing/ # Data cleaning scripts
│ ├── training/ # Model development
│ └── explainability/ # SHAP analysis
├── notebooks/ # Exploratory analysis
├── tests/ # Unit tests
├── requirements.txt # Dependencies
└── README.md # This file


## Setup & Installation

1. **Clone repository:**
   ```bash
   git clone https://github.com/David-De-Mozart/fraud-detection-fintech.git
   cd fraud-detection-fintech

2. Install dependencies:

    pip install -r requirements.txt

3. Run pipeline:

    # Preprocess data
    python src/preprocessing/ecommerce_preprocessing.py
    python src/preprocessing/creditcard_preprocessing.py

    # Balance datasets
    python src/balance_data.py

    # Train models
    python src/model_training.py

    # Generate SHAP explanations
    python src/explainability.py

License
This project is licensed under the MIT License - see LICENSE file for details.