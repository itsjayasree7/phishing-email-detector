# 📧 Phishing Email Detection Model

A machine learning model built with Scikit-learn that classifies emails as **Phishing** or **Safe** based on textual content and URL features.

---

## 📋 Features

- **Dataset** — Trained on 50 labeled emails (25 phishing, 25 legitimate)
- **Feature Extraction** — Detects URLs, urgent keywords, prize/money language, caps ratio, and more
- **ML Pipeline** — TF-IDF vectorizer + Logistic Regression
- **Evaluation** — Displays accuracy score, classification report, and confusion matrix
- **Live Predictions** — Classifies new emails with confidence score

---

## 🛠️ Tech Stack

- Python 3.x
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

---

## ⚙️ Installation

```bash
pip install scikit-learn pandas numpy matplotlib
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/itsjayasree7/phishing-email-detector.git
cd phishing-email-detector
```

**2. Run the model**
```bash
python phishing_email_detector.py
```

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | **90.00%** |
| Precision (avg) | 0.92 |
| Recall (avg) | 0.90 |
| F1-Score (avg) | 0.90 |

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

---

## 🔍 How It Works

1. **Feature Extraction** — Each email is analyzed for:
   - Presence and count of URLs
   - Urgent/warning language (`urgent`, `verify`, `alert`)
   - Prize/reward language (`won`, `prize`, `congratulations`)
   - Financial keywords (`$`, `refund`, `payment`)
   - Text length, exclamation marks, capitalization ratio

2. **TF-IDF Vectorization** — Converts email text into numerical features using bigrams

3. **Logistic Regression** — Trained classifier predicts Phishing (1) or Safe (0)

4. **Confidence Score** — Each prediction includes a probability confidence %

---

## 📁 Project Structure

```
phishing-email-detector/
│
├── phishing_email_detector.py   # Main model script
├── confusion_matrix.png         # Auto-generated confusion matrix
└── README.md                    # Project documentation
```

---

## 🔐 Ethical Usage

This tool is intended for **educational purposes only**.  
Do not use it to bypass email security systems or for any malicious purpose.

---
