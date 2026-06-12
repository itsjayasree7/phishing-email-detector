"""
Phishing Email Detection Model
Uses Scikit-learn to classify emails as Phishing or Safe.

Features:
  - Train on a dataset of phishing and legitimate emails
  - Extract and analyze email features (URLs, keywords, text patterns)
  - Classify emails as "Phishing" or "Safe"
  - Display accuracy and confusion matrix
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
import re
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. SAMPLE DATASET
# ─────────────────────────────────────────────

PHISHING_EMAILS = [
    "URGENT: Your account has been suspended. Click here to verify: http://secure-login.fakebank.com",
    "Congratulations! You've won $1,000,000. Claim your prize now at http://prizes.scam.net",
    "Your PayPal account is limited. Update your information immediately: http://paypal.verify.xyz",
    "Dear customer, your bank account will be closed. Login now: http://bank-update.phish.com",
    "Action required: Verify your Amazon account to avoid suspension http://amazon-verify.fake.com",
    "You have a pending payment of $500. Confirm your details: http://payment.fraud.net",
    "Your Apple ID has been locked. Unlock it now: http://appleid.scamsite.com/unlock",
    "WARNING: Unusual activity detected on your account. Click to secure: http://security.fake.net",
    "Free iPhone giveaway! You are selected. Click: http://freeiphone.scam.org",
    "Your Netflix subscription has expired. Renew now: http://netflix-renew.phish.co",
    "Verify your email to win a gift card. Click: http://win-gifts.fraud.com",
    "IRS Tax refund pending. Claim at: http://irs-refund.fake.gov.com",
    "Your password expires today! Reset here: http://password-reset.phish.net",
    "Urgent security alert: Login from unknown device. Verify: http://secure.fakelogin.com",
    "You have been selected for a survey. Win $500: http://survey-rewards.scam.com",
    "Your credit card has been charged $999. Dispute here: http://dispute.fraud.net",
    "DHL delivery failed. Update address: http://dhl-delivery.fake.com/update",
    "Microsoft account breach detected. Secure now: http://microsoft.phish.org",
    "FINAL WARNING: Your account will be deleted. Act now: http://account-save.fake.com",
    "Lucky winner! Claim your $10,000 prize: http://lottery.scam.net/claim",
    "Verify your identity to receive your inheritance: http://inheritance.fraud.com",
    "Your Google account needs verification: http://google-verify.phish.net",
    "Suspicious login attempt on your account. Confirm: http://login-confirm.fake.com",
    "Your subscription will auto-renew for $299. Cancel: http://cancel-now.scam.net",
    "Bank transfer of $2000 initiated. Not you? Click: http://bank.phish.org/cancel",
]

SAFE_EMAILS = [
    "Hi team, please find attached the meeting notes from yesterday's standup.",
    "Your order #12345 has been shipped and will arrive by Friday.",
    "Reminder: Project deadline is next Monday. Please submit your reports.",
    "Thank you for attending our webinar. Here is the recording link from our official site.",
    "Your monthly bank statement is now available in your secure online portal.",
    "Welcome to our newsletter! Here are this week's top tech articles.",
    "Team lunch is scheduled for Thursday at 1 PM in the conference room.",
    "Your appointment with Dr. Smith is confirmed for June 15 at 10 AM.",
    "Please review and sign the attached document at your earliest convenience.",
    "Happy birthday! Wishing you a wonderful day from the whole team.",
    "Your flight booking confirmation number is ABC123. Check-in opens 24 hours before.",
    "Quarterly performance review is scheduled. Please prepare your self-assessment.",
    "The new software update is available. Please update at your convenience.",
    "Thank you for your purchase. Your receipt is attached to this email.",
    "Weekly report: Sales increased by 12% compared to last quarter.",
    "Office will be closed on Monday for the public holiday.",
    "Your library books are due next week. Please return or renew them.",
    "Congratulations on your work anniversary! Five great years with the company.",
    "The team building event is confirmed for Saturday. Details are attached.",
    "Your password was successfully changed. If this was not you, contact support.",
    "Please join the Zoom meeting at 3 PM using the link on the calendar invite.",
    "Your tax documents are ready for download from the official HR portal.",
    "Feedback requested: Please complete the employee satisfaction survey.",
    "New policy update: Work from home guidelines have been revised. See attachment.",
    "Your insurance renewal is due next month. No action needed — auto-renews.",
]


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def extract_features(text):
    """Extract URL and keyword-based features from email text."""
    features = {}
    features["has_url"]          = int(bool(re.search(r'http[s]?://', text)))
    features["url_count"]        = len(re.findall(r'http[s]?://', text))
    features["has_ip_url"]       = int(bool(re.search(r'http[s]?://\d+\.\d+', text)))
    features["has_urgent"]       = int(bool(re.search(r'\b(urgent|immediately|warning|alert|final)\b', text, re.I)))
    features["has_prize"]        = int(bool(re.search(r'\b(won|winner|prize|reward|congratulations|lucky)\b', text, re.I)))
    features["has_account"]      = int(bool(re.search(r'\b(account|suspended|verify|login|password|confirm)\b', text, re.I)))
    features["has_money"]        = int(bool(re.search(r'\$\d+|\b(free|cash|payment|refund|transfer)\b', text, re.I)))
    features["text_length"]      = len(text)
    features["exclamation_count"]= text.count('!')
    features["caps_ratio"]       = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    return features


# ─────────────────────────────────────────────
# 3. BUILD DATASET
# ─────────────────────────────────────────────

def build_dataset():
    data = []
    for email in PHISHING_EMAILS:
        row = extract_features(email)
        row["text"]  = email
        row["label"] = 1  # Phishing
        data.append(row)
    for email in SAFE_EMAILS:
        row = extract_features(email)
        row["text"]  = email
        row["label"] = 0  # Safe
        data.append(row)
    df = pd.DataFrame(data)
    return df


# ─────────────────────────────────────────────
# 4. TRAIN MODEL
# ─────────────────────────────────────────────

def train_model(df):
    X_text  = df["text"]
    y       = df["label"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF + Logistic Regression pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=500,
            stop_words="english"
        )),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])

    pipeline.fit(X_train_text, y_train)
    y_pred = pipeline.predict(X_test_text)

    return pipeline, X_test_text, y_test, y_pred


# ─────────────────────────────────────────────
# 5. DISPLAY RESULTS
# ─────────────────────────────────────────────

def display_results(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print("         PHISHING EMAIL DETECTION — RESULTS")
    print("=" * 55)
    print(f"\n  ✅ Accuracy : {acc * 100:.2f}%")
    print("\n  Classification Report:")
    print("-" * 55)
    report = classification_report(y_test, y_pred, target_names=["Safe", "Phishing"])
    for line in report.splitlines():
        print(f"  {line}")
    print("=" * 55)


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Phishing"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Phishing Email Detection", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\n  [✔] Confusion matrix saved as: confusion_matrix.png")
    plt.close()


# ─────────────────────────────────────────────
# 6. PREDICT NEW EMAILS
# ─────────────────────────────────────────────

def predict_emails(model, emails):
    print("\n" + "=" * 55)
    print("         LIVE EMAIL PREDICTIONS")
    print("=" * 55)
    predictions = model.predict(emails)
    probabilities = model.predict_proba(emails)
    for email, pred, prob in zip(emails, predictions, probabilities):
        label      = "🚨 PHISHING" if pred == 1 else "✅ SAFE"
        confidence = max(prob) * 100
        short      = email[:65] + "..." if len(email) > 65 else email
        print(f"\n  Email   : {short}")
        print(f"  Result  : {label}  (Confidence: {confidence:.1f}%)")
    print("\n" + "=" * 55)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════════════╗")
    print("║    Phishing Email Detection Model        ║")
    print("╚══════════════════════════════════════════╝")

    print("\n[*] Building dataset ...")
    df = build_dataset()
    print(f"    Total emails : {len(df)}  |  Phishing: {df['label'].sum()}  |  Safe: {(df['label']==0).sum()}")

    print("[*] Training model ...")
    model, X_test, y_test, y_pred = train_model(df)
    print("    Training complete.")

    display_results(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)

    # Test on new unseen emails
    test_emails = [
        "URGENT: Your account has been locked! Verify now: http://secure.fakebank.xyz/login",
        "Hi Sarah, just a reminder about our team lunch tomorrow at noon.",
        "You have won a $500 gift card! Claim it here: http://gifts.scam.net/claim",
        "Your monthly invoice is attached. Please process by end of week.",
    ]
    predict_emails(model, test_emails)


if __name__ == "__main__":
    main()
