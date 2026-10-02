from flask import Flask, render_template, request
import os
import joblib
import sqlite3
from datetime import datetime

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "spam_model.pkl")
VECTORIZER_PATH = os.path.join("model", "tfidf_vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def init_db():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT,
            result TEXT,
            confidence REAL,
            date_time TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if message:
            features = vectorizer.transform([message])
            prediction = model.predict(features)[0]

            result = "SPAM" if prediction == 1 else "GENUINE"

            if hasattr(model, "predict_proba"):
                confidence = round(
                    max(model.predict_proba(features)[0]) * 100, 2
                )

            conn = sqlite3.connect("history.db")
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO history
                (message, result, confidence, date_time)
                VALUES (?, ?, ?, ?)
            """, (
                message,
                result,
                confidence,
                datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            ))

            conn.commit()
            conn.close()

    return render_template(
        "index.html",
        result=result,
        message=message,
        confidence=confidence
    )


@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("message", "").strip()

    result = None
    confidence = None

    if message:
        features = vectorizer.transform([message])
        prediction = model.predict(features)[0]

        result = "SPAM" if prediction == 1 else "GENUINE"

        if hasattr(model, "predict_proba"):
            confidence = round(
                max(model.predict_proba(features)[0]) * 100, 2
            )

        conn = sqlite3.connect("history.db")
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO history
            (message, result, confidence, date_time)
            VALUES (?, ?, ?, ?)
        """, (
            message,
            result,
            confidence,
            datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        ))

        conn.commit()
        conn.close()

    return render_template(
        "index.html",
        result=result,
        message=message,
        confidence=confidence
    )


@app.route("/history")
def history():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT message, result, confidence, date_time
        FROM history
        ORDER BY id DESC
    """)

    records = cursor.fetchall()
    conn.close()

    return render_template(
        "history.html",
        records=records
    )


if __name__ == "__main__":
    app.run(debug=True)