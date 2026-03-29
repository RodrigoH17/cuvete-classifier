# Covete Classifier

This project is responsible for real-time detection, classification, and counting of covetes in a production environment using a Raspberry Pi.

## 📌 Objective

The goal of this system is to automatically identify the type of meat in each covete as it passes in front of a fixed camera, and keep a count of all processed products.

The system operates continuously and is designed for industrial environments, providing reliable and automated product tracking.

---

## 🧠 System Architecture

This project is part of a complete pipeline composed of three components:

* **covete-capture** → image collection in the factory
* **covete-training** → model training
* **covete-classifier** → real-time classification and counting (this project)

---

## ⚙️ Environment Setup

Create a virtual environment:

python -m venv venv
```

Activate (Windows PowerShell):

venv\Scripts\Activate.ps1
```

Install dependencies:

pip install -r requirements.txt
```
---

## 👤 Author

Rodrigo Henriques
