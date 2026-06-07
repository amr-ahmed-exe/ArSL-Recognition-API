<h1 align="center"> Arabic Sign Language (ArSL) Recognition API</h1>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Model-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**Real-time Arabic Sign Language recognition API powered by Machine Learning and MediaPipe.**

</div>

---

##  Overview

The **ArSL API** is a dedicated backend service designed to recognize and classify **Arabic Sign Language (ArSL)** gestures. Unlike the ASL version which uses PyTorch GCNs, this service utilizes a highly optimized **Random Forest Model** (`arsl_rf_model.joblib`) trained on MediaPipe hand landmarks to predict Arabic letters in real-time.

###  Key Highlights

-  **High Accuracy Classification:** Uses a trained Random Forest Classifier tailored for the complex gestures of the Arabic alphabet.
-  **Real-time Inference:** Built with FastAPI to handle high-frequency incoming coordinate streams.
-  **Label Encoding:** Custom `arsl_label_encoder.joblib` to map model predictions back to Arabic Unicode characters.
-  **Docker Support:** Ready to be deployed instantly to cloud environments via Docker.

---

##  Getting Started

### Prerequisites

- Python 3.10+
- Docker (Optional)

### Run with Docker

```bash
git clone https://github.com/amr-ahmed-exe/ArSL_API.git
cd ArSL_API

docker build -t arsl-api .
docker run -p 8000:8000 arsl-api
```

### Run Locally

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start the API
uvicorn main:app --host 0.0.0.0 --port 8000
```
*The API will be available at `http://localhost:8000`*

---

##  Model Details

- **Model:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)
- **Input:** 21 3D hand landmarks (X, Y, Z coordinates) extracted via MediaPipe
- **Output:** Arabic letter classification (أ-ي)
- **Size:** The model is optimized and serialized into a `.joblib` file for extremely fast loading and inference times.

---

##  License

Copyright © 2026 **Amr Ahmed**. All Rights Reserved.

---

<div align="center">

Made with ❤️ as a graduation project · Suez Canal University 2026

If you find this useful, please consider giving it a star!
