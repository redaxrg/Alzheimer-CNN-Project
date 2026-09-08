# Alzheimer CNN Web App

A small Flask interface for running inference with the included PyTorch CNN model. The application is an educational/research prototype and is **not a medical device**. Its predictions must not be used for diagnosis or treatment decisions.

## Requirements

- Python 3.10+
- A CPU or CUDA installation compatible with the selected PyTorch build

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set a unique `SECRET_KEY` plus strong credentials. Do not commit `.env` or uploaded images.

Run the app with:

```powershell
python app.py
```

Open `http://127.0.0.1:5000` in a browser. For a deployment, use a production WSGI server and HTTPS. Do not expose Flask's development server directly to the internet.

## Project layout

- `app.py`: Flask routes, image preprocessing, and inference
- `model/alzheimer_cnn.pth`: trained model weights
- `templates/`: Jinja HTML templates
- `static/uploads/`: runtime uploads; ignored by Git

## Training dataset

The model was trained using the [Alzheimer_MRI dataset](https://huggingface.co/datasets/Falah/Alzheimer_MRI) published by Falah.G.Salieh on Hugging Face. The dataset README identifies it as Apache-2.0 licensed and reports 5,120 training images and 1,280 test images across these four classes:

1. Mild Demented
2. Moderate Demented
3. Non Demented
4. Very Mild Demented

These class names and this order match `labels_map` in `app.py`. Dataset attribution does not prove that the included weights were trained with the same split, preprocessing, or hyperparameters. Those training details should be added before claiming that the model is reproducible or clinically validated.

Suggested citation:

```text
Falah.G.Salieh. Alzheimer MRI Dataset. Hugging Face, 2023.
https://huggingface.co/datasets/Falah/Alzheimer_MRI
```

## Important limitations before deployment

The current prototype keeps appointments, reports, and the latest prediction in process memory. They are lost on restart and are not isolated between users or workers. A real deployment needs authenticated per-patient records in a protected database, CSRF protection, audit logging, access controls, retention rules, and a privacy/security review before handling real patient data.
