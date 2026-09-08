from functools import wraps
from hmac import compare_digest
from io import BytesIO
from pathlib import Path
from secrets import token_hex
from uuid import uuid4

from flask import Flask, render_template, request, redirect, url_for, session
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
from dotenv import load_dotenv


# Global variables
latest_prediction = None
latest_image_path = None
latest_patient_id = "Unknown"
rendezvous_list = []  # Stockage temporaire des RDV
rapports = []  # 🗂️ Stockage temporaire des rapports médicaux
latest_confirmation = None  # Ajouté pour le retour patient





BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

app.secret_key = os.environ.get("SECRET_KEY") or token_hex(32)

PATIENT_USER_ID = os.environ.get("PATIENT_USER_ID")
PATIENT_PASSWORD = os.environ.get("PATIENT_PASSWORD")
DOCTOR_USER_ID = os.environ.get("DOCTOR_USER_ID")
DOCTOR_PASSWORD = os.environ.get("DOCTOR_PASSWORD")


def role_required(role):
    def decorator(view):
        @wraps(view)
        def wrapped_view(*args, **kwargs):
            if session.get("user_role") != role:
                return redirect(url_for("login"))
            return view(*args, **kwargs)

        return wrapped_view

    return decorator

# Vérification de validité IRM
def is_valid_mri(image):
    if image.mode not in ["L", "LA", "RGB"]:
        return False
    width, height = image.size
    if width < 100 or height < 100 or width > 512 or height > 512:
        return False
    gray = image.convert("L").resize((128, 128))
    pixels = list(gray.getdata())
    avg_pixel = sum(pixels) / len(pixels)
    if avg_pixel > 240 or avg_pixel < 10:
        return False
    return True

# Définition du modèle CNN
class AlzheimerCNN(nn.Module):
    def __init__(self):
        super(AlzheimerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 30 * 30, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 30 * 30)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Chargement du modèle entraîné
model = AlzheimerCNN()
model_path = BASE_DIR / "model" / "alzheimer_cnn.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))
model.eval()

# Prétraitement image
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

labels_map = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

# Redirection par défaut
@app.route("/")
def home():
    return redirect(url_for("login"))

# Connexion utilisateur
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = request.form.get("user_id")
        password = request.form.get("password")

        if (PATIENT_USER_ID and PATIENT_PASSWORD and
                compare_digest(user_id or "", PATIENT_USER_ID) and
                compare_digest(password or "", PATIENT_PASSWORD)):
            session["user_role"] = "patient"  # ✅ ici c’est bon
            return redirect(url_for("patient"))

        elif (DOCTOR_USER_ID and DOCTOR_PASSWORD and
              compare_digest(user_id or "", DOCTOR_USER_ID) and
              compare_digest(password or "", DOCTOR_PASSWORD)):
            session["user_role"] = "doctor"  # ✅ ici aussi
            return redirect(url_for("doctor"))

        else:
            return render_template("login.html", error="Identifiants invalides")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()  # Vide les données de session (déconnexion)
    return redirect(url_for("login"))



# Espace Patient
@app.route("/patient", methods=["GET", "POST"])
@role_required("patient")
def patient():
    global latest_prediction, latest_image_path, latest_patient_id, latest_confirmation
    if request.method == "POST" and "image" in request.files:
        file = request.files["image"]
        if file and file.filename:
            try:
                image_bytes = file.read()
                image = Image.open(BytesIO(image_bytes))
                image.load()
            except (OSError, ValueError, Image.DecompressionBombError):
                return render_template("patient.html", error="Fichier image invalide.", latest_confirmation=latest_confirmation)

            if not is_valid_mri(image):
                return render_template("patient.html", error="Image invalide. Veuillez téléverser une IRM cérébrale correcte.", latest_confirmation=latest_confirmation)

            filename = f"{uuid4().hex}.png"
            image_path = UPLOAD_FOLDER / filename
            image.save(image_path, format="PNG")
            latest_patient_id = "patient"
            image = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
                prediction = labels_map[predicted.item()]
                latest_prediction = prediction
                latest_image_path = url_for("static", filename=f"uploads/{filename}")
            return render_template("resultat.html", prediction=prediction, image_path=latest_image_path)
    
    return render_template("patient.html", latest_confirmation=latest_confirmation)

# Formulaire de rendez-vous
@app.route("/rendezvous", methods=["POST"])
@role_required("patient")
def rendezvous():
    global rendezvous_list
    nom = request.form.get("nom")
    telephone = request.form.get("telephone")
    date = request.form.get("date")
    heure = request.form.get("heure")
    notes = request.form.get("notes")
    rdv = {
        "nom": nom,
        "telephone": telephone,
        "date": date,
        "heure": heure,
        "notes": notes
    }
    rendezvous_list.append(rdv)
    message = f"Rendez-vous enregistré pour le {date} à {heure}."
    return render_template("patient.html", rendezvous_message=message)

# Vue Docteur
@app.route("/doctor", methods=["GET", "POST"])
@role_required("doctor")
def doctor():
    global latest_prediction, latest_image_path, latest_patient_id, rendezvous_list, latest_confirmation
    if request.method == "POST":
        confirmation = request.form.get("confirmation")
        latest_confirmation = confirmation  # sauvegarde la confirmation
        return render_template("confirmation.html", patient_id=latest_patient_id, confirmation=confirmation)
    return render_template("doc.html", prediction=latest_prediction, image_path=latest_image_path, patient_id=latest_patient_id, rdvs=rendezvous_list, rapports=rapports)



@app.route("/rapport", methods=["POST"])
@role_required("doctor")
def rapport():
    global rapports
    contenu = request.form.get("rapport")
    if contenu:
        rapports.append(contenu)
    return redirect(url_for("doctor"))


# Lancement de l'application
if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
