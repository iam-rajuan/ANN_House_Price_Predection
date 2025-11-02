# from flask import Flask,request,render_template
# from tensorflow.keras.models import load_model
# import  pickle
# import numpy as np
#
# # loading models
# app = Flask(__name__)
# model = load_model("model_ann.h5")
# scaler = pickle.load(open('scaler.pkl','rb'))
#
# # creating routes
# @app.route('/')
# def index():
#     return render_template('index.html')
#
# @app.route("/house",methods=['POST','GET'])
# def house():
#     if request.method=='POST':
#        longitude = request.form['longitude']
#        latitude = request.form['latitude']
#        houseage = request.form['houseage']
#        houserooms = request.form['houserooms']
#        totlabedrooms = request.form['totlabedrooms']
#        population = request.form['population']
#        households = request.form['households']
#        medianincome = request.form['medianincome']
#        oceanproximity = request.form['oceanproximity']
#
#        features = np.array([longitude,latitude,houseage,houserooms,totlabedrooms,population,households,
#                             medianincome,oceanproximity], dtype=float)
#
#        features_scaled = scaler.transform([features])
#
#        price = model.predict(features_scaled).reshape(1,-1)
#        return render_template('index.html',result = price)
#
# @app.route("/about")
# def about():
#     return render_template('about.html')
#
# @app.route("/doc")
# def doc():
#     return render_template('doc.html')
#
# if __name__ == "__main__":
#     app.run(debug=True)













import os
import pickle
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import tensorflow as tf

# --- Paths & Flask ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# If your HTML files are in the project root (index.html, about.html, doc.html)
app = Flask(__name__, template_folder='.')   # if you move them into /templates, change to app = Flask(__name__)

# --- Load model & scaler (safe for legacy .h5) ---
MODEL_PATH = os.path.join(BASE_DIR, "model_ann.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Handle legacy 'mse' metric during load + avoid recompiling
model = load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={"mse": tf.keras.metrics.MeanSquaredError()}
)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Optional: map categorical ocean proximity -> numeric if your form sends strings.
# Comment this out if your form already sends numbers.
OCEAN_MAP = {
    "NEAR BAY": 0,
    "NEAR OCEAN": 1,
    "INLAND": 2,
    "<1H OCEAN": 3,
    "ISLAND": 4,
}
def coerce_ocean(val):
    # Try float first; if fails, map string category
    try:
        return float(val)
    except Exception:
        return float(OCEAN_MAP.get(str(val).strip().upper(), 0))

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/house", methods=["POST", "GET"])
def house():
    if request.method == "POST":
        # Read form inputs (strings)
        longitude = request.form["longitude"]
        latitude = request.form["latitude"]
        houseage = request.form["houseage"]
        houserooms = request.form["houserooms"]
        totlabedrooms = request.form["totlabedrooms"]
        population = request.form["population"]
        households = request.form["households"]
        medianincome = request.form["medianincome"]
        oceanproximity = request.form["oceanproximity"]

        # Convert to numeric (with robust ocean proximity handling)
        vals = [
            float(longitude),
            float(latitude),
            float(houseage),
            float(houserooms),
            float(totlabedrooms),
            float(population),
            float(households),
            float(medianincome),
            coerce_ocean(oceanproximity),
        ]

        # Scale & predict
        features_scaled = scaler.transform([vals])
        price = model.predict(features_scaled).reshape(-1).tolist()[0]

        # Show a clean number (adjust formatting as you like)
        return render_template("index.html", result=f"{price:,.2f}")

    # GET -> show empty page
    return render_template("index.html", result=None)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/doc")
def doc():
    return render_template("doc.html")

if __name__ == "__main__":
    app.run(debug=True)
