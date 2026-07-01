from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    present_price = float(request.form["Present_Price"])
    kms_driven = int(request.form["Kms_Driven"])
    owner = int(request.form["Owner"])
    car_age = int(request.form["Car_Age"])

    fuel = request.form["Fuel_Type"]
    seller = request.form["Seller_Type"]
    transmission = request.form["Transmission"]

    # Encode Fuel Type
    fuel_dict = {
        "Petrol": 0,
        "Diesel": 1,
        "CNG": 2
    }

    # Encode Seller Type
    seller_dict = {
        "Dealer": 0,
        "Individual": 1
    }

    # Encode Transmission
    transmission_dict = {
        "Manual": 0,
        "Automatic": 1
    }

    fuel = fuel_dict[fuel]
    seller = seller_dict[seller]
    transmission = transmission_dict[transmission]

    prediction = model.predict([[
        present_price,
        kms_driven,
        owner,
        fuel,
        seller,
        transmission,
        car_age
    ]])

    prediction = round(prediction[0], 2)

    return render_template(
        "index.html",
        prediction_text=f"₹ {prediction} Lakhs"
    )


if __name__ == "__main__":
    app.run(debug=True)