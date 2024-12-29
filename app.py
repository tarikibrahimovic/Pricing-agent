from flask import Flask, request, jsonify
from stable_baselines3 import DQN
import numpy as np

# Inicijalizacija Flask aplikacije
app = Flask(__name__)

# Učitaj trenirani RL model
model = DQN.load("pricing_agent")

# Funkcija za pripremu stanja za JEDAN proizvod
def prepare_state(user_data, product_data):
    """
    user_data: {
        'age': int,
        'average_spent': float,
        'price_sensitivity': float,
        ...
    }
    product_data: {
        'base_price': float,
        'margin': float,
        ...
    }
    """
    state = np.array(
        [
            (user_data["age"] - 18) / (70 - 18),
            user_data["average_spent"] / 1000,
            user_data["price_sensitivity"],
            product_data["base_price"] / 1000,
            product_data["margin"],
        ],
        dtype=np.float32,
    )
    return state

# Postojeća ruta za PREDIKCIJU CENE za JEDAN proizvod
@app.route("/predict-price", methods=["POST"])
def predict_price():
    data = request.json

    # Priprema ulaznih podataka
    user_data = data["user_data"]
    # Ovde očekujemo da 'product_data' sadrži informacije o jednom proizvodu:
    product_data = {
        "base_price": user_data["base_price"],
        "margin": user_data["margin"],
    }

    # Napravi stanje
    state = prepare_state(user_data, product_data)

    # Predikcija pomoću RL modela
    action, _ = model.predict(state)

    # Mapiranje akcije na promenu cene
    price_change = {0: -0.10, 1: -0.05, 2: 0.0, 3: 0.05, 4: 0.10}[int(action)]
    new_price = user_data["base_price"] * (1 + price_change)

    return jsonify(
        {"personalized_price": round(new_price, 2), "price_change": price_change * 100}
    )

# NOVA ruta za PREDIKCIJU CENA za VIŠE proizvoda
@app.route("/predict-prices-bulk", methods=["POST"])
def predict_prices_bulk():
    """
    Očekivani format JSON:
    {
        "user_data": {
            "age": 30,
            "average_spent": 300.0,
            "price_sensitivity": 0.5
        },
        "products": [
            { "product_id": 1, "base_price": 100.0, "margin": 0.2 },
            { "product_id": 2, "base_price": 50.0, "margin": 0.3 },
            ...
        ]
    }
    """
    data = request.json
    user_data = data["user_data"]
    products_data = data["products"]  # Lista proizvoda

    results = []
    for product in products_data:
        # Pripremi stanje za svaki proizvod
        state = prepare_state(user_data, product)

        # Predikcija pomoću RL modela
        action, _ = model.predict(state)
        price_change = {0: -0.10, 1: -0.05, 2: 0.0, 3: 0.05, 4: 0.10}[int(action)]

        new_price = product["base_price"] * (1 + price_change)

        # Dodaj rezultat za ovaj proizvod
        results.append(
            {
                "product_id": product.get("product_id", None),
                "personalized_price": round(new_price, 2),
                "price_change_percent": price_change * 100,
            }
        )

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
