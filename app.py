from flask import Flask, request, jsonify
from stable_baselines3 import DQN
import numpy as np
from recommendation_engine import EpsilonGreedyRecommender

app = Flask(__name__)

model = DQN.load("pricing_agent")
recommender = EpsilonGreedyRecommender(filepath="sneakers.json", epsilon=0.1)


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


@app.route("/recommend", methods=["GET"])
def recommend():
    try:
        num = int(request.args.get("num", 1))
        if num < 1:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Parametar 'num' mora biti pozitivan ceo broj."}), 400

    chosen_items = recommender.select_items(num_items=num)
    recommendations = [{"id": cid, "name": cname} for cid, cname in chosen_items]

    return jsonify({"recommendations": recommendations})


@app.route("/interact", methods=["POST"])
def interact():
    data = request.json
    chosen_id = data.get("id")
    interaction_type = data.get("interaction_type", "no_click")

    if chosen_id not in recommender.items:
        return jsonify({"error": "Nepoznata patika"}), 400

    recommender.update(chosen_id, interaction_type)
    return jsonify({"status": "updated"})


@app.route("/predict-price", methods=["POST"])
def predict_price():
    data = request.json

    user_data = data["user_data"]
    # Ovde očekujemo da 'product_data' sadrži informacije o jednom proizvodu:
    product_data = {
        "base_price": user_data["base_price"],
        "margin": user_data["margin"],
    }

    # Napravi stanje
    state = prepare_state(user_data, product_data)

    action, _ = model.predict(state)

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
