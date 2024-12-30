import os
from flask import Flask, request, jsonify
from stable_baselines3 import DQN
import numpy as np
from dotenv import load_dotenv
from recommendation_engine import EpsilonGreedyRecommender

app = Flask(__name__)

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception(
        "SUPABASE_URL i SUPABASE_KEY environment varijable moraju biti postavljene."
    )


model = DQN.load("pricing_agent")
recommender = EpsilonGreedyRecommender(
    supabase_url=SUPABASE_URL, supabase_key=SUPABASE_KEY, epsilon=0.1
)


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
    """
    Endpoint za preporuke patika.
    Može da primi query parametar 'num' koji određuje broj preporuka.
    """
    try:
        num = int(request.args.get("num", 1))
        if num < 1:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Parametar 'num' mora biti pozitivan ceo broj."}), 400

    # Odaberi patike za preporuku
    chosen_items = recommender.select_items(num_items=num)

    if not chosen_items:
        return jsonify({"error": "Nema dostupnih patika za preporuku."}), 404

    # Povuci dodatne podatke o preporučenim patikama iz Supabase
    try:
        sneaker_ids = [str(item[0]) for item in chosen_items]  # Convert IDs to strings
        response = (
            recommender.supabase.table("Product")
            .select("*")
            .in_("id", sneaker_ids)
            .execute()
        )

        # Access data directly from response
        sneakers_data = response.data

        if not sneakers_data:
            return jsonify({"error": "Nije moguće dobiti podatke o patikama."}), 404

        return jsonify({"recommendations": sneakers_data}), 200

    except Exception as e:
        return jsonify({"error": f"Greška prilikom dobijanja podataka: {str(e)}"}), 500


@app.route("/interact", methods=["POST"])
def interact():
    """
    Endpoint za interakcije korisnika sa preporukama.
    Očekuje JSON telo sa 'id' i 'interaction_type'.
    """
    data = request.json
    chosen_id = data.get("id")
    interaction_type = data.get(
        "interaction_type", "no_click"
    )  # 'click', 'purchase', 'no_click'

    if not chosen_id:
        return jsonify({"error": "Parametar 'id' je obavezan."}), 400

    if interaction_type not in ["click", "purchase", "no_click"]:
        return jsonify({"error": "Nepoznata vrednost za 'interaction_type'."}), 400

    try:
        recommender.update(chosen_id, interaction_type)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Greška prilikom ažuriranja podataka."}), 500

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
