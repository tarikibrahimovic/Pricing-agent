import os
from contextlib import contextmanager
from typing import Generator
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from stable_baselines3 import DQN
from database import SessionLocal, init_db
from recommendation_engine import EpsilonGreedyRecommender

import logging
from decimal import Decimal
import traceback
from gym import spaces
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


observation_space = spaces.Box(
    low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
    high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
    dtype=np.float32,
)

action_space = spaces.Discrete(5)  # 5 akcija (0-4) za promjenu cijene

model = DQN.load(
    "pricing_agent",
    custom_objects={
        "observation_space": observation_space,
        "action_space": action_space,
    },
)


@contextmanager
def get_session() -> Generator:
    """Context manager za database session"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def prepare_state(user_data: dict, product_data: dict) -> np.ndarray:
    """Pripremi state za RL model"""
    return np.array(
        [
            (user_data["age"] - 18) / (70 - 18),
            user_data["average_spent"] / 1000,
            user_data["price_sensitivity"],
            product_data["base_price"] / 1000,
            product_data["margin"],
        ],
        dtype=np.float32,
    )


@app.route("/recommend", methods=["GET"])
def recommend():
    try:
        num = int(request.args.get("num", 1))
        if num < 1:
            return jsonify({"error": "Broj preporuka mora biti pozitivan"}), 400

        with get_session() as session:
            recommender = EpsilonGreedyRecommender(session)
            items = recommender.select_items(num_items=num)

            if not items:
                return jsonify({"error": "Nema dostupnih preporuka"}), 404

            products = recommender.get_products_by_ids([int(id) for id, _ in items])

            recommendations = [p.to_dict() for p in products]

            return jsonify({"recommendations": recommendations}), 200

    except Exception as e:
        logging.error(f"Error in recommend route: {str(e)}")
        return jsonify({"error": "Interna serverska greška"}), 500


@app.route("/interact", methods=["POST"])
def interact():
    data = request.json
    if not data:
        return jsonify({"error": "Nedostaju podaci"}), 400

    chosen_id = data.get("id")
    interaction_type = data.get("interaction_type", "no_click")

    if not chosen_id:
        return jsonify({"error": "ID je obavezan"}), 400

    if interaction_type not in ["click", "purchase", "no_click"]:
        return jsonify({"error": "Nevažeći tip interakcije"}), 400

    try:
        with get_session() as session:
            recommender = EpsilonGreedyRecommender(session)
            recommender.update(chosen_id, interaction_type)
            return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": f"Greška pri ažuriranju: {str(e)}"}), 500


@app.route("/predict-price", methods=["POST"])
def predict_price():
    data = request.json
    if not data:
        return jsonify({"error": "Nedostaju podaci"}), 400

    try:
        user_data = data["user_data"]
        product_data = {
            "base_price": user_data["base_price"],
            "margin": user_data["margin"],
        }

        state = prepare_state(user_data, product_data)
        action, _ = model.predict(state)

        price_changes = {0: -0.10, 1: -0.05, 2: 0.0, 3: 0.05, 4: 0.10}
        price_change = price_changes[int(action)]
        new_price = user_data["base_price"] * (1 + price_change)

        return (
            jsonify(
                {
                    "personalized_price": round(new_price, 2),
                    "price_change": price_change * 100,
                }
            ),
            200,
        )

    except KeyError as e:
        return jsonify({"error": f"Nedostaje polje: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Greška: {str(e)}"}), 500


@app.route("/predict-prices-bulk", methods=["POST"])
def predict_prices_bulk():
    data = request.json
    if not data:
        return jsonify({"error": "Nedostaju podaci"}), 400

    try:
        user_data = data["user_data"]
        products_data = data["products"]

        results = []
        for product in products_data:
            state = prepare_state(user_data, product)
            action, _ = model.predict(state)

            price_changes = {0: -0.10, 1: -0.05, 2: 0.0, 3: 0.05, 4: 0.10}
            price_change = price_changes[int(action)]
            new_price = product["base_price"] * (1 + price_change)

            results.append(
                {
                    "product_id": product.get("product_id"),
                    "personalized_price": round(new_price, 2),
                    "price_change_percent": price_change * 100,
                }
            )

        return jsonify({"results": results}), 200

    except KeyError as e:
        return jsonify({"error": f"Nedostaje polje: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Greška: {str(e)}"}), 500


if __name__ == "__main__":
    init_db()  # Inicijalizuj bazu pri pokretanju
    app.run(port=5000, debug=True)
