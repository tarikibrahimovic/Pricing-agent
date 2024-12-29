from flask import Flask, request, jsonify
from stable_baselines3 import DQN
import numpy as np
from recommendation_engine import EpsilonGreedyRecommender

app = Flask(__name__)

model = DQN.load("pricing_agent")
recommender = EpsilonGreedyRecommender(filepath="sneakers.json", epsilon=0.1)


def prepare_state(user_data):
    state = np.array(
        [
            (user_data["age"] - 18) / (70 - 18),
            user_data["average_spent"] / 1000,
            user_data["price_sensitivity"],
            user_data["base_price"] / 1000,
            user_data["margin"],
        ],
        dtype=np.float32,
    )
    return state


@app.route("/recommend", methods=["GET"])
def recommend():
    # Dobavi broj preporuka iz query parametra, podrazumevano je 1
    try:
        num = int(request.args.get("num", 1))
        if num < 1:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Parametar 'num' mora biti pozitivan ceo broj."}), 400

    # Odaberi patike za preporuku
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
    state = prepare_state(user_data)

    action, _ = model.predict(state)

    price_change = {0: -0.10, 1: -0.05, 2: 0.0, 3: 0.05, 4: 0.10}[int(action)]
    new_price = user_data["base_price"] * (1 + price_change)

    return jsonify(
        {"personalized_price": round(new_price, 2), "price_change": price_change * 100}
    )


if __name__ == "__main__":
    app.run(port=5000, debug=True)
