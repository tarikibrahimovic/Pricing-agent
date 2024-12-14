from flask import Flask, request, jsonify
from stable_baselines3 import DQN
import numpy as np

# Inicijalizacija Flask aplikacije
app = Flask(__name__)

# Učitaj trenirani RL model
model = DQN.load("pricing_agent")


# Funkcija za pripremu stanja
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


@app.route("/predict-price", methods=["POST"])
def predict_price():
    data = request.json

    # Priprema ulaznih podataka
    user_data = data["user_data"]
    state = prepare_state(user_data)

    # Predikcija pomoću RL modela
    action, _ = model.predict(state)

    # Mapiranje akcije na promenu cene
    price_change = {0: -0.10, 1: -0.05, 2: 0.0, 3: 0.05, 4: 0.10}[int(action)]
    new_price = user_data["base_price"] * (1 + price_change)

    return jsonify(
        {"personalized_price": round(new_price, 2), "price_change": price_change * 100}
    )


if __name__ == "__main__":
    app.run(port=5000, debug=True)
