import random
import json
import os


class EpsilonGreedyRecommender:
    def __init__(self, filepath, epsilon=0.1):
        """
        :param filepath: Putanja do JSON fajla sa podacima o patikama
        :param epsilon: Verovatnoća za nasumično biranje kraka (exploration)
        """
        self.filepath = filepath
        self.epsilon = epsilon
        self.items = {}
        self.counts = {}
        self.rewards = {}
        self.load_data()

    def load_data(self):
        """
        Učitaj podatke o patikama iz JSON fajla.
        Ako fajl ne postoji, inicijalizuj prazan skup patika.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Fajl {self.filepath} ne postoji.")

        with open(self.filepath, "r") as f:
            data = json.load(f)
            for entry in data:
                sneaker_id = entry["id"]
                name = entry["name"]
                self.items[sneaker_id] = name
                self.counts[sneaker_id] = entry.get("count", 0)
                self.rewards[sneaker_id] = entry.get("reward", 0)

    def save_data(self):
        """
        Sačuvaj trenutne podatke o patikama u JSON fajl.
        """
        data = []
        for sneaker_id, name in self.items.items():
            data.append(
                {
                    "id": sneaker_id,
                    "name": name,
                    "count": self.counts.get(sneaker_id, 0),
                    "reward": self.rewards.get(sneaker_id, 0),
                }
            )

        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=4)

    def select_item(self):
        """
        Vrati item (patiku) koji treba da preporučiš korisniku.
        Radi se epsilon-greedy selekcija.
        """
        if random.random() < self.epsilon:
            chosen_id = random.choice(list(self.items.keys()))
            return chosen_id, self.items[chosen_id]

        best_item_id = None
        best_avg_reward = -float("inf")

        for sneaker_id in self.items:
            if self.counts[sneaker_id] == 0:
                avg_reward = 0
            else:
                avg_reward = self.rewards[sneaker_id] / float(self.counts[sneaker_id])

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_item_id = sneaker_id

        return best_item_id, self.items[best_item_id]

    def select_items(self, num_items=1):
        """
        Vrati listu preporučenih patika.
        :param num_items: Broj patika za preporuku
        :return: Lista tuple-ova (id, name)
        """
        recommendations = []
        selected_ids = set()

        for _ in range(num_items):
            chosen_id, chosen_name = self.select_item()
            # Ako je već odabrana u ovoj grupi, ponovi izbor
            while chosen_id in selected_ids and len(selected_ids) < len(self.items):
                chosen_id, chosen_name = self.select_item()

            if chosen_id not in selected_ids:
                recommendations.append((chosen_id, chosen_name))
                selected_ids.add(chosen_id)
            else:
                break  # Nema više jedinstvenih patika za preporuku

        return recommendations

    def update(self, chosen_id, interaction_type):
        """
        Ažuriraj broj pokušaja i ostvarenih reward-a za dati item, i sačuvaj podatke u fajl.
        :param chosen_id: ID patike koja je bila preporučena
        :param interaction_type: Tip interakcije ('click', 'purchase', 'no_click')
        """
        if chosen_id not in self.items:
            raise ValueError(f"Odabrana patika sa ID '{chosen_id}' nije poznata.")

        reward_mapping = {
            "click": 1,
            "purchase": 5,
            "no_click": -1,
        }

        reward = reward_mapping.get(interaction_type, 0)

        self.counts[chosen_id] += 1
        self.rewards[chosen_id] += reward
        self.save_data()
