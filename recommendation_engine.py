import random
from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()


class EpsilonGreedyRecommender:
    def __init__(self, supabase_url, supabase_key, epsilon=0.1):
        """
        :param supabase_url: URL Supabase projekta
        :param supabase_key: API ključ Supabase projekta (Service Role Key)
        :param epsilon: Verovatnoća za nasumično biranje kraka (exploration)
        """
        self.epsilon = epsilon
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.items = {}
        self.counts = {}
        self.rewards = {}
        self.load_data()

    def load_data(self):
        """
        Učitaj podatke o patikama iz Supabase baze.
        """
        response = self.supabase.table("Product").select("*").execute()

        # Access data directly from response.data
        data = response.data

        for entry in data:
            sneaker_id = entry["id"]
            name = entry["name"]
            self.items[sneaker_id] = name
            self.counts[sneaker_id] = entry.get("count", 0)
            self.rewards[sneaker_id] = entry.get("reward", 0)

    def save_data(self, sneaker_id):
        """
        Sačuvaj trenutne podatke o patikama u Supabase bazu.
        :param sneaker_id: ID patike koja je ažurirana
        """
        data = {"count": self.counts[sneaker_id], "reward": self.rewards[sneaker_id]}
        print(
            f"Saving data for sneaker_id {sneaker_id}: {data}, Type of sneaker_id: {type(sneaker_id)}"
        )  # Debug statement

        try:
            response = (
                self.supabase.table("Product")
                .update(data)
                .eq("id", sneaker_id)
                .execute()
            )

            print("Update successful:", response.data)

        except Exception as e:
            print(f"Error updating product: {str(e)}")

    def select_item(self):
        """
        Vrati jedan item (patiku) koji treba da preporučiš korisniku.
        Radi se epsilon-greedy selekcija.
        """
        # Slučaj kada biramo nasumično (exploration)
        if random.random() < self.epsilon:
            chosen_id = random.choice(list(self.items.keys()))
            return chosen_id, self.items[chosen_id]

        # Slučaj kada biramo najbolju patiku do sada (exploitation)
        best_item_ids = []
        best_avg_reward = -float("inf")

        # Shuffle the items to ensure random selection among equals
        items_shuffled = list(self.items.keys())
        random.shuffle(items_shuffled)

        for sneaker_id in items_shuffled:
            if self.counts[sneaker_id] == 0:
                avg_reward = (
                    0  # Ako nikad nismo prikazali taj item, postavimo avg_reward na 0
                )
            else:
                avg_reward = self.rewards[sneaker_id] / float(self.counts[sneaker_id])

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_item_ids = [sneaker_id]
            elif avg_reward == best_avg_reward:
                best_item_ids.append(sneaker_id)

        # Ako postoji više proizvoda sa istim najboljim avg_reward, nasumično odaberi jedan
        if best_item_ids:
            chosen_id = random.choice(best_item_ids)
            return chosen_id, self.items[chosen_id]
        else:
            # Fallback na nasumičan izbor ako nema najboljih proizvoda
            chosen_id = random.choice(list(self.items.keys()))
            return chosen_id, self.items[chosen_id]

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
        Ažuriraj broj pokušaja i ostvarenih reward-a za dati item, i sačuvaj podatke u Supabase bazi.
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
        print(reward)
        self.counts[chosen_id] += 1
        self.rewards[chosen_id] += reward
        self.save_data(chosen_id)
