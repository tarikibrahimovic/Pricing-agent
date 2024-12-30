import random
from supabase import create_client, Client
from dotenv import load_dotenv

# Učitaj environment varijable iz .env fajla
load_dotenv()


class EpsilonGreedyRecommender:
    def __init__(self, supabase_url, supabase_key, epsilon=0.1):
        """
        :param supabase_url: URL Supabase projekta
        :param supabase_key: API ključ Supabase projekta (Service Role Key)
        :param epsilon: Verovatnoća za nasumično biranje patika (exploration)
        """
        self.epsilon = epsilon
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.items = {}  # Ključ je id, vrednost je name
        self.counts = {}  # Ključ je id, vrednost je count
        self.rewards = {}  # Ključ je id, vrednost je reward
        self.load_data()

    def load_data(self):
        """Load products and rewards data from Supabase"""
        try:
            # Fetch products from Supabase
            response = self.supabase.table("Product").select("*").execute()
            products_data = response.data

            if not products_data:
                raise Exception("No products found in database")

            # Initialize items, rewards and counts
            self.items = {str(item["id"]): item["name"] for item in products_data}

            # Initialize rewards and counts if not already present
            for product_id in self.items.keys():
                if product_id not in self.rewards:
                    self.rewards[product_id] = 1.0  # Initial reward value
                if product_id not in self.counts:
                    self.counts[product_id] = 0  # Initial count

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def save_data(self, sneaker_id, penalty=0.0):
        """
        Sačuvaj trenutne podatke o patikama u Supabase bazu.
        :param sneaker_id: ID patike koja je ažurirana
        """
        data = {
            "count": self.counts[sneaker_id],
            "reward": self.rewards[sneaker_id] - penalty,
        }

        try:
            response = (
                self.supabase.table("Product")
                .update(data)
                .eq("id", sneaker_id)
                .execute()
            )

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

        # Shuffle the items da osiguraš nasumičan izbor među jednakima
        items_shuffled = list(self.items.keys())
        random.shuffle(items_shuffled)

        for sneaker_id in items_shuffled:
            if self.counts[sneaker_id] == 0:
                avg_reward = (
                    0.0  # Ako nikad nismo prikazali taj item, postavimo avg_reward na 0
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
                # Primeni kaznu odmah nakon odabira
                self.rewards[chosen_id] -= 0.1  # Mali penalizator
                self.save_data(chosen_id, penalty=0.1)
            else:
                break  # Nema više jedinstvenih patika za preporuku

        return recommendations

    def update(self, chosen_id, interaction_type):
        """
        Ažuriraj broj pokušaja i ostvarene reward-e za dati item.
        """
        chosen_id_str = str(chosen_id)

        # Initialize count if not exists
        if chosen_id_str not in self.counts:
            self.counts[chosen_id_str] = 0

        # Update count
        self.counts[chosen_id_str] += 1

        # Update rewards
        reward_mapping = {
            "click": 1,
            "purchase": 5,
            "no_click": -1,
        }
        reward = reward_mapping.get(interaction_type, 0)

        if chosen_id_str not in self.rewards:
            self.rewards[chosen_id_str] = 0

        self.rewards[chosen_id_str] += reward

        # Save to database
        self.save_data(chosen_id_str)
