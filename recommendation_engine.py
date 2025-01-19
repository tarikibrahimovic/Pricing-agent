import random
from typing import List, Tuple, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from database import Product
import logging

logger = logging.getLogger(__name__)


class EpsilonGreedyRecommender:
    def __init__(self, db_session: Session, epsilon: float = 0.1):
        self.db_session = db_session
        self.epsilon = epsilon
        self.items = {}
        self.counts = {}
        self.rewards = {}
        self.load_data()

    def load_data(self) -> None:
        """Učitaj proizvode iz baze i inicijalizuj lokalne mape"""
        try:
            products = self.db_session.query(Product).all()

            if not products:
                logger.warning("No products found in database")
                return

            for p in products:
                pid_str = str(p.id)
                self.items[pid_str] = p.name
                self.counts[pid_str] = p.count or 0
                self.rewards[pid_str] = p.reward or 1.0

        except SQLAlchemyError as e:
            logger.error(f"Error loading data: {e}")
            raise

    def save_data(self, sneaker_id: str, penalty: float = 0.0) -> None:
        """Sačuvaj promene u bazi"""
        try:
            product = (
                self.db_session.query(Product)
                .filter(Product.id == int(sneaker_id))
                .one_or_none()
            )

            if product:
                product.count = self.counts[sneaker_id]
                product.reward = self.rewards[sneaker_id] - penalty
                self.db_session.commit()
            else:
                logger.warning(f"Product with id={sneaker_id} not found")

        except SQLAlchemyError as e:
            logger.error(f"Error saving data: {e}")
            self.db_session.rollback()
            raise

    def select_item(self) -> Tuple[str, str]:
        """Izaberi jednu patiku epsilon-greedy metodom"""
        if random.random() < self.epsilon:
            chosen_id = random.choice(list(self.items.keys()))
            return chosen_id, self.items[chosen_id]

        best_items = []
        best_reward = float("-inf")

        shuffled_items = list(self.items.keys())
        random.shuffle(shuffled_items)

        for item_id in shuffled_items:
            count = self.counts[item_id]
            reward = self.rewards[item_id]
            avg_reward = reward / count if count > 0 else 0.0

            if avg_reward > best_reward:
                best_reward = avg_reward
                best_items = [item_id]
            elif avg_reward == best_reward:
                best_items.append(item_id)

        chosen_id = (
            random.choice(best_items) if best_items else random.choice(shuffled_items)
        )
        return chosen_id, self.items[chosen_id]

    def select_items(self, num_items: int = 1) -> List[Tuple[str, str]]:
        """Vrati listu preporučenih proizvoda"""
        recommendations = []
        selected = set()

        while len(recommendations) < num_items and len(selected) < len(self.items):
            item_id, item_name = self.select_item()
            if item_id not in selected:
                recommendations.append((item_id, item_name))
                selected.add(item_id)
                self.rewards[item_id] -= 0.1
                self.save_data(item_id, penalty=0.1)

        return recommendations

    def update(self, chosen_id: str, interaction_type: str) -> None:
        """Ažuriraj statistike za izabrani proizvod"""
        chosen_id = str(chosen_id)

        if chosen_id not in self.counts:
            self.counts[chosen_id] = 0
        if chosen_id not in self.rewards:
            self.rewards[chosen_id] = 0

        self.counts[chosen_id] += 1

        rewards = {"click": 1, "purchase": 5, "no_click": -1}
        reward = rewards.get(interaction_type, 0)
        self.rewards[chosen_id] += reward

        self.save_data(chosen_id)

    def get_products_by_ids(self, product_ids: list):
        try:
            str_product_ids = [str(pid) for pid in product_ids]
            products = (
                self.db_session.query(Product)
                .filter(Product.id.in_(str_product_ids))
                .all()
            )
            return products
        except SQLAlchemyError as e:
            print(f"Error fetching products: {str(e)}")
            return []
