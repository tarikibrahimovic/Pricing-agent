import pytest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommendation_engine import EpsilonGreedyRecommender


@pytest.fixture
def mock_session():
    return MagicMock()


@pytest.fixture
def recommender(mock_session):
    recommender = EpsilonGreedyRecommender(mock_session)
    # Setup test data
    recommender.items = {"1": "Product1", "2": "Product2", "3": "Product3"}
    recommender.counts = {"1": 10, "2": 5, "3": 2}
    recommender.rewards = {"1": 5, "2": 10, "3": 1}
    return recommender


@patch("random.random")
@patch("random.choice")
def test_select_item_exploration(mock_choice, mock_random, recommender):
    # Test exploration (epsilon case)
    mock_random.return_value = 0.05  # Less than epsilon (0.1)
    mock_choice.return_value = "2"

    item_id, item_name = recommender.select_item()

    assert item_id == "2"
    assert item_name == "Product2"
    mock_choice.assert_called_once()


@patch("random.random")
def test_select_item_exploitation(mock_random, recommender):
    # Test exploitation (best reward case)
    mock_random.return_value = 0.9  # Greater than epsilon

    item_id, item_name = recommender.select_item()

    # Should select item "2" as it has highest avg reward (10/5 = 2)
    assert item_id == "2"
    assert item_name == "Product2"


def test_select_items(recommender):
    num_items = 2
    with patch.object(recommender, "select_item") as mock_select:
        mock_select.side_effect = [("1", "Product1"), ("2", "Product2")]

        items = recommender.select_items(num_items)

        assert len(items) == 2
        assert items[0] == ("1", "Product1")
        assert items[1] == ("2", "Product2")
        assert mock_select.call_count == 2
