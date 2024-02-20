from card import Card
from deck import Deck

def play_matching_color(agent_hand, current_color):
    for card in agent_hand:
        if card.color == current_color:
            return card
    return None

def play_matching_number(agent_hand, current_number):
    for card in agent_hand:
        if card.value == current_number:
            return card
    return None

def play_wild_card(agent_hand):
    for card in agent_hand:
        if card.color == "wild":
            return card
    return None

def pick_card_from_deck(deck):
    if len(deck.cards) > 0:
        return deck.cards.pop()
    return None
