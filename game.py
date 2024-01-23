from deck import Deck
import random

class Game:

    def __init__(self,num_players):
        self.deck = Deck()
        self.num_players = num_players
        #Shuffle the cards
        random.shuffle(self.deck.cards)

    # Function to deal one card to each player
    def deal_single_cards(self):
        
        # 2D list to hold all players' hands
        players_hand=[]

        # Iterate through the players and create a list to hold each player's hand
        for y in range(self.num_players):
                # remove a card from the deck and add it to the player's hand
                list_hand = [self.deck.cards.pop()]  # Ensures players_hand is a list containing one string
                players_hand.append(list_hand)
                
        return players_hand

    # Function that calls the "deal_sigle_card" function and repeats the dealing for 7 rounds
    def deal_all_cards(self, rounds):
        #For each iteration, create an empty list for the hand of each player
        hands = [[] for _ in range(self.num_players)]

        # Deal cards for the specified number of rounds
        for round_number in range(rounds):
            round_hands = self.deal_single_cards()
            for i in range(self.num_players):
                hands[i].extend(round_hands[i])

            # Print the hands after each round
            print(f"Round {round_number + 1}:")
            for i, hand in enumerate(hands):
                hand_str = [str(card) for card in hand]  # Convert each card to a string
                print(f"  Player {i + 1}'s Hand: {hand_str}")

            print()  # Blank line for better readability

    # function to flip first card
    def flip_first_card(self):
         while True:
            # flip the first card
            first_card = self.deck.cards[0]
            # if it's action or wid card, flip the next one
            if first_card.color == "wild" or first_card.value in ["reverse", "draw", "skip"]:
                print(f"Illegal move, flip again: {first_card}")
                self.deck.cards.pop(0)
            else:
                print(f"Flipping first card: {first_card}")
                break