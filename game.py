from deck import Deck
import random

class Game:

    def __init__(self,num_players):
        self.deck = Deck()
        self.num_players = num_players
        self.players_hand = [[] for _ in range(num_players)]
        #Shuffle the cards
        random.shuffle(self.deck.cards)

    # Function to deal one card to each player
    def deal_single_cards(self):

        # Iterate through the players and add a card to each player's hand
     for player_index in range(self.num_players):
        if len(self.deck.cards) > 0:
            card = self.deck.cards.pop()
            self.players_hand[player_index].append(card)


    # Function that calls the "deal_sigle_card" function and repeats the dealing for 7 rounds
    def deal_all_cards(self):


         # Iterate through rounds
      for round_number in range(1,8):
        self.deal_single_cards()  # Deal one card to each player in order

        # Print the hands after each round
        print(f"Round {round_number}:")
        for i, hand in enumerate(self.players_hand):
            hand_str = [str(card) for card in hand]  # Convert each card to a string
            print(f"  Player {i + 1}'s Hand: {hand_str}")

            print()  # Blank line for better readability

    # function to flip first card
    def flip_first_card(self):
         while True:
            # flip the first card - remove from the deck
            self.first_card = self.deck.cards.pop(0)

            # if it's action or wid card, flip the next one
            if self.first_card.color == "wild" or self.first_card.value in ["reverse", "draw", "skip"]:
                print(f"Illegal move, flip again: {self.first_card}")
                self.deck.cards.pop(0)
            else:
                print(f"Flipping first card: {self.first_card}")
                break

    # Function to play card
    def play_card(self, player_num):
        player_hand = self.players_hand[player_num]

        print(f"Player {player_num + 1}'s Hand:")

        #access and print current player's hand
        for index, card in enumerate(player_hand, start=1):
            card_str = str(card)
            print(f"Card {index}: {card_str}")

        #Player input
        while True: 
            try:
                players_input = int(input(f"Choose your card, Player {player_num + 1} (1-{len(player_hand)}): "))

                if players_input >= 1 and players_input <= len(player_hand):
                    played_card_index = players_input-1

                    # remove the played card from players current hand
                    played_card = player_hand.pop(played_card_index)

                    #check played card is valid
                    if played_card.value == self.first_card.value or played_card.color == self.first_card.color: 
                        print(f"You played: {played_card}")
                        print(f"remaining cards: {len(player_hand)}")
                    else: 
                        print("card needs to match color or number")

                    break
                else:
                    print(f"Please enter a number between: (1-{len(player_hand)}")
            except ValueError:
                    print("Invalid input")

                

        

