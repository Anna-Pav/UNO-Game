from deck import Deck
from card import Card
import random

class Game:

    def __init__(self,num_players):
        self.deck = Deck()
        self.num_players = num_players
        self.players_hand = [[] for _ in range(num_players)]
        random.shuffle(self.deck.cards)                            # Shuffle the cards
        self.current_color = None                                  # Current color - essential for wild draw functionality
        self.skip_turn = False                                     # Flag to indicate when a player loses their turn

    # Function to deal one card to each player
    def deal_single_cards(self):  
     for player_index in range(self.num_players):                 # Iterate through the players and add a card to each player's hand
        if len(self.deck.cards) > 0:
            card = self.deck.cards.pop()
            self.players_hand[player_index].append(card)

    # Function that calls the "deal_sigle_card" function and repeats the dealing for 7 rounds
    def deal_all_cards(self):   
      for round_number in range(1,8):                             # Iterate through rounds
        self.deal_single_cards()                                  # Deal one card to each player in order
        print(f"Round {round_number}:")                           # Print the hands after each round
        for i, hand in enumerate(self.players_hand):
            hand_str = [str(card) for card in hand]               # Convert each card to a string
            print(f"  Player {i + 1}'s Hand: {hand_str}")
            print()                                               # Blank line for better readability

    # function to flip first card
    def flip_first_card(self):
         while True:
            self.first_card = self.deck.cards.pop(0)              # Flip the first card - remove from the deck
            if self.first_card.color == "wild" or self.first_card.value in ["reverse", "draw", "skip"]:  # if it's action or wid card, flip the next one
                print(f"Illegal move, flip again: {self.first_card}")
                self.deck.cards.pop(0)
            else:
                print(f"Flipping first card: {self.first_card}")
                break

    # Function to play card
    def play_card(self, player_num):
        player_hand = self.players_hand[player_num]
        #functionality to check if player has valid cards - pick from deck - call pass
        has_valid_card = any(card.color == self.current_color or card.value == self.first_card.value or card.color == "wild" for card in player_hand)

        if not has_valid_card:
            if len(self.deck.cards) > 0:
                print("no valid cards left. Pick up card")
                card = self.deck.cards.pop()
                player_hand.append(card)
                print(f"picked card:{card}")
                if card.value == self.first_card.value or card.color == "wild" or card.color ==self.current_color:
                    self.first_card = card
                    player_hand.remove(card) 
                else:
                    self.skip_turn = True
                    return
            else:
                print("Deck is empty")
                return
            
        #Player input
        while True: 
            print("------------------------------------")
            print(f"Player {player_num + 1}'s Hand:")
            for index, card in enumerate(player_hand, start=1):     # Access and print current player's hand
                card_str = str(card)
                print(f"Card {index}: {card_str}")
            try:
                print("------------------------------------------")
                players_input = int(input(f"Choose your card, Player {player_num + 1} (1-{len(player_hand)}): "))
                if players_input >= 1 and players_input <= len(player_hand):
                        played_card = player_hand[players_input - 1]

                        #check played card is valid
                        if played_card.value == self.first_card.value or played_card.color == self.first_card.color or played_card.color == "wild" or played_card.color ==self.current_color: 
                            print(f"You played: {played_card}")
                            player_hand.pop(players_input - 1)                                           # Remove the played card from players current hand
                            print(f"remaining cards: {len(player_hand)}")
                            self.first_card = played_card                                                 # First card now is the chosen valid card
                            print(f"new card facing up: {self.first_card}")
                            if played_card.value == "Draw 4":                                             # Enter logic for wild draw 4   
                                 for i in range(4):                                                       # Remove 4 from deck
                                    if len(self.deck.cards) > 0:
                                        card = self.deck.cards.pop()                                      # Add 4 to next players hand
                                        self.players_hand[(player_num+1)% self.num_players].append(card)  # The modulo creates a circular indexing ensuring the transition from the last player back to the first one 
                                    if self.skip_turn:
                                        print(f"Player {player_num + 1}'s turn is skipped.") # !!!!!!Issue with skipping 2 players when picked up card cannot be played !!!!!!!!
                                        self.skip_turn = False                               # Reset the flag for the next turn
                                        continue                                                       # Player who draws 4 cards loses their turn
                                 while True:
                                    chosen_color = input(f"Chosen color: Red, Blue, Green, Yellow").lower()  # Choose the color to continue the game after the wild 4 was played
                                    if chosen_color in Deck.colors:
                                        self.current_color = chosen_color
                                        print(f"new color is: {self.current_color}")
                                        self.first_card = Card(self.current_color, "Any")
                                        break
                                    else:
                                        print("Invalid color, choose again")
                            break                                           # Choose the color to continue the game after the wild 4 was played
                        else: 
                            print("card needs to match color or number")
                            print("------------------------------------")
                else:
                        print(f"Please enter a number between: (1-{len(player_hand)}")
            except ValueError:
                    print("Invalid input")

    def start_game(self):

        self.deal_all_cards()
        self.flip_first_card()
        while len(self.players_hand) > 0:                                # Players playing in turns
            for player_num in range(self.num_players):
              #  if self.skip_turn:
               #     print(f"Player {player_num + 1}'s turn is skipped.") # !!!!!!Issue with skipping 2 players when picked up card cannot be played !!!!!!!!
                #    self.skip_turn = False                               # Reset the flag for the next turn
                 #   continue                                             # Skip the rest of the loop and move to the next player   
                self.play_card(player_num)
        else: 
            print("game over")

                

        

