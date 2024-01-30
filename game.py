from deck import Deck
import random

class Game:

    def __init__(self,num_players):
        self.deck = Deck()
        self.num_players = num_players
        self.players_hand = [[] for _ in range(num_players)]
        #Shuffle the cards
        random.shuffle(self.deck.cards)

        #flag to indicate when a player loses their turn
        self.skip_turn = False

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

        #Player input
        while True: 

            print(f"Player {player_num + 1}'s Hand:")

            #access and print current player's hand
            for index, card in enumerate(player_hand, start=1):
                card_str = str(card)
                print(f"Card {index}: {card_str}")

            try:
                print("------------------------------------------")
                players_input = int(input(f"Choose your card, Player {player_num + 1} (1-{len(player_hand)}): "))

                if players_input >= 1 and players_input <= len(player_hand):
                        played_card = player_hand[players_input - 1]

                        #check played card is valid
                        if played_card.value == self.first_card.value or played_card.color == self.first_card.color or played_card.color == "wild": 
                            print(f"You played: {played_card}")
                            #remove the played card from players current hand
                            player_hand.pop(players_input - 1)

                            print(f"remaining cards: {len(player_hand)}")
                            
                            # first card now is the chosen valid card
                            self.first_card = played_card
                            print(f"new card facing up: {self.first_card}")

                            # enter logic for wild draw 4
                            if played_card.value == "Draw 4":
                                #remove 4 from deck
                                 for i in range(4):
                                    if len(self.deck.cards) > 0:
                                        card = self.deck.cards.pop()
                                        # add 4 to next players hand
                                        # the modulo creates a circular indexing ensuring the transition from the last player back to the first one
                                        self.players_hand[(player_num+1)% self.num_players].append(card)
                                    # player who draws 4 cards loses thier turn
                                    self.skip_turn = True
                            #implement logic after draw 4 the game skips to the next player

                            #exit loop after valid card is played
                            break
                            
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

        # players playing in turns
        while len(self.players_hand) > 0:
            for player_num in range(self.num_players):
                if self.skip_turn:
                    print(f"Player {player_num + 1}'s turn is skipped.")
                    # reset the flag for the next turn
                    self.skip_turn = False
                    # skip the rest of the loop and move to the next player
                    continue 
                    
                self.play_card(player_num)
        else: 
            print("game over")

                

        

