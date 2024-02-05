from deck import Deck
from card import Card
import random

class Game:

    def __init__(self, num_players):
        self.deck = Deck()
        self.num_players = num_players
        self.players_hand = [[] for _ in range(num_players)]
        random.shuffle(self.deck.cards)  # Shuffle the cards
        self.current_color = None  # Current color, important for gameplay after a wild card
        self.skip_turn = False  # Flag to indicate when a player's turn is skipped

    def deal_single_cards(self):
        for player_index in range(self.num_players):
            if len(self.deck.cards) > 0:
                card = self.deck.cards.pop()
                self.players_hand[player_index].append(card)

    def deal_all_cards(self):
        for _ in range(7):  # Each player gets 7 cards initially
            self.deal_single_cards()

    def flip_first_card(self):
        while True:
            first_card = self.deck.cards.pop(0)
            if first_card.color != "wild":
                self.current_color = first_card.color
                self.first_card = first_card
                print(f"First card: {self.first_card}")
                break
            else:
                self.deck.cards.append(first_card)
                random.shuffle(self.deck.cards)

    def play_card(self, player_num):
        player_hand = self.players_hand[player_num]

        while True:
            print(f"Player {player_num + 1}'s turn. Current card: {self.first_card}")
            for index, card in enumerate(player_hand, start=1):
                print(f"Card {index}: {card}")
            try:
                card_choice = int(input("Choose a card to play (or 0 to draw a card): "))
                if card_choice == 0:
                    if len(self.deck.cards) > 0:
                        card = self.deck.cards.pop()
                        player_hand.append(card)
                        print(f"Player {player_num + 1} draws a card: {card}")
                        for index, card in enumerate(player_hand, start=1):
                            print(f"Card {index}: {card}")
                        if card.color == "wild" or card.color == self.current_color or card.value == self.first_card.value:
                            continue  # Allow the player to choose to play the drawn card
                        else:
                            print("Drawn card cannot be played. Turn ends.")
                            return  # End the current player's turn
                    else:
                        print("Deck is empty. Turn ends.")
                        return
                elif 1 <= card_choice <= len(player_hand):
                    played_card = player_hand[card_choice - 1]
                    if played_card.color == self.first_card.color or played_card.value == self.first_card.value or played_card.color == "wild":
                        if played_card.color == "wild":
                            while True:
                                chosen_color = input("Choose a color for the next play (Red, Blue, Green, Yellow): ").lower()
                                if chosen_color in ["red", "blue", "green", "yellow"]:
                                    self.current_color = chosen_color
                                    print(f"Next color is {self.current_color}")
                                    self.first_card = Card(chosen_color, "Any")  # Update the first card to reflect the chosen color
                                    break
                                else:
                                    print("Invalid color. Please choose Red, Blue, Green, or Yellow.")

                        if played_card.value == "Draw 4":
                            print("Wild Draw 4 played. Next player draws 4 cards and loses their turn.")
                            for _ in range(4):
                                if len(self.deck.cards) > 0:
                                    self.players_hand[(player_num + 1) % self.num_players].append(self.deck.cards.pop())
                            self.skip_turn = True
                            break

                        if played_card.color != "wild":
                            self.current_color = played_card.color  # Update the current color to that of the played card

                        self.first_card = played_card  # Update the first_card to the played card
                        player_hand.remove(played_card)
                        print(f"Player {player_num + 1} played: {played_card}")
                        break
                    else:
                        print("Invalid card selection. Card does not match current color or value.")
                else:
                    print("Invalid selection. Please select a valid card number.")
            except ValueError:
                print("Invalid input. Please enter a number.")






    def start_game(self):
        self.deal_all_cards()
        self.flip_first_card()

        while any(self.players_hand):  # Continue until one player runs out of cards
            for player_num in range(self.num_players):
                if self.skip_turn:
                    print(f"Player {player_num + 1}'s turn is skipped.")
                    self.skip_turn = False
                    continue
                self.play_card(player_num)
                if not self.players_hand[player_num]:  # Check if the current player has won
                    print(f"Player {player_num + 1} has won the game!")
                    return  # End the game

    print("Game over.")

