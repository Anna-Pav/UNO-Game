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
        self.direction = 1 # direction of play - 1:clockwise, -1=anticlockwise 
        self.current_player = 0  # Start with the first player

    def get_current_player(self):
        # Returns the index of the current player
        return self.current_player

    def update_current_player(self):
        # Update to the next player, considering the direction of the game
        self.current_player = (self.current_player + self.direction) % self.num_players

    # method to find out who the next player is after a card is played or a turn is skipped
    def get_next_player(self, current_player):
        next_player = (current_player + self.direction) % self.num_players
        return next_player

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
            first_card = self.deck.cards.pop(0) #draw facing card
            if first_card.color != "wild" and first_card.value.isdigit():
                self.current_color = first_card.color
                self.first_card = first_card
                print(f"First card: {self.first_card}")
                break
            else:
                # Invlaid card goes at the bottom of the deck and deck gets reshuffled 
                self.deck.cards.append(first_card)
                random.shuffle(self.deck.cards)

    """def play_card(self, player_num):
        player_hand = self.players_hand[player_num]

        while True:
            print(f"Player {player_num + 1}'s turn. Current card: {self.first_card} with color {self.current_color}")
            for index, card in enumerate(player_hand, start=1):
                print(f"Card {index}: {card}")
            try:
                card_choice = int(input("Choose a card to play (or 0 to draw a card): ")) 

                #handle player's choice to draw a card
                if card_choice == 0:            
                    if len(self.deck.cards) > 0:
                        card = self.deck.cards.pop()
                        player_hand.append(card)
                        print(f"Player {player_num + 1} draws a card: {card}")
                        if card.color == self.current_color or card.value == self.first_card.value or card.color == "wild":
                            print("You drew a playable card! You can choose to play this card now.")
                            continue  # Allow the player to choose to play the drawn card
                        else:
                            print("Drawn card cannot be played. Turn ends.")
                            return  # End the current player's turn
                    else:
                        print("Deck is empty. Turn ends.")
                        return
                elif 1 <= card_choice <= len(player_hand):
                    played_card = player_hand[card_choice - 1]
                    if played_card.color == self.current_color or played_card.value == self.first_card.value or played_card.color == "wild":
                        # handle wild card logic
                        if played_card.color == "wild":
                            chosen_color = input("Choose a color for the next play (Red, Blue, Green, Yellow): ").lower()
                            while chosen_color not in ["red", "blue", "green", "yellow"]:
                                print("Invalid color. Please choose Red, Blue, Green, or Yellow.")
                                chosen_color = input("Choose a color for the next play (Red, Blue, Green, Yellow): ").lower()
                            self.current_color = chosen_color
                            print(f"Next color is {self.current_color}")
                            self.first_card = Card(chosen_color, "Any")  # Use "Any" to indicate any value is acceptable since the color has been changed.

                        # wild card draw 4 logic
                        if played_card.value == "Draw 4":
                            print("Wild Draw 4 played. Next player draws 4 cards and loses their turn.")
                            for _ in range(4):
                                if len(self.deck.cards) > 0:
                                    self.players_hand[(player_num + 1) % self.num_players].append(self.deck.cards.pop())
                            self.skip_turn = True
                            self.first_card = Card(self.current_color, "Any")  # Reset the first card to be a dummy wild card with the chosen color

                        # Handle "Draw Two" card
                        if played_card.value == "draw":
                            print(f"'Draw Two' card played. Next player must draw 2 cards and miss their turn.")
                            next_player = (player_num + 1) % self.num_players
                            for _ in range(2):
                                if len(self.deck.cards) > 0:
                                    self.players_hand[next_player].append(self.deck.cards.pop())
                                else:
                                    print("Deck is empty. Cannot draw more cards.")
                            self.skip_turn = True  # Skip next player's turn

                        # handle skip card
                        if played_card.value == "skip":
                            print(f"Skip card played. Next player loses their turn.")
                            self.skip_turn = True

                        if played_card.value == "reverse":
                            print("Reverse card played. Direction of game changed!")
                            self.direction *= -1 #multiply current direction by -1 to change it

                        if played_card.color != "wild":
                            self.current_color = played_card.color
                            self.first_card = played_card  # Update the first_card to the played card

                        player_hand.pop(card_choice - 1)  # Remove the played card from the player's hand
                        print(f"Player {player_num + 1} played: {played_card}")

                        if not player_hand:
                            print(f"Player {player_num + 1} has played their last card!")
                            return "Game Over"
                        break
                    else:
                        print("Invalid card selection. Card does not match current color or value.")
                else:
                    print("Invalid selection. Please select a valid card number.")
            except ValueError:
                print("Invalid input. Please enter a number.")"""
    
    def play_action(self, player_num, card):
        player_hand = self.players_hand[player_num]
        game_over = False

        if card in player_hand:
            player_hand.remove(card)  # Remove the played card from the player's hand
            print(f"Player {player_num + 1} played: {card}")

            # Basic card play logic
            if card.color != "wild":
                self.current_color = card.color
            self.first_card = card

            # Special card logic
            if card.color == "wild":
                # Simplified logic for choosing the next color after a wild card
                # In an AI scenario, you might have the AI choose based on its hand or a predefined strategy
                self.current_color = "red"  # Example: default to red

                if card.value == "Draw 4":
                    self.handle_draw_four(player_num)

            elif card.value == "draw":
                self.handle_draw_two(player_num)

            elif card.value == "skip":
                self.skip_turn = True

            elif card.value == "reverse":
                self.direction *= -1

            # Check for game over condition
            if not player_hand:
                game_over = True
                print(f"Player {player_num + 1} has won the game!")

        else:
            print("Card not in player's hand.")
            return None, -1, game_over  # -1 reward for invalid choice, adjust as needed

        # Prepare the next state
        next_state = {
            'current_color': self.current_color,
            'current_value': self.first_card.value,
            'agent_hand': self.players_hand[player_num],
        }

        reward = 1  # Assign a reward for successfully playing a card, adjust based on your game's rules

        return next_state, reward, game_over

    def handle_draw_four(self, player_num):
        # Logic to handle the "Draw 4" wild card
        print("Wild Draw 4 played. Next player draws 4 cards and loses their turn.")
        for _ in range(4):
            next_player = (player_num + self.direction) % self.num_players
            if len(self.deck.cards) > 0:
                self.players_hand[next_player].append(self.deck.cards.pop())
        self.skip_turn = True

    def handle_draw_two(self, player_num):
        # Logic to handle the "Draw Two" card
        print("'Draw Two' card played. Next player must draw 2 cards and miss their turn.")
        for _ in range(2):
            next_player = (player_num + self.direction) % self.num_players
            if len(self.deck.cards) > 0:
                self.players_hand[next_player].append(self.deck.cards.pop())
        self.skip_turn = True

    def pc_decide_action(self, player_num):
        player_hand = self.players_hand[player_num]
        playable_cards = [card for card in player_hand if card.color == self.current_color or card.value == self.first_card.value or card.color == "wild"]

        if playable_cards:
            return random.choice(playable_cards)  # Randomly choose a playable card
        else:
            # No playable card, draw from the deck
            if len(self.deck.cards) > 0:
                drawn_card = self.deck.cards.pop()
                player_hand.append(drawn_card)
                print(f"Player {player_num + 1} draws a card: {drawn_card}")
                # Check if the drawn card is playable
                if drawn_card.color == self.current_color or drawn_card.value == self.first_card.value or drawn_card.color == "wild":
                    return drawn_card  # Play the drawn card if it's playable
                else:
                    print("Drawn card cannot be played. Turn ends.")
                    return None  # Turn ends if the drawn card is not playable
            else:
                print("Deck is empty. Turn ends.")
                return None  # Turn ends if the deck is empty


    def get_state(self, player_num):
        if self.first_card is None:
            raise ValueError("Game state requested before first card was flipped. Ensure game setup is complete.")
        # Assuming player_num is the index for the AI agent
        agent_hand = self.players_hand[player_num]
        state = {
            'current_color': self.current_color,
            'current_value': self.first_card.value,  # Assuming first_card is the top card of the play pile
            'agent_hand': agent_hand,  # You might need to encode this more efficiently for the agent
        }
        return state


    def start_game(self):
        self.deal_all_cards()
        self.flip_first_card()

        player_num = 0  # Start with the first player
        while any(self.players_hand):  # Continue until one player runs out of cards
            if self.skip_turn:
                print(f"Player {player_num + 1}'s turn is skipped.")
                self.skip_turn = False
            else:
                self.play_card(player_num)
                if not self.players_hand[player_num]:  # Check if the current player has won
                    print(f"Player {player_num + 1} has won the game!")
                    break  # End the game

            # Move to the next player considering the direction, skipping is handled within play_card
            player_num = self.get_next_player(player_num)

        print("Game over.")

