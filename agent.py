import torch
import random
import numpy as np
from game import Game
from collections import deque
from model import Linear_QNet, QTrainer
from actions import *
import torch.nn.functional as F

MAX_MEMORY = 100_000
BATCH_SIZE=1_000
LR = 0.001

class Agent:

    color_map = {'red': 0, 'blue': 1, 'green': 2, 'yellow': 3, 'wild': 4}
    value_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'skip': 10, 'reverse': 11, 'draw': 12, 'Draw 4': 13}
    
    # Define the action mapping as a class attribute
    action_mapping = {}
    action_id = 0  # Start with 0 and increment for each unique card

    for color in ["blue", "green", "red", "yellow"]:
        # Number cards
        for number in range(10):  # 0-9
            action_mapping[f"{color}_{number}"] = action_id
            action_id += 1
            if number != 0:  # Number cards 1-9 appear twice
                action_mapping[f"{color}_{number}"] = action_id
                

        # Action cards
        for action in ["reverse", "draw", "skip"]:
            action_mapping[f"{color}_{action}"] = action_id
            action_id += 1
            # Each action card appears twice per color
            action_mapping[f"{color}_{action}"] = action_id
            action_id += 1

    # Wild cards
    for wild_card in ["wild", "wild_Draw 4"]:
        action_mapping[wild_card] = action_id
        action_id += 1
        # Each wild card appears four times
        action_mapping[wild_card] = action_id
        action_id += 1
        action_mapping[wild_card] = action_id
        action_id += 1
        action_mapping[wild_card] = action_id
        action_id += 1

    total_actions = action_id  # Total number of unique actions

    def __init__(self,game,number_of_games=1000):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, self.total_actions)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        self.game = None
        self.number_of_games = number_of_games
        self.game_over = False
        self.number_wins = 0
        self.number_loses = 0

        self.game = game

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        for sample in mini_sample:
            state, action, reward, next_state, game_over = sample
            state_tensor = self.convert_game_state_to_tensor(state)
            next_state_tensor = self.convert_game_state_to_tensor(next_state)
            action_int = self.encode_action(action)
            self.trainer.train_step(state_tensor, action_int, reward, next_state_tensor, game_over)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        # Convert state and next_state to tensors
        state_tensor = self.convert_game_state_to_tensor(state)
        next_state_tensor = self.convert_game_state_to_tensor(next_state)
        # Ensure action is an integer representing the chosen action
        action_int = self.encode_action(action)

        # Pass tensors and encoded action to train_step
        self.trainer.train_step(state_tensor, action_int, reward, next_state_tensor, game_over)

    

    def encode_action(self, action):
        # Ensure 'action' is a Card object
        if isinstance(action, Card):
            # Handle wild cards separately since they don't have a color
            if action.color == "wild":
                card_key = action.value  # For wild cards, use just the value as the key
            else:
                # For regular cards, use both color and value to form the key
                card_key = f"{action.color}_{action.value}"

            # Use the action mapping to convert the card key to an integer
            encoded_action = self.action_mapping.get(card_key, -1)  # Default to -1 if the card is not recognized

            return encoded_action
        else:
            # If action is not a Card object, handle accordingly
            return -1  # Placeholder for unrecognized actions
    
    def get_action(self, state, agent_hand, current_color, current_number, deck, player_num):
            action_size = self.total_actions  # Use the total number of actions instead of hardcoding
            choice = None

            # Adjust epsilon based on the number of games played (decay)
            self.epsilon = max(80 - self.game_count, 0)  # Ensure epsilon does not go negative

            # Exploration vs. Exploitation
            if random.randint(0, 100) < self.epsilon:
                # Exploration: Random action
                choice = random.randint(0, action_size - 1)
            else:
                # Exploitation: Use the model to choose action
                state0 = torch.tensor(state, dtype=torch.float)
                prediction = self.model(state0)
                choice = torch.argmax(prediction).item()

                # Execute the chosen action
            if choice == 0:
                card_played = play_matching_color(agent_hand, current_color)
            elif choice == 1:
                card_played = play_matching_number(agent_hand, current_number)
            elif choice == 2:
                card_played = play_wild_card(agent_hand)
            elif choice == 3:
                card_played = pick_card_from_deck(deck)

            # Calculate the reward
            reward = 0
            if card_played:
                if card_played.color == current_color or card_played.value == current_number or card_played.color == "wild":
                    reward = 1  # Positive reward for a matching action or playing a wild card
                else:
                    reward = -0.5  # Small penalty for not matching (if applicable, depends on your game logic)
            else:
                reward = -1  # Penalty for not being able to play a card
            
            # Call the Game class's play_action method
            next_state, reward, game_over = self.game.play_action(player_num, card_played)

            # Remember the action and reward
            self.remember(state, choice, reward, next_state, game_over)

            return card_played

    def decide_action(self, game_state, agent_hand, deck):
        # Use the convert_game_state_to_tensor method to properly encode the game state
        state_tensor = self.convert_game_state_to_tensor(game_state)
        
        # Use the model to get the action probabilities
        action_probs = self.model(state_tensor)
        return torch.argmax(action_probs).item()

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return  # Not enough samples in memory to train

        # Randomly sample experiences from memory
        mini_batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, game_overs = zip(*mini_batch)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        game_overs = torch.tensor(game_overs, dtype=torch.bool)

        # Get the current Q values from the model using the states from the mini-batch
        current_q = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Compute the expected Q values (target Q values)
        max_next_q = self.model(next_states).max(1)[0]
        expected_q = rewards + (self.gamma * max_next_q * (~game_overs))

        # Compute loss between current Q values and expected Q values
        loss = F.mse_loss(current_q, expected_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def encode_card(self, card):
        
        color_encoding = [0] * len(Agent.color_map)
        value_encoding = [0] * len(Agent.value_map)

        if card.color in Agent.color_map:
            color_encoding[Agent.color_map[card.color]] = 1
        if card.value in Agent.value_map:
            value_encoding[Agent.value_map[card.value]] = 1

        return color_encoding + value_encoding

    def encode_hand(self, hand):
        hand_encoding = []
        for card in hand:
            hand_encoding.extend(self.encode_card(card))
        return hand_encoding

    def convert_game_state_to_tensor(self, game_state):
        if game_state is None:
        # Handle the case where game_state is None
        # You might want to return a default tensor or handle this case before calling the function
          return torch.zeros([1, 11]) 
        
        # Initialize encodings
        current_color_encoding = [0] * 4  # One hot encoding for 4 colors
        current_value_encoding = [0] * 4  # One hot encoding for categories: number, skip, reverse, draw/draw4

        # Set current color encoding
        if game_state['current_color'] in self.color_map:
            current_color_encoding[self.color_map[game_state['current_color']]] = 1

        # Set current value encoding based on categories
        if game_state['current_value'].isdigit():
            current_value_encoding[0] = 1  # Number cards
        elif game_state['current_value'] in ['skip', 'reverse']:
            current_value_encoding[1] = 1  # Skip or reverse
        elif game_state['current_value'] == 'draw':
            current_value_encoding[2] = 1  # Draw two
        elif game_state['current_value'] == 'Draw 4':
            current_value_encoding[3] = 1  # Wild draw four

        # Hand encoding: Count of cards per color + wild cards
        hand_encoding = [0] * 5  # 4 colors + wild
        for card in game_state['agent_hand']:
            hand_encoding[self.color_map[card.color]] += 1

        # Combined encoding to fit 11 elements
        combined_encoding = current_color_encoding + current_value_encoding + hand_encoding[:3]  # Adjust to ensure total length is 11

        return torch.FloatTensor([combined_encoding])

    # Usage
    def usage(self, player_num):
        game_state = self.game.get_state(player_num)
        tensor_state = self.convert_game_state_to_tensor(game_state)


        
