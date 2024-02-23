from game import Game
from agent import Agent  # Ensure you have an Agent class that implements the AI logic
from actions import play_matching_color, play_matching_number, play_wild_card, pick_card_from_deck


if __name__ == "__main__":
    num_players = 4  # Including the AI agent
    game = Game(num_players)
    game.deal_all_cards()  # Initially deal 7 cards to each player
    game.flip_first_card()  # Flip the first card to start the game

    agent = Agent(game)  # Initialize your AI agent here
    agent_player_num = 0  # Assuming the agent is the first player

    while True:  # Game loop
        current_player = game.get_current_player()

        game_state = game.get_state(agent_player_num)
        agent_hand = game.players_hand[agent_player_num]
        deck = game.deck 

        if current_player == agent_player_num:
            # It's the agent's turn
            state = game.get_state(agent_player_num)  # Get the current state for the agent
            action = agent.decide_action(game_state, agent_hand, deck)  # Agent decides on an action

            # Map the action to game actions
            if action == 0:
                action = play_matching_color(agent_hand, game_state['current_color'])  # Assuming game_state[0] is current color
            elif action == 1:
                action =  play_matching_number(agent_hand, game_state['current_value'])  # Assuming game_state[1] is current number
            elif action == 2:
                action =  play_wild_card(agent_hand)
            elif action == 3:
                action =  pick_card_from_deck(deck)
        
            card = action  # Assuming 'action' directly corresponds to a card object; adjust as needed

            # Execute the action and update the game state
            next_state, reward, game_over = game.play_action(agent_player_num, card)

            # Train the agent with the immediate feedback (short-term memory)
            agent.train_short_memory(state, action, reward, next_state, game_over)

            # Store the experience for later training (long-term memory)
            agent.remember(state, action, reward, next_state, game_over)

            if game_over:
                print("Game Over. The agent has won!")
                break  # Exit the game loop

        else:
            # Handle turns for PC players
            pc_state = game.get_state(current_player)  # Get the current state for the PC player
            pc_action = game.pc_decide_action(current_player)  # PC decides on an action based on a predefined strategy
            _, _, game_over = game.play_action(current_player, pc_action)  # Execute the action for the PC player

            if game_over:
                print(f"Game Over. Player {current_player + 1} has won!")
                break  # Exit the game loop if a PC player wins

        game.update_current_player()  # Move to the next player

    # After the game ends, train the agent with experiences stored in long-term memory
    agent.train_long_memory()
