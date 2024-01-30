from game import Game

if __name__ == "__main__":
    num_players = 4  # Set the number of players
    game = Game(num_players)
    game.deal_all_cards()
    game.flip_first_card()
    game.play_card(0)