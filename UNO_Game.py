import random

cards = []
colors = ["blue", "green","red", "yellow"]
action_cards = ["reverse", "draw", "skip"]

# Function to create the deck
def createDeck():

    #Iterates four times, one for each color
    for i in range(4):
        number = 0

        # Iterates ten times, creating cards with numbers 0 to 9 for each color
        for j in range(10):
            card = colors[i] + "_" + str(number)
            #if number is zero add one of each colour. But for the numbers 1-9 add them twice for each colour.
            if number == 0:
                cards.append(card)
            else:
                cards.append(card)
                cards.append(card)
            number += 1
    # Action cards - each action card is represented twice for each colour 
    for i in range(4):
        index = 0
        for j in range(3):
            card = colors[i] + "_" + action_cards[index]
            cards.append(card)
            cards.append(card)
            index += 1
    # Wild cards
    for i in range(4):
        cards.append("wild")
        cards.append("wild Draw 4")

createDeck()

#Shuffle the cards
random.shuffle(cards)

# Function to deal one card to each player
def deal_single_cards(num_players):
    
    # 2D list to hold all players' hands
    players_hand=[]

    # Iterate through the players and create a list to hold each player's hand
    for y in range(num_players):
            # remove a card from the deck and add it to the player's hand
            list_hand = [cards.pop()]  # Ensures players_hand is a list containing one string
            players_hand.append(list_hand)
            
    return players_hand

# Function that calls the "deal_sigle_card" function and repeats the dealing for 7 rounds
def deal_all_cards(num_players, rounds):
    #For each iteration, create an empty list for the hand of each player
    hands = [[] for _ in range(num_players)]

    # Deal cards for the specified number of rounds
    for round_number in range(rounds):
        round_hands = deal_single_cards(num_players)
        for i in range(num_players):
            hands[i].extend(round_hands[i])

        # Print the hands after each round
        print(f"Round {round_number + 1}:")
        for i, hand in enumerate(hands):
            print(f"  Player {i + 1}'s Hand: {hand}")

        print()  # Blank line for better readability

    return hands

players_hands = deal_all_cards(4, 7)

