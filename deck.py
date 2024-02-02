import random
from card import Card

class Deck:

    colors = ["blue", "green","red", "yellow"]

    def __init__(self):
        self.cards = []
        self.createDeck()

    # Function to create the deck
    def createDeck(self):

        
        action_cards = ["reverse", "draw", "skip"]

        #Iterates four times, one for each color
        for i in range(4):
            color = self.colors[i] # get color name
            number = 0

            # Iterates ten times, creating cards with numbers 0 to 9 for each color
            for j in range(10):
                card = Card(color, str(number))
                #if number is zero add one of each colour. But for the numbers 1-9 add them twice for each colour.
                if number == 0:
                    self.cards.append(card)
                else:
                    self.cards.append(card)
                    self.cards.append(card)
                number += 1
             # Action cards - each action card is represented twice for each colour 
            for action_card in action_cards:
                self.cards.append(card)
                self.cards.append(card)

        # Wild cards
        for i in range(4):
            self.cards.append(Card("wild", ""))
            self.cards.append(Card("wild", "Draw 4"))