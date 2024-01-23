class Card:
    def __init__(self, color, value):
        self.color = color
        self.value = value

    # returns a string representation of the object.
    def __str__(self):
        return f"{self.color}_{self.value}"