from dataclasses import dataclass

class Colour:
    def __init__(self, r:int, g:int, b:int):
        if r < 0 or r > 255: raise Exception("R must be between 0 and 255")
        if g < 0 or g > 255: raise Exception("G must be between 0 and 255")
        if b < 0 or b > 255: raise Exception("B must be between 0 and 255")

        self.r = r
        self.g = g
        self.b = b
    
    def rgb(self):
        return (self.r, self.g, self.b)
    
    def hex(self):
        r = hex(self.r)
        g = hex(self.g)
        b = hex(self.b)

        return f"#{r}{g}{b}"

cmap = [
        Colour(28, 26, 228),  # #e41a1c
        Colour(184, 126, 55),  # #377eb8
        Colour(74, 175, 77),  # #4daf4a
        Colour(163, 78, 152),  # #984ea3
        Colour(0, 127, 255),  # #ff7f00
        Colour(51, 255, 255),  # #ffff33
        Colour(40, 86, 166),  # #a65628
        Colour(191, 129, 247),  # #f781bf
        Colour(153, 153, 153),  # #999999
    ]
