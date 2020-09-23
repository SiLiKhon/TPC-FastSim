import numpy as np


class Identity:
    def scale(self, x):
        return x
    
    def unscale(self, x):
        return x


class Logarithmic:
    def scale(self, x):
        return np.log10(1 + x)

    def unscale(self, x):
        return 10 ** x - 1