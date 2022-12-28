from numpy import random as rnd


def random(a, b, n):
    return (a + rnd.rand(n) * (b - a))
