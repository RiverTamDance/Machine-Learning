"""
Created by Taylor Richards
taylordrichards@gmail.com
April 04, 2024
"""
import time
from pyboy import PyBoy
import random

def main():
    start_time = time.perf_counter()

    #pyboy = PyBoy(r"C:\Users\John Q Hackerman\OneDrive\Documents\GameBoy\PokemonRed.gb", sound=True)
    pyboy = PyBoy(r"C:\Users\John Q Hackerman\OneDrive\Documents\GameBoy\Donkey Kong Country.gbc", sound=True)

    def a_press():
        pyboy.button('a')
        pyboy.tick()
    
    def b_press():
        pyboy.button('b')
        pyboy.tick()
    
    def right_press():
        pyboy.button('right')
        pyboy.tick()

    moveset = [a_press, b_press, right_press]

    # pyboy.tick()
    

    # game_area = pyboy.game_area()

    # pyboy.stop()

    # print(len(game_area))
    # print(len(game_area[0]))

    # for s in game_area:
    #     print(s)

    #The time between moves probably should also be randomly determined. or "no action" should be an action.

    while True:
        f = random.choice(moveset)
        f()

    # i = 0
    # while True:
    #     pyboy.tick()
    #     i+=1
    #     if i % 1000 == 0:
    #         print(pyboy.game_area())

    end_time = time.perf_counter()
    print("--- %s seconds ---" % (end_time - start_time))

if __name__ == "__main__":
    main()
