"""
Created by Taylor Richards
taylordrichards@gmail.com
April 04, 2024
"""
import time
import pyautogui as pag

def main():
    start_time = time.perf_counter()

    screenWidth, screenHeight = pag.size()

    print("starting")
    for i in range(5):
        print(".")
        time.sleep(1)


    pag.keyDown('right')
    time.sleep(6)

    end_time = time.perf_counter()
    print("--- %s seconds ---" % (end_time - start_time))

if __name__ == "__main__":
    main()