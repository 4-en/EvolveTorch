from .testing.XOR import XOR_test as XOR
from .testing.cosine import cosine_test
from .testing.birdgame import test_birdgame as bg

def xor():
    # test untrained model
    XOR.testXOR()

    # train model with gen algorithm
    model = XOR.trainXOR()

    # test trained model
    XOR.testXOR(model)

def cosine():
    cosine_test.test_cosine()
    model = cosine_test.train_cosine()
    cosine_test.test_cosine(model=model)

def birdgame():
    # you play as player 1 and the ai as player 2
    # press space to fly
    # r to reset
    bg.run_game() # test with pretrained
    model = bg.test_birdgame(10) # train for 10 gen
    bg.run_game(model=model) # test with newly trained

def main():
    # run all tests
    xor()
    cosine()
    birdgame()

if __name__ == "__main__":
    # run all tests
    main()