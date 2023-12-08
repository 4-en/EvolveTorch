from testing.XOR import XOR_test as XOR
from testing.cosine import cosine_test as cosine
from testing.birdgame import test_birdgame as bg

def xor():
    # test untrained model
    XOR.testXOR()

    # train model with gen algorithm
    model = XOR.trainXOR()

    # test trained model
    XOR.testXOR(model)
#cosine.cosine_test()
#bg.test_birdgame(50, name="Bird test 100pop 50gen no_crossover 0.02_mutate 2_layers pretrained")
#bg.run_game()

xor()