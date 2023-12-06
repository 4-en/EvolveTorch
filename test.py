from testing.XOR import XOR_test as XOR
from testing.cosine import cosine_test as cosine
from testing.birdgame import test_birdgame as bg

#XOR.testXOR()
#cosine.cosine_test()
bg.test_birdgame(20, name="Bird test 128pop 20gen no_crossover 0.02_mutate 2_layers old_mutate_alg")
bg.run_game()