from unittest import TestCase
from unittest.main import main



class TestCartPole(TestCase):


    def test_methods(self):
        from .cartpole import CartPoleEnv
        env = CartPoleEnv()
        env.degrade()
        env.set_parameters()



if __name__ == '__main__':
    main()
