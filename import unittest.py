import unittest
from GA_sum_fitness import GA_sum_fitness 
import numpy as np

class TestGA_sum(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(GA_sum_fitness.get_fitness(np.array([1,1,1,1,1])), 5)
        self.assertEqual(GA_sum_fitness.get_fitness(np.array([0,0,0,0,0])), 0)
        self.assertEqual(GA_sum_fitness.get_fitness(np.array([0,0,1,0,0])), 1)
        self.assertEqual(GA_sum_fitness.get_fitness(np.array([1,0,1,0,1])), 3)
    
if __name__ == '__main__':
    unittest.main()
