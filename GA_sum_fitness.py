import numpy as np

class GA_sum_fitness():
    @staticmethod
    def get_fitness(chromosome):
        return np.sum(chromosome)

