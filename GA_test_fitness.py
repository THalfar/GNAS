import numpy as np

def binary_to_logarithmic_float(chromosome, start_high, end_high, end_low):
    """
    Convert a chromosome segment from binary to a floating-point number using logarithmic scaling.
    This function assumes the chromosome segment is divided into two parts: high and low.
    
    Parameters:
    chromosome (list): The chromosome as a list of binary digits.
    start_high (int): The starting index for the high part.
    end_high (int): The ending index (exclusive) for the high part.
    end_low (int): The ending index (exclusive) for the low part.
    
    Returns:
    float: The converted floating-point number.
    """
    output = 0.0

    # Convert the high part
    for i in range(start_high, end_high):
        output += chromosome[i] * 2 ** (i - start_high)

    # Convert the low part with inverse scaling
    for j in range(end_high, end_low):
        output += chromosome[j] * 2 ** (end_high - j - 1)

    return output


class GA_sum_fitness():
    @staticmethod
    def get_fitness(chromosome):
        return np.sum(chromosome)

class GA_Ackley_fitness():
    @staticmethod
    def get_fitness(chromosome):

        x = binary_to_logarithmic_float(chromosome, 4, 14, 24)
        y = binary_to_logarithmic_float(chromosome, 24, 34, 44)
        
        part1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        part2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        ackley_value = part1 + part2 + np.e + 20
                
        return -ackley_value

class GA_Rastrigin_fitness():
    @staticmethod
    def get_fitness(chromosome):
        """
        Calculate the extended Rastrigin function value for a given chromosome
        that includes variables x, y, z, and u.

        Parameters:
        chromosome (list or array): A numpy array representing a chromosome, where each element is a gene.

        Returns:
        float: The Rastrigin function value for the given chromosome.
        """
        x = binary_to_logarithmic_float(chromosome, 4, 14, 24)
        y = binary_to_logarithmic_float(chromosome, 24, 34, 44)
        z = binary_to_logarithmic_float(chromosome, 44, 54, 64)
        u = binary_to_logarithmic_float(chromosome, 64, 74, 84)  # Lisätään u:n laskenta
        
        # Laske Rastrigin-funktio neljälle muuttujalle
        A = 10
        n = 4  # Käsittelemme neljää muuttujaa
        sum_terms = (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y)) + (z**2 - A * np.cos(2 * np.pi * z)) + (u**2 - A * np.cos(2 * np.pi * u))
        rastrigin_value = A * n + sum_terms
        
        # Käytetään negatiivista Rastrigin-arvoa fitness-arvona
        return -rastrigin_value
