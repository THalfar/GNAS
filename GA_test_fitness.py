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


class GA_sum_fitness():
    @staticmethod
    def get_fitness(chromosome):
        return np.sum(chromosome)


class GA_Ackley_fitness:
    def __init__(self, integer_bits, fraction_bits):        
        self.integer_bits = integer_bits
        self.fraction_bits = fraction_bits
        self.total_bits_per_variable = integer_bits + fraction_bits
        self.__name__ = 'Ackley'

    def get_fitness(self, chromosome):
        x_start = 0
        y_start = self.total_bits_per_variable
        
        x = binary_to_logarithmic_float(chromosome, x_start, x_start + self.integer_bits, x_start + self.total_bits_per_variable)
        y = binary_to_logarithmic_float(chromosome, y_start, y_start + self.integer_bits, y_start + self.total_bits_per_variable)
        
        part1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        part2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        ackley_value = part1 + part2 + np.e + 20
                
        return -ackley_value

    def get_solution(self, chromosome):
        x_start = 0
        y_start = self.total_bits_per_variable
        
        x = binary_to_logarithmic_float(chromosome, x_start, x_start + self.integer_bits, x_start + self.total_bits_per_variable)
        y = binary_to_logarithmic_float(chromosome, y_start, y_start + self.integer_bits, y_start + self.total_bits_per_variable)
        
        return (x, y)

    def calculate_chromosome_length(self):        
        return  self.total_bits_per_variable * 2

    
class GA_Rastrigin_fitness:
    def __init__(self, integer_bits, fraction_bits, dimensions):
        
        self.integer_bits = integer_bits
        self.fraction_bits = fraction_bits
        self.total_bits_per_variable = integer_bits + fraction_bits
        self.dimensions = dimensions  
        self.__name__ = 'Rastrigin'

    def get_fitness(self, chromosome):
        args = [binary_to_logarithmic_float(chromosome,  i * self.total_bits_per_variable, self.integer_bits + i * self.total_bits_per_variable, self.total_bits_per_variable + i * self.total_bits_per_variable) for i in range(self.dimensions)]
        
        A = 10
        sum_terms = sum(x**2 - A * np.cos(2 * np.pi * x) for x in args)
        rastrigin_value = A * self.dimensions + sum_terms

        return -rastrigin_value

    def get_solution(self, chromosome):
        return tuple([binary_to_logarithmic_float(chromosome,  i * self.total_bits_per_variable, self.integer_bits + i * self.total_bits_per_variable, self.total_bits_per_variable + i * self.total_bits_per_variable) for i in range(self.dimensions)])

    def calculate_chromosome_length(self):
        return self.dimensions * self.total_bits_per_variable


