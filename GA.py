import numpy as np 
import pandas as pd
import random 

"""
This is the main class for the genetic algorithm. It contains the main functions for the genetic algorithm.
The genetic algorithm principle is simple where only the fittest individuals are selected to produce the next generation.
The main functions are:
- initilize_population: Create the initial population
- calculate_fitness: Calculate the fitness for each chromosome in the population
- select_parents: Select the parents for the next generation
- crossover_population: Create the next generation by crossover
- mutate_population: Mutate the population based on the mutation rate
- give_population_history: Return the population history as a dataframe
- get_best_chromosome: Return the best chromosome
- get_best_fitness: Return the best fitness
"""
class GA():

    def __init__(self, config, optimization_class):

        # Initialize the genetic algorithm parameters
        self.chromosome_length = config.get('chromosome_length', 20)
        self.starting_population_size = config.get('starting_population_size', 50)
        self.crossover_parts = config.get('crossover_parts', 1)
        self.mutation_rate = config.get('mutation_rate', 0.03)        
        self.max_generations = config.get('max_generations', 10)
        self.num_parents = config.get('num_parents', 10)
        self.num_childrens = config.get('num_childrens', [5,3,2])
        self.max_fitness = config.get('max_fitness', None)
        self.max_fitness_tolerance = config.get('max_fitness_tolerance', 1e-6)
        self.min_mutation_rate = config.get('min_mutation_rate', 0.01)
        self.len_mutation_rate_gene = config.get('len_mutation_rate_gene', 3)
        self.invidual_mutation_rate = config.get('invidual_mutation_rate', 0.01)
        self.mutation_rate_scale = config.get('mutation_rate_scale', 0.005)

        self.generation = 0
        self.optimization_class = optimization_class
        self.best_fitness = float('-inf')
        self.best_chromosome = None
        if self.max_fitness is not None:
            self.max_fitness = self.max_fitness
        self.best_solution_found = False
        

    def binary_to_integer(self, chromosome, start_index, end_index):
        """
        Convert a binary sequence within a chromosome from a specified start index to an end index into an integer.

        Parameters:
        chromosome (list or array): The chromosome represented as a sequence of binary digits (0s and 1s).
        start_index (int): The starting index of the slice of the chromosome to be converted.
        end_index (int): The ending index (exclusive) of the slice of the chromosome to be converted.

        Returns:
        int: The integer value of the specified binary sequence within the chromosome.
        """
        integer_value = 0
        for i in range(start_index, end_index):
            integer_value += chromosome[i] * 2 ** (i - start_index)

        return integer_value
        
    def initilize_population(self):

        # Create the initial population
        population = np.random.randint(2, size=(self.starting_population_size, self.chromosome_length))
        
        # Create the chromosome names for first generation
        chromosome_names = [f"gen{self.generation}_{idx}" for idx in range(population.shape[0])]
                            
        # Create the dataframe for the population and initiliaze the fitness column                            
        self.df_population = pd.DataFrame(population, index=chromosome_names)
        self.df_population.insert(0, 'mutrate', 0.0)
        self.df_population.insert(0, 'fitness', 0.0)        

        for idx in range(self.df_population.shape[0]):
            chromosome = self.df_population.iloc[idx, 2:self.len_mutation_rate_gene+2]        
            mutation_rate = self.min_mutation_rate + self.binary_to_integer(chromosome, 0, self.len_mutation_rate_gene) * self.mutation_rate_scale
            # print(f"Mutation rate: {mutation_rate}")
            # print(f"Chromosome:\n{chromosome[:4]}")            
            # print(f"{self.df_population.iloc[idx, 1]}")    
            self.df_population.iloc[idx, 1] = mutation_rate
            
    
        # Create the dataframe for the population history
        self.df_population_history = None
        self.df_best_history = None

    def calculate_fitness(self):

        # Calculate the fitness for each chromosome in the population
        for idx in range(self.df_population.shape[0]):
            chromosome = self.df_population.iloc[idx, 2:]            
            # print(chromosome.values)
            # print(chromosome.values.shape)
            self.df_population.iloc[idx, 0] = self.optimization_class.get_fitness(chromosome)
            chromosome_mut = self.df_population.iloc[idx, 2:self.len_mutation_rate_gene+2]            
            # print(chromosome_mut.values)
            # print(chromosome_mut.values.shape)
            mutation_rate = self.min_mutation_rate + self.binary_to_integer(chromosome_mut, 0, self.len_mutation_rate_gene) * self.mutation_rate_scale
            # print(f"Mutation rate: {mutation_rate}")
            # print(f"Chromosome:\n{chromosome[:4]}")
            self.df_population.iloc[idx, 1] = mutation_rate   

        if self.df_population['fitness'].max() > self.best_fitness:            
            self.best_fitness = self.df_population['fitness'].max()
            max_fitness_idx = self.df_population['fitness'].idxmax()
            # print(max_fitness_idx)
            # print(self.df_population.loc[max_fitness_idx])

            self.best_chromosome = self.df_population.loc[max_fitness_idx]
            self.best_chromosome_name = max_fitness_idx
            self.best_chromosome = self.best_chromosome[2:]
            self.best_chromosome = np.array(self.best_chromosome,dtype=int)            
            solution = self.optimization_class.get_solution(self.best_chromosome)

            print("*" * 10)
            print(f"Found new best fitness:{self.best_fitness}, solution:{solution} name:{self.best_chromosome_name}")
            print("*" * 10)

            if self.df_best_history is None:
                self.df_best_history = pd.DataFrame(columns=['best_fitness'])                
            self.df_best_history.loc[self.generation] = [self.best_fitness]
            
            # Break GA if given max fitness is reached
            if self.max_fitness is not None and np.abs(self.best_fitness -self.max_fitness) <= self.max_fitness_tolerance:
                print("!" * 10)
                print(f"Max fitness reached in generation:{self.generation} with fitness:{self.best_fitness}, max_fitness:{self.max_fitness}, max_fitness_tolerance:{self.max_fitness_tolerance}")
                print("!"*10)
                self.best_solution_found = True
                    

        if self.df_population_history is None:
            self.df_population_history = self.df_population
        else: 
            self.df_population_history = pd.concat([self.df_population_history, self.df_population], axis=0)

        
    def select_parents(self):
        # Select the parents for the next generation
        self.df_population = self.df_population.sort_values(by='fitness', ascending=False)
        return self.df_population.iloc[:self.num_parents, :]


    def crossover_population(self, parents):

        child_num = 0
        next_population = []
        next_population_names = []
        children_idx = 0
        child_num = 0

        for i in range(parents.shape[0]):
            
            parent1 = parents.iloc[i, 1:].values
            available_parents = parents.drop(parents.index[i])

            if children_idx < len(self.num_childrens):
                amount_children = self.num_childrens[children_idx]                
            else:
                amount_children = self.num_childrens[-1]
                
            for _ in range(amount_children):
                        
                        parent2_idx = random.choice(available_parents.index.tolist())
                        parent2 = available_parents.loc[parent2_idx, 0:].values    
                        crossover_point = random.randint(1, self.chromosome_length-1)  
                    
                        child = np.zeros(self.chromosome_length, dtype=int)
                        child[:crossover_point] = parent1[:crossover_point]
                        child[crossover_point:] = parent2[crossover_point:]   
                        next_population.append(child)
                        next_population_names.append(f"gen{self.generation}_{child_num}")
                        child_num += 1

            children_idx += 1

        next_population = pd.DataFrame(next_population, index=next_population_names)
        next_population.insert(0, 'mutrate', 0.0)
        next_population.insert(0, 'fitness', 0.0)
        self.df_population = next_population


    def mutate_population(self):
        # Mutate the population based on the mutation rate
        for idx in range(self.df_population.shape[0]):
            mutation_rate = self.df_population.iloc[idx, 1]
            chromosome = self.df_population.iloc[idx, 2:]            
                  
            if random.random() < self.invidual_mutation_rate:   
                for gene_idx in range(len(chromosome)):                
                    if random.random() < mutation_rate:
                        chromosome[gene_idx] = 1 - chromosome[gene_idx]
                self.df_population.iloc[idx, 2:] = chromosome
        

    
    def print_generation_stats(self):
        # Print the statistics for the current generation        
        print(f"Population Statistics for generation:{self.generation}")
        print(f"Population   size: {self.df_population.shape[0]}")
        print(f"Population    Max: {self.df_population['fitness'].max():.8f}")
        print(f"Population    Min: {self.df_population['fitness'].min():.8f}")
        print(f"Population Median: {self.df_population['fitness'].median():.8f}")
        print(f"Population   Mean: {self.df_population['fitness'].mean():.8f}")   
        print(f"Population    std: {self.df_population['fitness'].std():.8f}")     
        print(f"Mutarate     mean: {self.df_population['mutrate'].mean():.3f}")
        print(f"Mutarate   median: {self.df_population['mutrate'].median():.3f}")
        print(f"Mutarate      max: {self.df_population['mutrate'].max():.3f}")   
        print(f"Mutarate      min: {self.df_population['mutrate'].min():.3f}")   
        print("---------------------------------------------")
        

    def run(self):

        self.initilize_population()
        while self.generation <= self.max_generations and not self.best_solution_found:
            self.calculate_fitness()
            self.print_generation_stats()
            parents = self.select_parents()
            self.crossover_population(parents)            
            self.mutate_population()
            self.generation += 1
        
    def give_population_history(self):
        return self.df_population_history

    def give_best_history(self):    
        return self.df_best_history
    
    def get_best_chromosome(self):
        return self.best_chromosome
    
    def get_best_fitness(self):
        return self.best_fitness
