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
        self.chromosome_length = config.get('chromosome_length', 50)
        self.starting_population_size = config.get('population_size', 50)
        self.crossover_parts = config.get('crossover_rate', 1)
        self.mutation_rate = config.get('mutation_rate', 0.03)        
        self.max_generations = config.get('max_generations', 10)
        self.num_parents = config.get('parent_number', 25)
        self.num_children = config.get('children_number', 2)

        self.generation = 0
        self.optimization_class = optimization_class
        self.best_fitness = 0
        self.best_chromosome = None
        
    def initilize_population(self):

        # Create the initial population
        population = np.random.randint(2, size=(self.starting_population_size, self.chromosome_length))
        
        # Create the chromosome names for first generation
        chromosome_names = [f"gen{self.generation}_{idx}" for idx in range(population.shape[0])]
                            
        # Create the dataframe for the population and initiliaze the fitness column                            
        self.df_population = pd.DataFrame(population, index=chromosome_names)
        self.df_population.insert(0, 'fitness', 0.0)

        # Create the dataframe for the population history
        self.df_population_history = None
        

    def calculate_fitness(self):

        # Calculate the fitness for each chromosome in the population
        for idx in range(self.df_population.shape[0]):
            chromosome = self.df_population.iloc[idx, 1:]
            fitness = self.optimization_class.get_fitness
            self.df_population.iloc[idx, 0] = fitness(chromosome)

        if self.df_population_history is None:
            self.df_population_history = self.df_population
        else: 
            self.df_population_history = pd.concat([self.df_population_history, self.df_population], axis=0)

        
    def select_parents(self):
        # Select the parents for the next generation
        self.df_population = self.df_population.sort_values(by='fitness', ascending=False)
        
        if self.df_population['fitness'].max() > self.best_fitness:            
            self.best_fitness = self.df_population['fitness'].max()
            self.best_chromosome = self.df_population.iloc[0, 1:].values
            self.best_chromosome = np.array(self.best_chromosome,dtype=int)
            self.best_chromosome_name = self.df_population.index[0]
            print(f"Found new best fitness:{self.best_fitness} name:{self.best_chromosome_name} chromosome:{self.best_chromosome}")
        
        return self.df_population.iloc[:self.num_parents, :]


    def crossover_population(self, parents):

        self.generation += 1
        child_num = 0
        next_population = []
        next_population_names = []
        
        for i in range(parents.shape[0]):

            parent1 = parents.iloc[i, 1:].values
            available_parents = parents.drop(parents.index[i])

            for j in range(self.num_children):

                parent2_idx = random.choice(available_parents.index.tolist())
                parent2 = available_parents.loc[parent2_idx, 0:].values    
                
                crossover_point = random.randint(1, self.chromosome_length-1)  
            
                child = np.zeros(self.chromosome_length, dtype=int)
                child[:crossover_point] = parent1[:crossover_point]
                child[crossover_point:] = parent2[crossover_point:]   
                next_population.append(child)
                next_population_names.append(f"gen{self.generation}_{child_num}")
                child_num += 1

        next_population = pd.DataFrame(next_population, index=next_population_names)
        next_population.insert(0, 'fitness', 0.0)
        self.df_population = next_population


    def mutate_population(self):
        # Mutate the population based on the mutation rate
        for idx in range(self.df_population.shape[0]):
            chromosome = self.df_population.iloc[idx, 1:]
            for gene_idx in range(len(chromosome)):
                if random.random() < self.mutation_rate:
                    chromosome[gene_idx] = 1 - chromosome[gene_idx]
            self.df_population.iloc[idx, 1:] = chromosome

    def run(self):

        self.initilize_population()
        for i in range(self.max_generations):
            self.calculate_fitness()
            parents = self.select_parents()
            self.crossover_population(parents)
            self.mutate_population()
        
    def give_population_history(self):
        return self.df_population_history
    
    def get_best_chromosome(self):
        return self.best_chromosome
    
    def get_best_fitness(self):
        return self.best_fitness
    



        

    
    