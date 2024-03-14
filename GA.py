import numpy as np 
import pandas as pd
import random 

class GA():

    def __init__(self, config, optimization_class):

        # Initialize the genetic algorithm parameters
        self.chromosome_length = config.get('chromosome_length', 10)
        self.population_size = config.get('population_size', 10)
        self.crossover_parts = config.get('crossover_rate', 1)
        self.mutation_rate = config.get('mutation_rate', 0.03)        
        self.max_generations = config.get('max_generations', 10)
        self.num_parents = config.get('parent_number', 5)
        self.num_children = config.get('children_number', 2)

        self.generation = 0
        self.optimization_class = optimization_class
        
    def initilize_population(self):

        # Create the initial population
        population = np.random.randint(2, size=(self.population_size, self.chromosome_length))
        
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

        # print(f"Population history fitness: \n{self.df_population_history}")
        
        # print(self.df_population)

    
    def select_parents(self):
        # Select the parents for the next generation
        self.df_population = self.df_population.sort_values(by='fitness', ascending=False)
        
        
        # print(self.df_population)
        # print(self.df_population.iloc[:self.num_parents, :])
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
    


        
from GA_sum_fitness import GA_sum_fitness     
config = {}
testi = GA(config, GA_sum_fitness)
testi.initilize_population()

for i in range(10):
    testi.calculate_fitness()
    parents = testi.select_parents()
    testi.crossover_population(parents)
    testi.mutate_population()



        

    
    