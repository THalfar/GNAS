import pandas as pd
from GA_test_fitness import GA_Ackley_fitness, GA_Rastrigin_fitness
from GA import GA 
import matplotlib.pyplot as plt
import time 

"""
This is a test script to run the genetic algorithm and plot the mean fitness and mutation rate over generations.
The chromosome length is with purpose too long to study GA behavior.
"""

def run_and_plot_ga(config, optimization_function):
    testi = GA(config, optimization_function)
    testi.run()
    pd.set_option('display.max_rows', 10000)
    history = testi.give_population_history()
    best_history = testi.give_best_history()

    parsed_history = history.copy()
    parsed_history['generation'] = parsed_history.index.str.split('_').str[0].str[3:].astype(int)
    parsed_history['chromosome_idx'] = parsed_history.index.str.split('_').str[1].astype(int)

    aggregated_history = parsed_history.groupby('generation').agg({'fitness': 'mean'})
    fitness_mean = aggregated_history['fitness']


    time.sleep(0.5) # WSL2 problem with showing the plot
    plt.figure(figsize=(15, 6))

    color = 'tab:blue'
    plt.xlabel('Generation')
    plt.ylabel('Mean Fitness', color=color)
    plt.plot(fitness_mean, color=color, label='Mean Fitness')
    plt.scatter(best_history.index, best_history['best_fitness'], color='green', s=20, label='Best Fitness', zorder=5)
    plt.tick_params(axis='y', labelcolor=color)

    plt.title(f'Mean Fitness and Mutation Rate. Num chrosomes:{parsed_history.shape[0]}, {optimization_function.__name__} minimization')
    plt.tight_layout()

    # TODO: Some WSL2 problem with showing the plot
    plt.show()


ackley = GA_Ackley_fitness(integer_bits=4, fraction_bits=8)
len_ackley_gene = ackley.calculate_chromosome_length()


# Configuration for Ackley function minimization. 
config_ackley = {'starting_population_size' : 200,
                'chromosome_length' : len_ackley_gene,
                'num_parents' : 50,
                'children_number' : [10,5,2],
                'max_generations' : 50,
                'max_fitness' : 0.0,
                'mutation' : [0.20, 0.15, 0.05],
                'gen_mutation' : [0.1, 0.05, 0.03, 0.01]            
                }

run_and_plot_ga(config_ackley, ackley)
print(f"len_ackley_gene: {len_ackley_gene}")

rastrigin = GA_Rastrigin_fitness(integer_bits=4, fraction_bits=8, dimensions=8)
len_rastrigin_gene = rastrigin.calculate_chromosome_length()


# Configuration for Rastrigin function minimization. 
config_rastrigin = {'starting_population_size' : 1000,
                    'chromosome_length' : len_rastrigin_gene,
                    'num_parents' : 500,
                    'num_childrens' : [2],
                    'max_generations' : 200,
                    'max_fitness' : 0.0,   
                    'mutation' : [0.20, 0.1, 0.05],
                    'gen_mutation' : [0.05, 0.03, 0.02, 0.02]                                                                       
                    }

run_and_plot_ga(config_rastrigin, rastrigin)
print(f"len_rastrigin_gene: {len_rastrigin_gene}")