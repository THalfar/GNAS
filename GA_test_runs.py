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
    mutation_mean = parsed_history.groupby('generation').agg({'mutrate': 'mean'})

    fitness_mean = aggregated_history['fitness']
    mutation_mean = mutation_mean['mutrate']

    time.sleep(0.1) # WSL2 problem with showing the plot
    fig, ax1 = plt.subplots(figsize=(15, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Mean Fitness', color=color)
    ax1.plot(fitness_mean, color=color, label='Mean Fitness')
    ax1.scatter(best_history.index, best_history['best_fitness'], color='green', s=20, label='Best Fitness', zorder=5)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Mean Mutation Rate', color=color)
    ax2.plot(mutation_mean, color=color, label='Mean Mutation Rate')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Mean Fitness and Mutation Rate. Num chrosomes:{parsed_history.shape[0]}, {optimization_function.__name__} minimization')
    fig.tight_layout()

    # TODO: Some WSL2 problem with showing the plot
    plt.show()

# Configuration for Ackley function minimization. Keeping minimum mutation rate higher to avoid premature convergence.
config_ackley = {'starting_population_size' : 200,
                'chromosome_length' : 33,
                'num_parents' : 30,
                'children_number' : [5,3,2],
                'max_generations' : 50,
                'max_fitness' : '0.0',
                'min_mutation_rate' : 0.03
                }

# Configuration for Rastrigin function minimization. Keeping minimum mutation rate higher to avoid premature convergence.
config_rastrigin = {'starting_population_size' : 300,
                    'chromosome_length' : 61,
                    'num_parents' : 40,
                    'children_number' : [5,3,2],
                    'max_generations' : 100,
                    'max_fitness' : '0.0',
                    'min_mutation_rate' : 0.02
                    }

# Run and plot GA for Ackley function minimization
run_and_plot_ga(config_ackley, GA_Ackley_fitness)

# Run and plot GA for Rastrigin function minimization
run_and_plot_ga(config_rastrigin, GA_Rastrigin_fitness)
