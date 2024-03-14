import pandas as pd
from GA_sum_fitness import GA_sum_fitness  
from GA import GA 
import matplotlib.pyplot as plt

config = {'starting_population_size' : 100,
            'chromosome_length' : 200,
            'num_parents' : 25,
            'children_number' : [10,5,2],
            'max_generations' : 50,
            'optimization_class' : GA_sum_fitness
          }

testi = GA(config, GA_sum_fitness)
testi.run()
pd.set_option('display.max_rows', 10000)
history = testi.give_population_history()
history.info()

# Parse the history dataframe
parsed_history = history.copy()
parsed_history['generation'] = parsed_history.index.str.split('_').str[0].str[3:].astype(int)
parsed_history['chromosome_idx'] = parsed_history.index.str.split('_').str[1].astype(int)

# Aggregate different generations together
aggregated_history = parsed_history.groupby('generation').agg({'fitness': 'mean'})

# Calculate the fitness mean of all chromosomes in each generation
fitness_mean = aggregated_history['fitness']

# Print the fitness mean of each generation
print(fitness_mean)



plt.plot(fitness_mean)
plt.xlabel('Generation')
plt.ylabel('Mean Fitness')
plt.title('Mean Fitness Through Generations')
plt.show()