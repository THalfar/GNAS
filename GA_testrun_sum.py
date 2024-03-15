import pandas as pd
from GA_sum_fitness import GA_sum_fitness, GA_Ackley_fitness
from GA import GA 
import matplotlib.pyplot as plt

config = {'starting_population_size' : 200,
            'chromosome_length' : 44,
            'num_parents' : 30,
            'children_number' : [10,3,2],
            'max_generations' : 50
          }

testi = GA(config, GA_Ackley_fitness)
testi.run()
pd.set_option('display.max_rows', 10000)
history = testi.give_population_history()
best_history = testi.give_best_history()
print(history.head(10000))

parsed_history = history.copy()
parsed_history['generation'] = parsed_history.index.str.split('_').str[0].str[3:].astype(int)
parsed_history['chromosome_idx'] = parsed_history.index.str.split('_').str[1].astype(int)

aggregated_history = parsed_history.groupby('generation').agg({'fitness': 'mean'})
mutation_mean = parsed_history.groupby('generation').agg({'mutrate': 'mean'})

fitness_mean = aggregated_history['fitness']
mutation_mean = mutation_mean['mutrate']

fig, ax1 = plt.subplots()

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

plt.title('Mean Fitness and Mutation Rate Through Generations')
fig.tight_layout()

plt.show()
