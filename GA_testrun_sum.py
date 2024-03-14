import pandas as pd
import GA 
from GA_sum_fitness import GA_sum_fitness     
config = {}
testi = GA(config, GA_sum_fitness)
testi.initilize_population()
testi.run()
print(testi.give_population_history())

