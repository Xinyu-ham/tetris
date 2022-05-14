from game import Field
from geneticalgo.base_model import BaseModel, ModelLayer
from geneticalgo.evolution import Population
import numpy as np
import geneticalgo as ga
import random
from evaluations import *
import dill
import time
from multiprocessing import Queue

import os, json

evaluation_functions = [max_height, avg_height, n_gaps,
                        lines_cleared, evenness, smoothness,
                        bumpiness, spikiness,
                        row_sides, death]
hidden = 10


class CustomTetrisField(Field):
    def __init__(self, evaluation_functions):
        Field.__init__(self)
        for function in evaluation_functions:
            self.add_evaluation_function(function)


def _get_uniform_variable():
    return random.uniform(0, 5)


class TetrisBot(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)

        self.tetris_game = CustomTetrisField
        if hidden:
            input_layer = ModelLayer(len(evaluation_functions))
            hidden_layer = ModelLayer(hidden)
            output_layer = ModelLayer(1)
            self.add_layers([input_layer, hidden_layer, output_layer])
        else:
            input_layer = ModelLayer(len(evaluation_functions))
            output_layer = ModelLayer(1)
            self.add_layers([input_layer, output_layer])

    def get_fitness(self):
        fitness = []
        for _ in range(3):
            tetris = self.tetris_game(evaluation_functions)
            score = tetris.start_game(self.chromosome, verbose_speed=0)
            fitness.append(score)
        return np.mean(fitness)


# Create Population and natural selection rules
if hidden:
    n_genes = len(evaluation_functions)*hidden + hidden
else:
    n_genes = len(evaluation_functions)
selection = ga.RouletteSelection()
crossover = ga.UniformCrossover()
mutation = ga.NoisyMutation(0.1, 0.05)

bots = Population(
    128, n_genes, TetrisBot,
    selection_method=selection,
    crossover_method=crossover,
    mutation_method=mutation,
    random_function=_get_uniform_variable
)

if __name__ == '__main__':
    population_file = 'current_pop.json'
    if population_file in os.listdir():
        current_pop = json.load(
            open(population_file, 'r')
        )

        current_pop = [np.array(current_pop[i]) for i in current_pop]
        bots.load_population(current_pop)
        print('Loaded previous results...')

    best_bot = bots.train_population(
        cycle=5, elitism=0.1,
        verbose=1, stopping_threshold=0,
        save=population_file)

    with open('best_bot.pkl', 'wb') as f:
        dill.dump(best_bot, f)









