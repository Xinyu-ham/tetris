import multiprocessing

import json, os

if __name__ == '__main__':
    from base_model import BaseModel
else:
    from .base_model import BaseModel

import numpy as np
from multiprocessing import Process, Queue
from multiprocessing import Pool


class Population:
    def __init__(self, size, n_genes, base_model, selection_method=None,
                 crossover_method=None, mutation_method=None, random_function=None):
        self.size = size
        self.base_model = base_model
        self.n_genes = n_genes
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.generation = 0
        self.best_model = None
        self.mean_fitness = 1
        self.prev_mean_fitness = 1
        self.random_function = random_function
        self.stopping_threshold = 0
        self.population = self.initialize_population()

    def initialize_population(self):
        population = [
            Chromosome(
                genes=np.array([self.random_function() for _ in range(self.n_genes)]),
                model=self.base_model(),
                population=self
            ) for s in range(self.size)
        ]
        return population

    def train_population(self, cycle=-1, elitism=0, verbose=1, stopping_threshold=0.01, save=''):
        self.stopping_threshold = stopping_threshold
        while cycle != 0:
            # processes = []
            # fitnesses = []
            # for i, chromosome in enumerate(self.population):
            #     process = Process(target=_processing_get_fitness, args=(chromosome, i, queue))
            #     process.start()
            #     processes.append(process)
            #     chromosome.test = True
            #
            # for process in processes:
            #     process.join()
            #     fitnesses.append(queue.get())

            # while any([process.is_alive() for process in processes]):
            #     print('start process..')
            #     time.sleep(2)
            #
            # while fitnesses:
            #     i, fitness = fitnesses.pop()
            #     self.population[i].fitness = fitness

            ##### POOL ######
            pool = Pool(processes=multiprocessing.cpu_count() - 1)
            fitnesses = pool.imap(_processing_get_fitness_pool, self.population)

            for i, fitness in enumerate(fitnesses):
                self.population[i].fitness = fitness


            self.prev_mean_fitness = self.mean_fitness
            self._best_fitness()

            if verbose == 1:
                print(f'########## GEN {self.generation}, POP SIZE {self.size} ##########')
                print(f'Best score: {self.best_model.fitness}, mean fitness: {self.mean_fitness}')
                print(f'Improved by {100 * (self.mean_fitness - self.prev_mean_fitness)/self.prev_mean_fitness:.5f}%')
            elif verbose == 2:
                print(f'########## GEN {self.generation}, POP SIZE {self.size} ##########')
                print([m.fitness for m in self.population])
                print(f'Best score: {self.best_model.fitness}, mean fitness: {self.mean_fitness}')
                print(f'Improved by {100 * (self.mean_fitness - self.prev_mean_fitness) / self.prev_mean_fitness:.5f}%')

            elites = self._get_elites(elitism)
            select_size = self.size - len(elites)

            parents = self.selection_method.select_parent_generation(self.population, select_size)
            children = [self.crossover_method.breed(pair[0], pair[1]) for pair in parents]
            [self.mutation_method.mutate(child) for child in children]
            self.population = elites + children
            self.generation += 1

            cycle -= 1
            if cycle < 0 and self._stopping_condition(self.stopping_threshold):
                break

        if verbose:
            print('#' * 50)
            print('#' * 50)
            print(f'Training ran for {self.generation} generations')
            print(f'Best fitness: {self.best_model.fitness}')

        if save:
            try:
                pop_genes = {i: chromosome.genes.tolist() for i, chromosome in enumerate(self.population)}
                with open(save, 'w') as f:
                    f.write(json.dumps(pop_genes))
            except TypeError as e:
                os.remove(save)
                raise e


        return self.best_model

    def _stopping_condition(self, threshold):
        if self.generation < 10:
            return False
        return abs(self.mean_fitness - self.prev_mean_fitness)/self.prev_mean_fitness < threshold

    def _best_fitness(self):
        self.best_model = max([m for m in self.population], key=lambda chromosome: chromosome.fitness)
        self.mean_fitness = np.mean([m.fitness for m in self.population])

    def _get_elites(self, elitism):
        residents = list(sorted(self.population,
                                key=lambda chromosome: chromosome.fitness, reverse=True))
        threshold = int(elitism * self.size)
        return residents[:threshold]

    def load_population(self, existing_population):
        assert len(existing_population) == len(self.population)
        for i, genes in enumerate(existing_population):
            self.population[i].write_genes(genes)


class Chromosome:
    def __init__(self, genes, model, population):
        self.genes = genes
        self.length = len(self.genes)
        self.model = model
        self.population = population
        self.fitness = 0
        self.initialize_model()
        self.test = False

    def inherit(self, gene, position):
        self.genes[position] = gene

    def gene(self, position):
        return self.genes[position]

    def get_fitness(self):
        return self.model.get_fitness()

    def initialize_model(self):
        self.model.set_params(self.genes)
        self.model.chromosome = self

    def write_genes(self, genes):
        self.genes = genes


def _processing_get_fitness(chromosome, i, q):
    q.put([i, chromosome.get_fitness()])

def _processing_get_fitness_pool(chromosome):
    return chromosome.get_fitness()


