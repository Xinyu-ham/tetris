from numpy.random import choice
from random import randrange, sample

class SelectionMethod():
    def __init__(self):
        pass

    def select_parent_generation(self, population, n):
        parent_generation = []

        while len(parent_generation) < n:
            parent_generation.append(self._select_parents(population.copy()))
        return parent_generation

    def _select_parents(self, population):
        pass



class RouletteSelection(SelectionMethod):
    def __init__(self):
        SelectionMethod.__init__(self)

    def _select_parents(self, population):
        parents = []

        for _ in (1, 2):
            total_fitness = sum([max(chromosome.fitness**2, 0) for chromosome in population])
            weights = [max(chromosome.fitness**2/total_fitness, 0) for chromosome in population]
            parent = choice(population, p=weights)
            parents.append(parent)
            population.remove(parent)
        return parents


class RankSelection(SelectionMethod):
    def __init__(self):
        SelectionMethod.__init__(self)

    def _select_parents(self, population):
        parents = []
        population = list(sorted(population, key=lambda chromosome: chromosome.fitness))
        pop_size = len(population)

        weights = []
        for i in range(pop_size):
            weights += [i]*i


        for _ in (1, 2):
            w_len = len(weights)
            index = randrange(0, w_len)
            parents.append(population[index])

            w_len = [i for i in weights if i != index]
            population.pop(index)
        return parents


class TournamentSelection(SelectionMethod):
    def __init__(self, tournament_size):
        SelectionMethod.__init__(self)
        self.tournament_size = tournament_size

    def _select_parents(self, population):
        population = sample(population, self.tournament_size)
        return list(
            sorted(population,
                   key=lambda chromosome: chromosome.fitness,
                   reverse=True)
        )[:2]