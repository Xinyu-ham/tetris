import numpy as np
from random import randrange, sample

if __name__ == '__main__':
    from evolution import Chromosome
else:
    from .evolution import Chromosome


class CrossoverMethod:
    def __init__(self):
        pass

    def breed(self, parent1, parent2):
        assert parent1.length == parent2.length
        n_genes = parent1.length
        rule = self.get_rule(n_genes)

        child_chromosome = Chromosome(np.zeros(n_genes), parent1.population.base_model(), parent1.population)

        for i, v in enumerate(rule):
            if v:
                child_chromosome.inherit(parent1.gene(i), i)
            else:
                child_chromosome.inherit(parent2.gene(i), i)

        return child_chromosome

    def get_rule(self, n_genes):
        pass


class OnePointCrossover(CrossoverMethod):
    def __init__(self):
        CrossoverMethod.__init__(self)

    def get_rule(self, n_genes):
        threshold = randrange(0, n_genes)
        return [i < threshold for i in range(n_genes)]


class KPointCrossover(CrossoverMethod):
    def __init__(self, k):
        CrossoverMethod.__init__(self)
        self.k = k

    def get_rule(self, n_genes):
        thresholds = sample(range(n_genes), self.k)
        thresholds.sort()

        rule = [1] * n_genes
        for t in thresholds:
            rule = [rule[i] if i < t else rule[i]*-1 for i in range(n_genes)]
        return [r == 1 for r in rule]


class UniformCrossover(CrossoverMethod):
    def __init__(self):
        CrossoverMethod.__init__(self)

    def get_rule(self, n_genes):
        return [v > 0.5 for v in np.random.rand(n_genes)]
