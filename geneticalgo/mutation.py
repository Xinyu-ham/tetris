from random import random, choice, sample

if __name__ == '__main__':
    from evolution import Chromosome
else:
    from .evolution import Chromosome


class MutationMethod:
    def __init__(self, rate, volume):
        self.rate = rate
        self.volume = volume

    def mutate(self, chromosome):
        pass


class NoisyMutation(MutationMethod):
    def __init__(self, rate, volume):
        MutationMethod.__init__(self, rate, volume)

    def mutate(self, chromosome):
        for i in range(chromosome.length):
            gene = chromosome.gene(i)
            if random() < self.rate:
                chromosome.inherit(self._mutate_single_gene(gene), i)

    def _mutate_single_gene(self, gene):
        signs = [1, -1]
        return (1 + choice(signs)*self.volume) * gene

class FlipMutation(MutationMethod):
    def __init__(self, rate, volume):
        MutationMethod.__init__(self, rate, volume)

    def mutate(self, chromosome):
        for i in range(chromosome.length):
            if random() < self.rate:
                chromosome.inherit(chromosome.gene[i] * -1, i)

class SwapMutation(MutationMethod):
    def __init__(self, rate, volume):
        MutationMethod.__init__(self, rate, volume)

    def mutate(self, chromosome):
        if random() < 2*self.rate:
            positions = sample(range(chromosome.length), 2)
            gene1, gene2 = (chromosome.gene(p) for p in positions)

            chromosome.inherit(gene1, positions[0])
            chromosome.inherit(gene2, positions[1])


