import numpy as np

class BaseModel():
    def __init__(self):
        self.layers = {
            "input": None,
            "dense": [],
            "output": None
        }
        self.n_params = 0
        self.chromosome = None

    def get_fitness(self):
        pass


    def add_layer(self, layer, layer_type):
        if layer_type == 'dense':
            self.layers[layer_type].append(layer)
        else:
            self.layers[layer_type] = layer

    def add_layers(self, layers, initialize=True):
        assert len(layers) >= 2
        output_layer = layers.pop()
        self.add_layer(output_layer, 'output')
        input_layer = layers.pop(0)
        self.add_layer(input_layer, 'input')
        [self.add_layer(layer, 'dense') for layer in layers]

        if initialize:
            self.initialize_layers()

    def initialize_layers(self):
        prev_n_nodes = self.layers['input'].n_nodes
        for layer in self.layers['dense']:
            layer._initiate_params(prev_n_nodes)
            prev_n_nodes = layer.n_nodes
        self.layers['output']._initiate_params(prev_n_nodes)
        self.n_params = sum([layer.n_params for layer in self.layers['dense']]) + self.layers['output'].n_params

    def set_params(self, params):
        assert self.n_params == len(params)

        for layer in self.layers['dense']:
            layer._set_params(params[:layer.n_params])
            params = params[layer.n_params:]

        self.layers['output']._set_params(params)

    def get_score(self, features):
        for layer in self.layers['dense']:
            features = [np.dot(row, features) for row in layer.params]
        return np.dot(features, self.layers['output'].params[0])



class ModelLayer():
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.params = []
        self.n_params = 0

    def _initiate_params(self, prev_n_nodes):
        self.params = np.zeros([self.n_nodes, prev_n_nodes])
        self.n_params = self.n_nodes * prev_n_nodes

    def _set_params(self, params):
        assert self.n_params == len(params)
        self.params = params.reshape(self.params.shape)



