import numpy as np
from abc import ABCMeta, abstractstaticmethod

class BaseModel(metaclass=ABCMeta):
    """
    A model that takes in a series of values and outputs a score. It has a similar multi-layer architecture as a feedforward neural network, except the weights are determined using genetic algorithm instead of back propagation.

    If dense layers are empty, it will just be a simple linear model.

    Attributes
    ----------
    layers : dict(ModelLayer)
        A dictionary containing keys: input, output and dense. Input and output are ModelLayer instances whereas dense is a list of ModelLayer.
    n_param: int
        Total number of weights that needs to be optimized. 
    chromosome: evolution.Chromosome
        Chromosome instance that uses this model

    Methods
    ----------
    get_fitness()
        Calculate fitness function that is used to encourage reproduction in a Genetic Algorithm. Does nothing until specified in child class
    add_layer(layer, layer_type)
        Add a layer to the network, can be input, output or dense
    add_layers(layers, initialize=True)
        Add multiple layers to the network from input to output layer
    initialize_layers()
        Set initial values for each node in each layer
    set_params(params)
        Set specific values for each node in each layer
    get_score(feature)
        Compute output score by feeding values across layers
    """
    def __init__(self):
        """
        Parameters:
        ----------
        layers: dict()
            "input": ModelLayer
            "dense": list(ModelLayer)
            "output: ModelLayer
            Layers of the neural network infrastruture. If dense layer is empty, a simple linear model will be used.
        n_params: int
            The total number of weights that needs to be optimized.
        chromosome: evolution.Chromosome
            A single unit in the genetic algorithm. Each chromosome is tied to a BaseModel instance and will alter model weights throughout each iteration.
        """
        self.layers = {
            "input": None,
            "dense": [],
            "output": None
        }
        self.n_params = 0
        self.chromosome = None

    @abstractstaticmethod
    def get_fitness():
        """A function to calculate a fitness that represent the likelihood of a chromosome to reproduce.
        
        Fitness needs to be separately defined by user so that the genetic algorithm can optimize weights such that the fitness value is maximized.
        """
        


    def add_layer(self, layer, layer_type):
        """Add a layer of a specific layer_type to the model
        
        Parameters:
        ----------
        layer: ModelLayer
            A ModelLayer instance that contains nodes
        layer_type: str
            Can be 'input', 'dense' or 'output.'
        """
        if layer_type == 'dense':
            self.layers[layer_type].append(layer)
        else:
            self.layers[layer_type] = layer

    def add_layers(self, layers, initialize=True):
        """Add all layers in a list to the model. 
        
        Paramaters:
        ----------
        layers: list(ModelLayers)
            The layers should be ordered (input, dense(optional), dense(optional),..., output).
        Initialize: boolean
            Whether to initialize all weights with random values. Default is True. 

        Raises
        ----------
        AssertionError
            Fewer than 2 layers are in the layer input. Requires at least an input and output layer.
        """
        assert len(layers) >= 2
        output_layer = layers.pop()
        self.add_layer(output_layer, 'output')
        input_layer = layers.pop(0)
        self.add_layer(input_layer, 'input')
        [self.add_layer(layer, 'dense') for layer in layers]

        if initialize:
            self.initialize_layers()

    def initialize_layers(self):
        """Initialize all weights with random values. Also calculates the total number of weights needed and updates n_params parameter
        """
        prev_n_nodes = self.layers['input'].n_nodes
        for layer in self.layers['dense']:
            layer._initiate_params(prev_n_nodes)
            prev_n_nodes = layer.n_nodes
        self.layers['output']._initiate_params(prev_n_nodes)
        self.n_params = sum([layer.n_params for layer in self.layers['dense']]) + self.layers['output'].n_params

    def set_params(self, params):
        """Replace all weight values with input params

        Parameters:
        ----------
        params: list(float)
            List representing all the new weights to be set
        
        Raises:
        ----------
        AssertionError
            The number of weights in params does not match n_params
        """
        assert self.n_params == len(params)

        for layer in self.layers['dense']:
            layer._set_params(params[:layer.n_params])
            params = params[layer.n_params:]

        self.layers['output']._set_params(params)

    def get_score(self, features):
        """Calculate values for each node of each layer until output by multiplying previous node values/features with weights. Returns final score to the output layer
        
        Parameters:
        ----------
        features: list(float)
            Input features to be passed into the network
        """
        for layer in self.layers['dense']:
            features = [np.dot(row, features) for row in layer.params]
        return np.dot(features, self.layers['output'].params[0])



class ModelLayer():
    """
    Class that represents a layer in a BaseModel neural network. Each layer can be either an input, output or a hidden dense layer. Each layer needs to be added to the BaseModel

    Attributes:
    -----------
    n_nodes: int
        The number of nodes the layer contains. If layer is input then it must be the same as the number of features. If the layer is output the algorithm currently only supports single node.
    params: ndarray
        The 2d-array of weights from each of the previous nodes to current layer nodes. Shape = (current layer nodes, previous layer nodes)
    n_params: int
        Total number of weights this layer contains

    Methods:
    _initiate_params(prev_n_nodes)
        Set all weights to 0 and update n_params
    _set_param(params)
        Set weights to specific values
    ----------

    """
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



