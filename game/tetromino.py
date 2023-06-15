import numpy as np
import random

color_list = {
    'I': 2,
    'J': 3,
    'L': 4,
    'O': 5,
    'S': 6,
    'T': 7,
    'Z': 8,
}


# Do you know that a piece of Tetris is called a Tetromino?
class Tetromino():
    """
    A class to represent the game pieces in a Tetris game. Each piece is called a tetromino and is made up of 4 blocks.

    The pieces are represented by the alphabets 'I', 'J' ,'L' ,'O' ,'S' ,'T' and 'Z.' 
    
    By default each tetromino will be placed in the middle above the finish line before dropping into the field.

    Attributes
    ----------
    type: str
        Name of tetromino represented by a letter
    color: int
        A number that represent an element of a matplotlib color-map
    xloc: int
        Horizontal position of the left-most block of the tetromino. Default as 4.
    yloc: int
        Vertical position of the left-most block of the tetromino. Default as 24.
    orientation: int
        Integer from 0 to 3 (inclusive) to represent the orientation of the tetromino
    shapes: list(ndarray)
        List of 2d-array representations of the shapes of tetromino correspond to each orientation

    Methods
    ----------
    get_shape()
        Get the 2d-array representation of blocks that shape the tetromino
    hit_box()
        Get the size of the smallest rectangle that contains the shape of the tetromino
    height()
        Get the the number of vertical blocks of hit-box
    width()
        Get the the number of horizontal blocks of hit-box
    rotate(n=1)
        Increase orientation by n, modulo by 4
    place(x, y)
        Set xloc and yloc of the tetromino
    """
    def __init__(self, type):
        """
        Parameters
        ----------
        type : str
            Accepts either 'I', 'J' ,'L' ,'O' ,'S' ,'T' or 'Z.' 
        color: int
            Integer that associate to a type of tetromino which represents a distinct color in a color-map in matplotlib. The color changes according to the color theme
        xloc: int
            Horizontal position of the left-most block, ranges from 0 to field.width
        yloc: int
            Vertical position of the lowest block, ranges from 0 to field.height
        orientation: int
            The number of 90-degree clockwise rotation from default position the tetromino is currently in
        shapes: list(ndarray)
            List of 2d-array representation of the shape of the tetromino correspond to each orientation 
        get_bottom_gap()
            Calculate the number of empty space between the bottom of the hit-box to the lowest block
        """
        self.type = type
        self.color = color_list[type]
        # Location in y-axis relative to left of piece
        self.xloc = 4
        # Location in y-axis relative to bottom of piece
        self.yloc = 24

        self.orientation = 0
        self.shapes = []

    def get_shape(self):
        """Return shape of the tetromino in a rectangular box represented in a 2d-array. Where 1 represents a block and 0 empty space.
        """
        return self.shapes[self.orientation]

    def hit_box(self):
        """Return the dimension of the smallest rectangle that contains the tetromino in its current orientation
        """
        return self.get_shape().shape

    def height(self):
        """Vertical dimension of the hit-box
        """
        return self.hit_box()[0]

    def width(self):
        """Horizontal dimension of the hit-box
        """
        return self.hit_box()[1]

    def rotate(self, n=1):
        """Increase the orientation by n % 4 times

        Parameters
        ----------
        n: int
            Number of 90-degree clockwise rotate to make to the tetromino
        """
        n_shapes = len(self.shapes)
        self.orientation = (self.orientation + n) % n_shapes

    def place(self, x, y):
        """Set new xloc and yloc location of the tetromino.

        Parameters
        ----------
        x: int
            New horiztonal postion of the left-most block. Ranges from 0 to field.width
        y: int
            New vertical postion of the lowest block. Ranges from 0 to field.height
        """
        self.xloc = x
        self.yloc = y

    def get_bottom_gap(self):
        """Return an array the width of the block that counts how many empty cells between the bottom of hit-box to the lowest solid block.

        e.g. upside-down L will return (2, 0), horizontal Z will return (1, 0, 0)
        """
        shape, h, w = self.get_shape(), self.height(), self.width()
        gaps = np.zeros(w)
        for i in range(h):
            for j in range(w):
                if shape[h - i - 1][j] == 0:
                    gaps[j] += 1
                else:
                    gaps[j] = 0
        return gaps





class ITet(Tetromino):
    """
    Class that represents the I-tetromino
    """
    def __init__(self):
        Tetromino.__init__(self, 'I')
        self.shapes = [
            self.color * np.ones([4, 1]),
            self.color * np.ones([1, 4])
        ]


class JTet(Tetromino):
    """
    Class that represents the J-tetromino
    """
    def __init__(self):
        Tetromino.__init__(self, 'J')

        shape0 = np.array([[1, 1], [0, 1], [0, 1]])
        self.shapes.append(self.color * shape0)
        for i in range(3):
            self.shapes.append(
                np.rot90(self.shapes[i])
            )


class LTet(Tetromino):
    """
    Class that represents the L-tetromino
    """
    def __init__(self):
        Tetromino.__init__(self, 'L')

        shape0 = np.array([[1, 1], [1, 0], [1, 0]])
        self.shapes.append(self.color * shape0)
        for i in range(3):
            self.shapes.append(
                np.rot90(self.shapes[i])
            )


class OTet(Tetromino):
    """
    Class that represents the O-tetromino
    """
    def __init__(self):
        Tetromino.__init__(self, 'O')
        self.shapes.append(self.color * np.ones([2, 2]))


class STet(Tetromino):
    """
    Class that represents the S-tetromino
    """
    def __init__(self):
        Tetromino.__init__(self, 'S')

        shape0 = np.array([[0, 1], [1, 1], [1, 0]])
        self.shapes.append(self.color * shape0)
        self.shapes.append(np.rot90(self.shapes[0]))

class TTet(Tetromino):
    """
    Class that represents the T-tetromino
    """
    def __init__(self):
        Tetromino.__init__(self, 'T')

        shape0 = np.array([[0, 1], [1, 1], [0, 1]])
        self.shapes.append(self.color * shape0)
        for i in range(3):
            self.shapes.append(
                np.rot90(self.shapes[i])
            )


class ZTet(Tetromino):
    """
    Class that represents the Z-tetromino
    """
    def __init__(self):
        Tetromino.__init__(self, 'Z')

        shape0 = np.array([[1, 0], [1, 1], [0, 1]])
        self.shapes.append(self.color * shape0)
        self.shapes.append(np.rot90(self.shapes[0]))


class TetrominoBag():
    """
    An object that randomly generates a tetromino

    Attributes
    ----------
    catalog: dict
        Dictionary that associates a letter to a tetromino object

    Methods
    ----------
    get_tet_by_name(name)
        Return a Tetromino based on the name
    get_random()
        Generate a random Tetromino instance
    """
    def __init__(self):
        """
        Parameter
        ----------
        catalog: dict
            Keys contains 'I', 'J' ,'L' ,'O' ,'S' ,'T' or 'Z' that gives the class representing the tetromino 
        """
        self.catalog = {
            'I': ITet,
            'J': JTet,
            'L': LTet,
            'O': OTet,
            'S': STet,
            'T': TTet,
            'Z': ZTet,
        }

    def get_tet_by_name(self, name):
        """Return a Tetromino instance correspond to the name
        
        Parameters:
        ----------
        name: str
            Accepts either 'I', 'J' ,'L' ,'O' ,'S' ,'T' or 'Z'
        """
        return self.catalog[name]()

    def get_random(self):
        """Randomly generates a Tetromino instance
        """
        return list(self.catalog.values())[random.randint(0,6)]()


if __name__ == '__main__':
    bag = TetrominoBag()
    jt = bag.get_random()
    print(jt.get_shape())
    print(jt.get_bottom_gap())
    jt.rotate()
    print(jt.get_shape())
    print(jt.get_bottom_gap())
    jt.rotate()
    print(jt.get_shape())
    print(jt.get_bottom_gap())

