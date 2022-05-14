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
    def __init__(self, type):
        self.type = type
        self.color = color_list[type]
        # Location in y-axis relative to left of piece
        self.xloc = 4
        # Location in y-axis relative to bottom of piece
        self.yloc = 24

        self.orientation = 0
        self.shapes = []

    def get_shape(self):
        return self.shapes[self.orientation]

    def hit_box(self):
        return self.get_shape().shape

    def height(self):
        return self.hit_box()[0]

    def width(self):
        return self.hit_box()[1]

    def rotate(self, n=1):
        n_shapes = len(self.shapes)
        self.orientation = (self.orientation + n) % n_shapes

    def place(self, x, y):
        self.xloc = x
        self.yloc = y

    # Return an array the width of the block that
    # counts how many empty cells between the bottom of block to the cell with value
    # e.g. upside-down L will return (2, 0)
    def get_bottom_gap(self):
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
    def __init__(self):
        Tetromino.__init__(self, 'I')
        self.shapes = [
            self.color * np.ones([4, 1]),
            self.color * np.ones([1, 4])
        ]


class JTet(Tetromino):
    def __init__(self):
        Tetromino.__init__(self, 'J')

        shape0 = np.array([[1, 1], [0, 1], [0, 1]])
        self.shapes.append(self.color * shape0)
        for i in range(3):
            self.shapes.append(
                np.rot90(self.shapes[i])
            )


class LTet(Tetromino):
    def __init__(self):
        Tetromino.__init__(self, 'L')

        shape0 = np.array([[1, 1], [1, 0], [1, 0]])
        self.shapes.append(self.color * shape0)
        for i in range(3):
            self.shapes.append(
                np.rot90(self.shapes[i])
            )


class OTet(Tetromino):
    def __init__(self):
        Tetromino.__init__(self, 'O')
        self.shapes.append(self.color * np.ones([2, 2]))


class STet(Tetromino):
    def __init__(self):
        Tetromino.__init__(self, 'S')

        shape0 = np.array([[0, 1], [1, 1], [1, 0]])
        self.shapes.append(self.color * shape0)
        self.shapes.append(np.rot90(self.shapes[0]))

class TTet(Tetromino):
    def __init__(self):
        Tetromino.__init__(self, 'T')

        shape0 = np.array([[0, 1], [1, 1], [0, 1]])
        self.shapes.append(self.color * shape0)
        for i in range(3):
            self.shapes.append(
                np.rot90(self.shapes[i])
            )


class ZTet(Tetromino):
    def __init__(self):
        Tetromino.__init__(self, 'Z')

        shape0 = np.array([[1, 0], [1, 1], [0, 1]])
        self.shapes.append(self.color * shape0)
        self.shapes.append(np.rot90(self.shapes[0]))


class TetrominoBag():
    def __init__(self):
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
        return self.catalog[name]()

    def get_random(self):
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

