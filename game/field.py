import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
if __name__ == '__main__':
    from tetromino import TetrominoBag
else:
    from .tetromino import TetrominoBag

plt.style.use('dark_background')

class Field:
    def __init__(self):
        self.width = 10
        self.height = 30
        self.limit = 20
        self.img = np.zeros([self.height, self.width])
        self.score = 0
        self.alive = True
        self.tetromino_bag = TetrominoBag()
        self.evaluation_function = []
        self.evaluations = []

    def place_tetromino(self, tetromino):
        shape, h, w = tetromino.get_shape()\
            , tetromino.height(), tetromino.width()

        x, y = tetromino.xloc, tetromino.yloc
        assert x >= 0 and x < 10 - w + 1
        self.img[y:y + h, x:x + w] += shape

    def remove_tetromino(self, tetromino):
        shape, h, w = tetromino.get_shape() \
            , tetromino.height(), tetromino.width()

        x, y = tetromino.xloc, tetromino.yloc
        assert (self.img[y:y + h, x:x + w] == shape).all()
        self.img[y:y + h, x:x + w] -= shape

    def check_end_game(self):
        if any([v != 0 for v in self.img[20]]):
            self.alive = False

    def start_game(self, bot, verbose_speed=0):
        if verbose_speed:
            fg = plt.figure()
            ax = fg.gca()
            score_label = ax.text(0, 25, str(self.score))
            ax.axhline(y=20, color='r', linestyle='--')
        else:
            ax = None


        while self.alive:
            # Randomly generate a tetromino
            tet = self.tetromino_bag.get_random()
            # Display at top
            if verbose_speed:
                self.place_tetromino(tet)
                self.display(ax, verbose_speed)
                self.remove_tetromino(tet)

            x, rotation = self.play_turn(bot, tet)
            tet.rotate(rotation)
            self.drop_tetromino(tet, x)
            self.display(ax, verbose_speed)

        return self.score

    def _ground_level(self, col):
        column = self.img.T[col]
        level = len(column)
        for i in range(level):
            if column[-i-1] != 0:
                return level - i
        return 0

    def valid_range(self, tetromino, x):
        w = tetromino.width()
        return x >= 0 and x < self.width - w + 1

    def drop_tetromino(self, tetromino, x):
        shape, h, w = tetromino.get_shape()\
            , tetromino.height(), tetromino.width()
        assert self.valid_range(tetromino, x)

        tetromino.xloc = x

        heights = []
        for row in self.img.T[x: x + w]:
            max_h = 0
            for i, cell in enumerate(row):
                if cell != 0:
                    max_h = i
            heights.append(max_h)

        min_h = min(heights)
        for y in range(min_h, self.limit):
            area = self.img[y: y+h, x: x+w]
            if np.array([v == 0 for v in area*shape]).all():
                break

        tetromino.yloc = y

        self.place_tetromino(tetromino)
        self.check_clear_line()
        self.check_end_game()
        self.update_evaluations()

    def check_clear_line(self):
        limit = self.limit
        for i in range(limit + 1):
            if all(v != 0 for v in self.img[limit - i]):
                self.img = np.delete(self.img, limit - i, axis=0)
                self.img = np.append(self.img, np.zeros([1, self.width]), axis=0)
                self.score += 1

    def display(self, ax, verbose_speed):
        if not verbose_speed:
            return
        ax.imshow(self.img, origin='lower', cmap='CMRmap')
        ax.texts[-1].set_text(str(self.score))
        plt.pause(verbose_speed)

    def copy(self):
        return deepcopy(self)

    def add_evaluation_function(self, function):
        self.evaluation_function.append(function)
        self.evaluations.append(0)

    def update_evaluations(self):
        self.evaluations = np.array([function(self) for function in self.evaluation_function])

    def play_turn(self, bot, tet):
        possible_moves = []
        orientations = range(len(tet.shapes))
        for orientation in orientations:
            max_x = self.width - tet.shapes[orientation].shape[1] + 1
            for x in range(max_x):
                t = self.tetromino_bag\
                    .get_tet_by_name(tet.type)
                t.rotate(orientation)
                f = self.copy()
                f.drop_tetromino(t, x)

                features = f.evaluations - self.evaluations

                score = bot.model.get_score(features)
                # print(f'Shape = {t.type}; rotation = {orientation}, x = {x}, features = {features}, score = {score}')
                possible_moves.append((x, orientation, score))

        best_move = max(possible_moves, key=lambda x: x[2])
        return best_move[0], best_move[1]




if __name__ == '__main__':
    f = Field()
    bag = TetrominoBag()

    t1 = bag.get_tet_by_name('I')
    t1.rotate()
    f.drop_tetromino(t1, 0)
    plt.imshow(f.img, origin='lower')
    plt.pause(0.5)

    t2 = bag.get_tet_by_name('I')
    t2.rotate()
    f.drop_tetromino(t2, 4)
    plt.imshow(f.img, origin='lower')
    plt.pause(0.5)

    t3 = bag.get_tet_by_name('O')
    f.drop_tetromino(t3, 8)
    plt.imshow(f.img, origin='lower')
    plt.pause(0.5)

    f.drop_tetromino(t3, 8)
    plt.imshow(f.img, origin='lower')
    plt.pause(0.5)

    f.drop_tetromino(t3, 8)
    plt.imshow(f.img, origin='lower')
    plt.pause(0.5)


    print(f.score)



