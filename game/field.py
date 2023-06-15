import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
if __name__ == '__main__':
    from tetromino import TetrominoBag
else:
    from .tetromino import TetrominoBag


plt.style.use('dark_background')


class Field:
    """
    A class to represent the field of a tetris game. A classic game of Tetris has a field size of 20 x 10 blocks

    Attributes
    ----------
    width : int
        width of the field in blocks
    height : int
        Maximum visible height of the field in blocks
    limit: int
        Maximum height pieces can stack up to before losing the game
    img: 2darray 
        A current snapshot of the the blocks in the game
    score: int
        The number of lines cleared by the player
    alive: boolean
        Whether the game can continue (max height of placed blocks < limit)
    tetromino_bag: TetrominoBag
        An object that generates a random Tetromino each turn
    evaluation_function: list(functions)
        A list of functions used to measure the fitness of a state in the game
    evaluations: list(float)
        A list of values after applying each respective evaluation function to the current state of the game

    Methods
    ----------
    place_tetromino(tetromino)
        Place the into the field based on the xloc and yloc position indicator of the tetromino if position is valid
    remove_tetromino(tetromino)
        Remove a tetromino that is currently in the field
    check_end_game()
        Check if the placed block height exceeds limit and updates alive status
    start_game(bot, verbose_speed=0)
        Start a game by inputting a player. Set verbose_speed > 0 to visualize each game.
    _ground_level(col):
        Return the lowest row with block placed given a column. 
    valid_range(tetromino, x)
        Check if the position of tetromino does not exceed width of the field
    drop_tetromino(tetromino, x)
        Calculate where tetromino will land at position x, and place it there
    check_clear_line()
        Check how many lines are being clear from dropping a tetromino
    display(ax, verbose_speed)
        Configure visuals of each game
    copy()
        Make a deep-copy of the field instance
    add_evaluation_function(function)
        Add a user defined function to track across the game to evaluation_functions
    update_evalutions()
        Calculate values using evaluation functions on the current state of the game
    play_turn()
        Play one turn by obtaining all evaluations for every possibile move and feed to the bot
    """
    def __init__(self):
        """
        Parameters
        ----------
        width : int
            width of the field in blocks, standard game size is 10
        height : int
            Maximum visible height of the field in blocks
        limit: int
            Maximum height pieces can stack up to before losing the game, standard game size is 20
        img: 2darray 
            A current snapshot of the the blocks in the game
        score: int
            The number of lines cleared by the player
        alive: boolean
            Whether the game can continue (max height of placed blocks < limit)
        tetromino_bag: TetrominoBag
            An object that generates a random Tetromino each turn
        evaluation_function: list(functions)
            A list of functions used to measure the fitness of a state in the game
        evaluations: list(float)
            A list of values after applying each respective evaluation function to the current state of the game
        """
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
        """Place tetromino in the playing field using the shape, orientation, xloc and yloc attributes in tetromino

        Parameters
        ----------
        tetromino: Tetromino
            An instance of a Tetromino
        
        Raises
        ----------
        AssertionError
            If some blocks of the tetromino falls outside the 20 x 10 field
        """
        shape, h, w = tetromino.get_shape()\
            , tetromino.height(), tetromino.width()

        x, y = tetromino.xloc, tetromino.yloc
        assert x >= 0 and x < 10 - w + 1
        self.img[y:y + h, x:x + w] += shape

    def remove_tetromino(self, tetromino):
        """Remove a tetromino from the field. This is only used to remove the display tetromino before it enters the actual playing field

        Parameters
        ----------
        tetromino: Tetromino
            An instance of a Tetromino
        
        Raises
        ----------
        AssertionError
            If the tetromino doesn't exist in the location
        """
        shape, h, w = tetromino.get_shape() \
            , tetromino.height(), tetromino.width()

        x, y = tetromino.xloc, tetromino.yloc
        assert (self.img[y:y + h, x:x + w] == shape).all()
        self.img[y:y + h, x:x + w] -= shape

    def check_end_game(self):
        """Set alive status to false if the height of the place blocks exceeds limit of 20"""
        if any([v != 0 for v in self.img[20]]):
            self.alive = False
        
    def start_game(self, bot, verbose_speed=0):
        """Input a bot that will play the game and start the game. The bot will make decision on which move to make each turn, drop the tetromino to the right spot in the right orientation, until the alive status becomes false. 
        
        Returns final score of the game.

        If verbose_speed is 0, then there will not be a display using matplotlib. Otherwise, shorter verbose_speed will display a more sped up game. 

        Parameters
        ----------
        bot: BaseModel
            An instance of BaseModel object that outputs a score given a state of the game
        verbose_speed: float, optional
            Then number of seconds pause between each move shown in the display (default is 0)
        """
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
        """Helper function to determine where the lowest block placed given a certain column.

        Parameters
        ----------
        col: int
        """
        column = self.img.T[col]
        level = len(column)
        # Count the number of empty blocks from the top of column
        for i in range(level):
            if column[-i-1] != 0:
                return level - i
        return 0


    def valid_range(self, tetromino, x):
        """Return true if the left-most block of tetromino is placed >= 0 and right-most block placed <= 20

        Parameters
        ----------
        tetromino: Tetromino
            An instance of a Tetromino
        x: int
            X position to place the left-most block of the tetromino
        """
        w = tetromino.width()
        return x >= 0 and x < self.width - w + 1

    def drop_tetromino(self, tetromino, x):
        """Given a certain horizontal position, x, drop the tetromino in the lowest unfilled position.

        Once dropped, the game will trigger line clearing, check for losing condition and update evaluations scores.

        Parameters
        ----------
        tetromino: Tetromino
            An instance of a Tetromino
        x: int
            X position to place the left-most block of the tetromino

        Raises
        ----------
        AssertionError
            If the tetromino doesn't exist in the location
        """
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
        """Check if there are any rows of blocks being fully field, if there are, remove the rows and append new empty rows on top.

        Also increase score by the number of lines cleared.
        """
        limit = self.limit
        for i in range(limit + 1):
            if all(v != 0 for v in self.img[limit - i]):
                self.img = np.delete(self.img, limit - i, axis=0)
                self.img = np.append(self.img, np.zeros([1, self.width]), axis=0)
                self.score += 1

    def display(self, ax, verbose_speed):
        """Display is toggled on using a matplotlib axes instance. The speed of the gameplay display is determined by verbose_speed

        Parameters
        ----------
        ax: matplotlib.Axes
            Axes instance returned by plt.gca
        verbose_speed: float
            The number, or fraction, of seconds between two frames of the animation
        """
        if not verbose_speed:
            return
        ax.imshow(self.img, origin='lower', cmap='CMRmap')
        ax.texts[-1].set_text(str(self.score))
        plt.pause(verbose_speed)

    def copy(self):
        """Create a deep copy of the entire field
        """
        return deepcopy(self)

    def add_evaluation_function(self, function):
        """Append an evaluation function to list which will be used to calculate the score given a state of a field

        Parameters
        ----------
        function: function
            A function that takes the field instance as the input and return a single value
        """
        self.evaluation_function.append(function)
        self.evaluations.append(0)

    def update_evaluations(self):
        """Run each evaluation function to the current state of the field and return a list of outputs
        """
        self.evaluations = np.array([function(self) for function in self.evaluation_function])

    def play_turn(self, bot, tet):
        """Create a copy of the field for every possible placement of a tetromino and calculate the evaluation score for each scenario.

        Bot will decide which outcome is the best and return the field that made the best move

        Parameters
        ----------
        bot: BaseModel
            A base model that contains weights to determine the best move to make
        tet: tetromino
            A tetromino of a specific shape
        """
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



