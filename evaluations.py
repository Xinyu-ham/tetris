import numpy as np

# Define a list of evaluation function to
# give a score to each action done by the bot


# Height of highest cell occupied
def max_height(field):
    heights = []
    for row in field.img.T:
        h = 0
        for i, cell in enumerate(row):
            if cell != 0:
                h = i + 1
        heights.append(h)
    return 1.5*max(heights)


def avg_height(field):
    heights = []
    for row in field.img.T:
        h = 0
        for i, cell in enumerate(row):
            if cell != 0:
                h = i + 1
        heights.append(h)
    return 1.5*np.mean(heights)


# Total number of empty cells between
# base and highest cell of the column
def n_gaps(field):
    empty = 0
    for row in field.img.T:
        h = 0
        for i, cell in enumerate(row):
            if cell != 0:
                h = i + 1
        empty += list(row)[:h].count(0)
    return -3*empty


# The number of lines cleared
def lines_cleared(field):
    return 2*field.score


# How far the shape of the field is from a rectangle
def evenness(field):
    heights = []
    for row in field.img.T:
        h = 0
        for i, cell in enumerate(row):
            if cell != 0:
                h = i + 1
        heights.append(h)
    max_h = max(heights)
    return -1*sum([max_h - h for h in heights])


def smoothness(field):
    heights = []
    for row in field.img.T:
        h = 0
        for i, cell in enumerate(row):
            if cell != 0:
                h = i + 1
        heights.append(h)
    return -10*np.std(heights)


def bumpiness(field):
    def get_height(column):
        h =0
        for i, cell in enumerate(column):
            if cell != 0:
                h = i + 1
        return h

    bump = []
    for i in range(field.width):
        h = get_height(field.img.T[i])
        neighbors = []
        if i > 0:
            neighbors.append(get_height(field.img.T[i-1]) - h)
        if i < field.width - 1:
            neighbors.append(get_height(field.img.T[i + 1]) - h)
        bump.append(min(neighbors))
    return -2*max(bump)


def spikiness(field):
    heights = []
    for row in field.img.T:
        h = 0
        for i, cell in enumerate(row):
            if cell != 0:
                h = i + 1
        heights.append(h)
    return -2*(max(heights) - min(heights))


# the number of sides in each row
# if place close to wall, there will be fewer one side
def row_sides(field):
    total_sides = 0
    for row in field.img:
        row_sides = 0
        prev = 1
        for cell in row:
            if cell == 0 and prev != 0:
                row_sides += 1
            if cell != 0 and prev == 0:
                row_sides += 1
        total_sides += row_sides
    return -1*total_sides



# If you lose
def death(field):
    return 0 if field.alive else -9999
