from enum import Enum
from numpy import random
import numpy as np
from shapely.geometry import Point, box
from shapely import affinity
import aggdraw
from PIL import Image

DIM = 64
X_MIN, X_MAX = (8, 48)
ONE_QUARTER = (X_MAX - X_MIN) // 3
X_MIN_34, X_MAX_34 = (X_MIN + ONE_QUARTER, X_MAX - ONE_QUARTER)
BUFFER = 10
SIZE_MIN, SIZE_MAX = (3, 8)

TWOFIVEFIVE = np.float32(255)

COLORS = ['red', 'blue', 'green', 'yellow', 'white', 'gray']
BRUSHES = {c: aggdraw.Brush(c) for c in COLORS}
PENS = {c: aggdraw.Pen(c) for c in COLORS}

"""Objects used in generate_shapeworld_data.py"""

class I:
    def __init__(self):
        self.image = Image.new('RGB', (DIM, DIM))
        #  self.draw = ImageDraw.Draw(self.image)
        self.draw = aggdraw.Draw(self.image)

    def draw_shapes(self, shapes, flush=True):
        for shape in shapes:
            shape.draw(self)
        if flush:
            self.draw.flush()

    def show(self):
        self.image.show()
        #  self.image.resize((64, 64), Image.ANTIALIAS).show()

    def array(self):
        return np.array(self.image, dtype=np.uint8)

    def float_array(self):
        return np.divide(np.array(self.image), TWOFIVEFIVE)

    def save(self, path, filetype='PNG'):
        self.image.save(path, filetype)

def rand_size():
    return random.randint(SIZE_MIN, SIZE_MAX)


def rand_size_2():
    """Slightly bigger."""
    return random.randint(SIZE_MIN + 2, SIZE_MAX + 2)


def rand_pos():
    return random.randint(X_MIN, X_MAX)


class ShapeSpec(Enum):
    SHAPE = 0
    COLOR = 1
    BOTH = 2


class ConfigProps(Enum):
    SHAPE_1_COLOR = 0
    SHAPE_1_SHAPE = 1
    SHAPE_2_COLOR = 2
    SHAPE_2_SHAPE = 3
    RELATION_DIR = 4


SHAPE_SPECS = list(ShapeSpec)


class Shape:
    def __init__(self,
                 x=None,
                 y=None,
                 relation=None,
                 relation_dir=None,
                 color=None):
        if color is None:
            self.color = random.choice(COLORS)
        else:
            self.color = color
        if x is not None or y is not None:
            assert x is not None and y is not None
            assert relation is None and relation_dir is None
            self.x = x
            self.y = y
        elif relation is None and relation_dir is None:
            self.x = rand_pos()
            self.y = rand_pos()
        else:
            # Generate on 3/4 of image according to relation dir
            if relation == 0:
                # x matters - y is totally random
                self.y = rand_pos()
                if relation_dir == 0:
                    # Place right 3/4 of screen, so second shape
                    # can be placed LEFT
                    self.x = random.randint(X_MIN_34, X_MAX)
                else:
                    # Place left 3/4
                    self.x = random.randint(X_MIN, X_MAX_34)
            else:
                # y matters - x is totally random
                self.x = rand_pos()
                if relation_dir == 0:
                    # Place top 3/4 of screen, so second shape can be placed
                    # BELOW
                    # NOTE: Remember coords for y travel in opp dir
                    self.y = random.randint(X_MIN, X_MAX_34)
                else:
                    self.y = random.randint(X_MIN_34, X_MAX)
        self.init_shape()

    def draw(self, image):
        image.draw.polygon(self.coords, PENS[self.color])

    def intersects(self, oth):
        return self.shape.intersects(oth.shape)


class Ellipse(Shape):
    def init_shape(self, min_skew=1.5):
        self.dx = rand_size()
        # Dy must be at least 1.6x dx, to remove ambiguity with circle
        bigger = int(self.dx * min_skew)
        if bigger >= SIZE_MAX:
            smaller = int(self.dx / min_skew)
            assert smaller > SIZE_MIN, ("{} {}".format(smaller, self.dx))
            self.dy = random.randint(SIZE_MIN, smaller)
        else:
            self.dy = random.randint(bigger, SIZE_MAX)
        if random.random() < 0.5:
            # Switch dx, dy
            self.dx, self.dy = self.dy, self.dx

        shape = Point(self.x, self.y).buffer(1)
        shape = affinity.scale(shape, self.dx, self.dy)
        shape = affinity.rotate(shape, random.randint(360))
        self.shape = shape

        #  self.coords = [int(x) for x in self.shape.bounds]
        self.coords = np.round(np.array(self.shape.boundary).astype(np.int))
        #  print(len(np.array(self.shape.convex_hull)))
        #  print(len(np.array(self.shape.convex_hull.boundary)))
        #  print(len(np.array(self.shape.exterior)))
        self.coords = np.unique(self.coords, axis=0).flatten()


class Circle(Ellipse):
    def init_shape(self):
        self.r = rand_size()
        self.shape = Point(self.x, self.y).buffer(self.r)
        self.coords = [int(x) for x in self.shape.bounds]

    def draw(self, image):
        image.draw.ellipse(self.coords, BRUSHES[self.color])


class Rectangle(Shape):
    def init_shape(self, min_skew=1.5):
        self.dx = rand_size_2()
        bigger = int(self.dx * min_skew)
        if bigger >= SIZE_MAX:
            smaller = int(self.dx / min_skew)
            self.dy = random.randint(SIZE_MIN, smaller)
        else:
            self.dy = random.randint(bigger, SIZE_MAX)
        if random.random() < 0.5:
            # Switch dx, dy
            self.dx, self.dy = self.dy, self.dx

        shape = box(self.x, self.y, self.x + self.dx, self.y + self.dy)
        # Rotation
        shape = affinity.rotate(shape, random.randint(90))
        self.shape = shape

        # Get coords
        self.coords = np.round(
            np.array(self.shape.exterior.coords)[:-1].flatten()).astype(
                np.int).tolist()

    def draw(self, image):
        image.draw.polygon(self.coords, BRUSHES[self.color], PENS[self.color])


class Square(Rectangle):
    def init_shape(self):
        self.size = rand_size_2()
        shape = box(self.x, self.y, self.x + self.size, self.y + self.size)
        # Rotation
        shape = affinity.rotate(shape, random.randint(90))
        self.shape = shape

        # Get coords
        self.coords = np.round(
            np.array(self.shape.exterior.coords)[:-1].flatten()).astype(
                np.int).tolist()

