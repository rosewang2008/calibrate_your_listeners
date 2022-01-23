"""
Generate shapeworld reference games

Command:

python generate_shapeworld_data.py
"""

import numpy as np
from numpy import random
from enum import Enum
from tqdm import tqdm
import os
import multiprocessing as mp
from collections import namedtuple

from calibrate_your_listeners.src.datasets.shapeworld_objects import (
    Circle,
    Ellipse,
    Square,
    Rectangle,
    ShapeSpec,
    I
)


SHAPES = ['circle', 'square', 'rectangle', 'ellipse']
COLORS = ['red', 'blue', 'green', 'yellow', 'white', 'gray']
VOCAB = ['gray', 'shape', 'blue', 'square', 'circle', 'green', 'red', 'rectangle', 'yellow', 'ellipse', 'white']


MAX_PLACEMENT_ATTEMPTS = 5


SHAPE_IMPLS = {
    'circle': Circle,
    'ellipse': Ellipse,
    'square': Square,
    'rectangle': Rectangle,
    # TODO: Triangle, semicircle
}

def random_shape():
    return random.choice(SHAPES)


def random_color():
    return random.choice(COLORS)


def random_shape_from_spec(spec):
    color = None
    shape = None
    if spec == ShapeSpec.SHAPE:
        shape = random_shape()
    elif spec == ShapeSpec.COLOR:
        color = random_color()
    elif spec == ShapeSpec.BOTH:
        shape = random_shape()
        color = random_color()
    else:
        raise ValueError("Unknown spec {}".format(spec))
    return (color, shape)


SpatialConfig = namedtuple('SpatialConfig', ['shapes', 'distractors', 'relation', 'dir'])
SingleConfig = namedtuple('SingleConfig', ['shape', 'color'])


def random_config_single():
    shape_spec = ShapeSpec(random.randint(3))
    shape = random_shape_from_spec(shape_spec)
    return SingleConfig(*shape)


def random_config_spatial():
    # 0 -> only shape specified
    # 1 -> only color specified
    # 2 -> only both specified
    shape_1_spec = ShapeSpec(random.randint(3))
    shape_2_spec = ShapeSpec(random.randint(3))
    shape_1 = random_shape_from_spec(shape_1_spec)
    shape_2 = random_shape_from_spec(shape_2_spec)
    if shape_1 == shape_2:
        return random_config_spatial()
    relation = random.randint(2)
    relation_dir = random.randint(2)
    return SpatialConfig([shape_1, shape_2], None, relation, relation_dir)


def add_shape_from_spec(spec,
                        relation,
                        relation_dir,
                        shapes=None,
                        attempt=1):
    if attempt > MAX_PLACEMENT_ATTEMPTS:
        return None
    color, shape_ = spec
    if shape_ is None:
        shape_ = random_shape()
    shape = SHAPE_IMPLS[shape_](
        relation=relation, relation_dir=relation_dir, color=color)
    if shapes is not None:
        for oth in shapes:
            if shape.intersects(oth):
                return add_shape_from_spec(
                    spec,
                    relation,
                    relation_dir,
                    shapes=shapes,
                    attempt=attempt + 1)
        shapes.append(shape)
        return shape
    return shape


def add_shape_rel(spec, oth_shape, relation, relation_dir):
    """
    Add shape, obeying the relation/relation_dir w.r.t. oth shape
    """
    color, shape_ = spec
    if shape_ is None:
        shape_ = random_shape()
    if relation == 0:
        new_y = rand_pos()
        if relation_dir == 0:
            # Shape must be LEFT of oth shape
            new_x = random.randint(X_MIN, oth_shape.x - BUFFER)
        else:
            # Shape RIGHT of oth shape
            new_x = random.randint(oth_shape.x + BUFFER, X_MAX)
    else:
        new_x = rand_pos()
        if relation_dir == 0:
            # BELOW (remember y coords reversed)
            new_y = random.randint(oth_shape.y + BUFFER, X_MAX)
        else:
            # ABOVE
            new_y = random.randint(X_MIN, oth_shape.y - BUFFER)
    return SHAPE_IMPLS[shape_](x=new_x, y=new_y, color=color)


def new_color(existing_color):
    new_c = existing_color
    while new_c == existing_color:
        new_c = random.choice(COLORS)
    return new_c


def new_shape(existing_shape):
    new_s = existing_shape
    while new_s == existing_shape:
        new_s = random.choice(SHAPES)
    return new_s


def invalidate_spatial(config):
    # Invalidate by randomly choosing one property to change:
    ((shape_1_color, shape_1_shape),
     (shape_2_color,
      shape_2_shape)), extra_shape_specs, relation, relation_dir = config
    properties = []
    if shape_1_color is not None:
        properties.append(ConfigProps.SHAPE_1_COLOR)
    if shape_1_shape is not None:
        properties.append(ConfigProps.SHAPE_1_SHAPE)
    if shape_2_color is not None:
        properties.append(ConfigProps.SHAPE_2_COLOR)
    if shape_2_shape is not None:
        properties.append(ConfigProps.SHAPE_2_SHAPE)
    properties.append(ConfigProps.RELATION_DIR)
    # Randomly select property to invalidate
    # TODO: Support for invalidating multiple properties
    invalid_prop = random.choice(properties)

    if invalid_prop == ConfigProps.SHAPE_1_COLOR:
        return ((new_color(shape_1_color), shape_1_shape),
                (shape_2_color,
                 shape_2_shape)), extra_shape_specs, relation, relation_dir
    elif invalid_prop == ConfigProps.SHAPE_1_SHAPE:
        return ((shape_1_color, new_shape(shape_1_shape)),
                (shape_2_color,
                 shape_2_shape)), extra_shape_specs, relation, relation_dir
    elif invalid_prop == ConfigProps.SHAPE_2_COLOR:
        return ((shape_1_color, shape_1_shape),
                (new_color(shape_2_color),
                 shape_2_shape)), extra_shape_specs, relation, relation_dir
    elif invalid_prop == ConfigProps.SHAPE_2_SHAPE:
        return ((shape_1_color, shape_1_shape), (shape_2_color,
                                                 new_shape(shape_2_shape))
                ), extra_shape_specs, relation, relation_dir
    elif invalid_prop == ConfigProps.RELATION_DIR:
        return ((shape_1_color, shape_1_shape),
                (shape_2_color,
                 shape_2_shape)), extra_shape_specs, relation, 1 - relation_dir
    else:
        raise RuntimeError


def fmt_config(config):
    if isinstance(config, SingleConfig):
        return _fmt_config_single(config)
    elif isinstance(config, SpatialConfig):
        return _fmt_config_spatial(config)
    else:
        raise NotImplementedError(type(config))


def _fmt_config_single(config):
    color, shape = config
    shape_txt = 'shape'
    color_txt = ''
    if shape is not None:
        shape_txt = shape
    if color is not None:
        color_txt = color + ' '
    return '{}{}'.format(color_txt, shape_txt)


def _fmt_config_spatial(config):
    (s1, s2), extra, relation, relation_dir = config
    if relation == 0:
        if relation_dir == 0:
            rel_txt = 'left'
        else:
            rel_txt = 'right'
    else:
        if relation_dir == 0:
            rel_txt = 'below'
        else:
            rel_txt = 'above'
    if s1[0] is None:
        s1_0_txt = ''
    else:
        s1_0_txt = s1[0]
    if s1[1] is None:
        s1_1_txt = 'shape'
    else:
        s1_1_txt = s1[1]
    if s2[0] is None:
        s2_0_txt = ''
    else:
        s2_0_txt = s2[0]
    if s2[1] is None:
        s2_1_txt = 'shape'
    else:
        s2_1_txt = s2[1]
    return '{} {} {} {} {}'.format(s1_0_txt, s1_1_txt, rel_txt,
                                   s2_0_txt, s2_1_txt)


def generate_spatial(mp_args):
    """
    Generate a single image
    """
    random.seed()
    n_images, correct, i, data_type, context = mp_args
    # Get shapes and relations
    imgs = np.zeros((n_images, 64, 64, 3), dtype=np.uint8)
    labels = np.zeros((n_images, ), dtype=np.uint8)
    config = random_config_spatial()
    # Minimum of 2 correct worlds/2 distractors
    if data_type == 'concept':
        n_target = 2
        n_distract = 2
    else:
        n_target = 1
        n_distract = n_images  # Never run out of distractors
    idx_rand = list(range(n_images))
    # random.shuffle(idx_rand)
    for w_idx in idx_rand:
        if n_target > 0:
            label = 1
            n_target -= 1
        elif n_distract > 0:
            label = 0
            n_distract -= 1
        else:
            label = (random.random() < correct)
        new_config = config if label else invalidate_spatial(config)
        (ss1, ss2), extra_shape_specs, relation, relation_dir = new_config
        s2 = add_shape_from_spec(ss2, relation, relation_dir)

        attempts = 0
        while attempts < MAX_PLACEMENT_ATTEMPTS:
            # TODO: Support extra shapes
            s1 = add_shape_rel(ss1, s2, relation, relation_dir)
            if not s2.intersects(s1):
                break
        else:
            # Failed
            raise RuntimeError

        # Create image and draw shapes
        img = I()
        img.draw_shapes([s1, s2])
        imgs[w_idx] = img.array()
        labels[w_idx] = label
    return imgs, labels, config, i


def invalidate_single(config):
    color, shape_ = config
    if shape_ is not None and color is not None:
        # Sample random part to invalidate
        # Here, we can invalidate shape, or invalidate color, OR invalidate both
        part_to_invalidate = random.randint(3)
        if part_to_invalidate == 0:
            return (new_color(color), shape_)
        elif part_to_invalidate == 1:
            return (color, new_shape(shape_))
        elif part_to_invalidate == 2:
            return (new_color(color), new_shape(shape_))
        else:
            raise RuntimeError
    elif shape_ is not None:
        assert color is None
        return (None, new_shape(shape_))
    elif color is not None:
        assert shape_ is None
        return (new_color(color), None)
    else:
        raise RuntimeError


def generate_single(mp_args):
    random.seed()
    n_images, correct, i, data_type, context = mp_args
    imgs = np.zeros((n_images, 64, 64, 3), dtype=np.uint8)
    labels = np.zeros((n_images, ), dtype=np.uint8)
    config = random_config_single()
    if context != None:
        is_none = True
        while is_none:
            target_color, target_shape = config
            if target_color is None or target_shape is None:
                config = random_config_single()
            else:
                is_none = False
    if data_type == 'concept':
        n_target = 2
        n_distract = 2
    else:
        n_target = 1
        n_distract = n_images  # Never run out of distractors
    idx = list(range(n_images))
    shapes = []
    colors = []
    for w_idx in idx:
        if n_target > 0:
            label = 1
            n_target -= 1
        elif n_distract > 0:
            label = 0
            n_distract -= 1
        else:
            label = (random.random() < correct)
        new_config = config if label else invalidate_single(config)

        color_, shape_ = new_config
        if shape_ is None:
            shape_ = random_shape()

        if context != None:
            target_color, target_shape = config
            if label == 1:
                shape_ = target_shape
                color_ = target_color
            if label == 0:
                if context == 'shape':
                    shape_ = target_shape
                    same_color = True
                    while same_color:
                        color_ = random_color()
                        if color_ != target_color:
                            same_color = False
                if context == 'color':
                    color_ = target_color
                    same_shape = True
                    while same_shape:
                        shape_ = random_shape()
                        if shape_ != target_shape:
                            same_shape = False
                if context == 'both':
                    if w_idx == 1:
                        shape_ = target_shape
                        same_color = True
                        while same_color:
                            color_ = random_color()
                            if color_ != target_color:
                                same_color = False
                    elif w_idx == 2:
                        color_ = target_color
                        same_shape = True
                        while same_shape:
                            shape_ = random_shape()
                            if shape_ != target_shape:
                                same_shape = False
                if context == 'none':
                    same_color = True
                    while same_color:
                        color_ = random_color()
                        if color_ != target_color:
                            same_color = False
                    same_shape = True
                    while same_shape:
                        shape_ = random_shape()
                        if shape_ != target_shape:
                            same_shape = False
        else:
            # shape generalization - train
            """
            if shape_ == 'square':
                square = True
            else:
                square = False
            while square:
                shape_ = random_shape()
                if shape_ != 'square':
                    square = False"""
            # shape generalization - test
            """
            if label == 1:
                shape_ = 'square'"""

            # color generalization - train
            """
            if color_ == 'red':
                red = True
            else:
                red = False
            while red:
                color_ = random_color()
                if color_ != 'red':
                    red = False"""
            # color generalization - test
            """
            if label == 1:
                color_ = 'red'"""

            # combo generalization - train
            """
            if (color_ == 'red' and shape_ == 'circle') or (color_ == 'blue' and shape_ == 'square') or (color_ == 'green' and shape_ == 'rectangle') or (color_ == 'yellow' and shape_ == 'ellipse') or (color_ == 'white' and shape_ == 'circle') or (color_ == 'gray' and shape_ == 'square'):
                combo = True
            else:
                combo = False
            while combo:
                color_ = random_color()
                shape_ = random_shape()
                if not ((color_ == 'red' and shape_ == 'circle') or (color_ == 'blue' and shape_ == 'square') or (color_ == 'green' and shape_ == 'rectangle') or (color_ == 'yellow' and shape_ == 'ellipse') or (color_ == 'white' and shape_ == 'circle') or (color_ == 'gray' and shape_ == 'square')):
                    combo = False"""
            # combo generalization - test
            """
            if label == 1:
                combos = [('red','circle'),('blue','square'),('green','rectangle'),('yellow','ellipse'),('white','circle'),('gray','square')]
                combo = combos[np.random.randint(0,len(combos))]
                color_ = combo[0]
                shape_ = combo[1]"""

        shapes.append(shape_)
        colors.append(color_)
        shape = SHAPE_IMPLS[shape_](color=color_)

        # Create image and draw shape
        img = I()
        img.draw_shapes([shape])
        imgs[w_idx] = img.array()
        labels[w_idx] = label

    if colors.count(colors[0])==1 and shapes.count(shapes[0])==1:
        if np.random.randint(0,2) == 0:
            config = SingleConfig(colors[0],None)
        else:
            config = SingleConfig(None,shapes[0])
    elif colors.count(colors[0])==1:
        config = SingleConfig(colors[0],None)
    elif shapes.count(shapes[0])==1:
        config = SingleConfig(None,shapes[0])
    else:
        config = SingleConfig(colors[0],shapes[0])

    return imgs, labels, config, i


def generate(n,
             n_images,
             correct,
             data_type='concept',
             img_func=generate_spatial,
             float_type=False,
             n_cpu=None,
             pool=None,
             do_mp=True,
             verbose=False,
             context=None):
    if not do_mp and pool is not None:
        raise ValueError("Can't specify pool if do_mp=True")
    if do_mp:
        pool_was_none = False
        if pool is None:
            pool_was_none = True
            if n_cpu is None:
                n_cpu = mp.cpu_count()
            pool = mp.Pool(n_cpu)

    if data_type == 'concept':
        if n_images == 4:
            print("Warning: n_images == 4, min targets/distractors both 2, no variance")
        else:
            assert n_images > 4, "Too few n_images"
    elif data_type == 'reference':
        assert n_images > 1, "Too few n_images"
    else:
        raise NotImplementedError("data_type = {}".format(data_type))

    all_imgs = np.zeros((n, n_images, 64, 64, 3), dtype=np.uint8)
    all_labels = np.zeros((n, n_images), dtype=np.uint8)
    configs = []

    mp_args = [(n_images, correct, i, data_type, context) for i in range(n)]
    if do_mp:
        gen_iter = pool.imap(img_func, mp_args)
    else:
        gen_iter = map(img_func, mp_args)
    if verbose:
        gen_iter = tqdm(gen_iter, total=n)

    for imgs, labels, config, i in gen_iter:
        all_imgs[i, ] = imgs
        all_labels[i, ] = labels
        configs.append(config)
    if do_mp and pool_was_none:  # Remember to close the pool
        pool.close()
        pool.join()

    if float_type:
        all_imgs = np.divide(all_imgs, TWOFIVEFIVE)
        all_labels = all_labels.astype(np.float32)
    langs = np.array([fmt_config(c) for c in configs], dtype=np.unicode)
    return {'imgs': all_imgs, 'labels': all_labels, 'langs': langs}


def save_images(img_dir, data):
    # Save to test directory
    for instance_idx, (instance, instance_labels, *rest) in enumerate(data):
        for world_idx, (world, label) in enumerate(
                zip(instance, instance_labels)):
            Image.fromarray(world).save(
                os.path.join(img_dir, '{}_{}.png'.format(instance_idx, world_idx)))

    index_fname = os.path.join(img_dir, 'index.html')
    with open(index_fname, 'w') as f:
        # Sorry for this code
        f.write('''
            <!DOCTYPE html>
            <html>
            <head>
            <title>Shapeworld Fast</title>
            <style>
            img {{
                padding: 10px;
            }}
            img.yes {{
                background-color: green;
            }}
            img.no {{
                background-color: red;
            }}
            </style>
            </head>
            <body>
            {}
            </body>
            </html>
            '''.format(''.join(
            '<h1>{}</h1><p>{}</p>'.format(
                ' '.join(lang), ''.join(
                    '<img src="{}_{}.png" class="{}">'.format(
                        instance_idx, world_idx, 'yes' if label else 'no')
                    for world_idx, (
                        world,
                        label) in enumerate(zip(instance, instance_labels))))
            for instance_idx, (
                instance, instance_labels,
                lang, *rest) in enumerate(data))))
    np.savez_compressed('test.npz', imgs=data.imgs, labels=data.labels)


IMG_FUNCS = {
    'single': generate_single,
    'spatial': generate_spatial,
}

def _directory_check(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def run(config):
    # data_dir = './data/shapeworld/reference-1000-'
    data_dir = config.data_dir # os.path.join(constants.MAIN_REPO_DIR, config.data_dir)
    print(f"[ config ] Data directory at {data_dir}")
    _directory_check(data_dir)

    files = [f"{data_dir}/reference-1000-{idx}.npz" for idx in range(0, 75)]
    for f in files:
        data = generate(
            config.n_examples,
            config.n_images,
            config.p_correct,
            verbose=True,
            data_type=config.data_type,
            img_func=IMG_FUNCS[config.image_type],
            do_mp=config.multi_processing,
            context=None)
        np.savez_compressed(f, **data)

if __name__ == '__main__':
    run()

