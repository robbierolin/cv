# import caffe
# import cv2
# import numpy as np
# import scipy.ndimage as nd
# from google.protobuf import text_format


# def preprocess(net, img):
#     return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
#
#
# def deprocess(net, img):
#     return np.dstack((img + net.transformer.mean['data'])[::-1])
#
#
# def objective_L2(dst):
#     dst.diff[:] = dst.data
#
#
# def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True, objective=objective_L2):
#     """Gradient ascent step."""
#     src = net.blobs['data']
#     dst = net.blobs[end]
#
#     ox, oy = np.random.randint(-jitter, jitter+1, 2)
#     src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2)  # apply jitter
#
#     net.forward(end=end)
#     objective(dst)
#     net.backward(start=end)
#     g = src.diff[0]
#     # Normalize.
#     src.data[:] += step_size / np.abs(g).mean() * g
#
#     src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)  # unshift
#
#     if clip:
#         bias = net.transformer.mean['data']
#         src.data[:] = np.clip(src.data, -bias, 255-bias)
#
#
# def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', clip=True, **step_params):
#     octaves = [preprocess(net, base_img)]
#     for i in range(octave_n-1):
#         octaves.append(nd.zoom(octaves[-1], (1, 1 / octave_scale, 1 / octave_scale), order=1))
#
#     src = net.blobs['data']
#     detail = np.zeros_like(octaves[-1])
#     for octave, octave_base in enumerate(octaves[::-1]):
#         h, w = octave_base.shape[-2:]
#         if octave > 0:
#             h1, w1 = detail.shape[-2:]
#             detail = nd.zoom(detail, (1, h / h1, w / w1), order=1)
#
#         src.reshape(1, 3, h, w)
#         src.data[0] = octave_base + detail
#         for i in range(iter_n):
#             make_step(net, end=end, clip=clip, **step_params)
#
#             vis = deprocess(net, src.data[0])
#             if not clip:
#                 vis = vis * (255 / np.percentile(vis, 99.98))
#             cv2.imshow('vis', vis)
#             print(octave)
#             print(i)
#             print(end)
#             print(vis.shape)
#
#         detail = src.data[0] - octave_base
#
#     return deprocess(net, src.data[0])
#
#
# def main():
#     model_path = '/Users/robbierolin/cv/deepdream/model'
#     net_fn = model_path + 'deploy.prototxt'
#     param_fn = model_path + 'bvlc_googlenet.caffemodel'
#
#     model = caffe.io.caffe_pb2.NetParameter()
#     text_format.Merge(open(net_fn).read(), model)
#     model.force_backward = True
#     open('tmp.prototxt', 'w').write(str(model))
#
#     net = caffe.Classifier('tmp.prototxt', param_fn,
#                            mean=np.float32([104.0, 116.0, 112.0]),
#                            channel_swap=(2, 1, 0))
#
#     image_path = '/Users/robbierolin/cv/deepdream/stealie.jpg'
#     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#
#     deepdream(net, img)
#
#
from functools import partial

import cv2
import numpy as np
import tensorflow as tf


def visstd(a, s=0.1):
    return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5


def save(path, img):
    save_img = np.uint8(np.clip(img, 0, 1) * 255)
    cv2.imwrite(path, save_img)


def render_naive(img0, t_obj, t_input, iter_n, sess, step):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]

    img = img0.copy()
    for i in range(iter_n):
        g, score = sess.run([t_grad, t_score], {t_input: img})
        g /= g.std() + 1e-8
        img += g * step
        print(score)

    return visstd(img)


def calc_grad_tiled(img, t_grad, sess, t_input, tile_size=512):
    """
    Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over
    multiple iterations.
    """
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def render_multiscale(img0, t_obj, t_input, iter_n, sess, step, octave_n, octave_scale):
    def tffunc(*argtypes):
        placeholders = list(map(tf.placeholder, argtypes))

        def wrap(f):
            out = f(*placeholders)

            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))

            return wrapper

        return wrap

    # Helper function that uses TF to resize an image
    def resize(img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0, :, :, :]

    resize = tffunc(np.float32, np.int32)(resize)

    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]

    img = img0.copy()
    for octave in range(octave_n):
        if octave > 0:
            hw = np.float32(img.shape[:2])*octave_scale
            img = resize(img, np.int32(hw))
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad, sess, t_input)
            g /= g.std() + 1e-8
            img += g*step

    return visstd(img)


def render_lapnorm(img0, t_obj, t_input, iter_n, sess, step, octave_n, octave_scale, lap_n):
    def tffunc(*argtypes):
        placeholders = list(map(tf.placeholder, argtypes))

        def wrap(f):
            out = f(*placeholders)

            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))

            return wrapper

        return wrap

    # Helper function that uses TF to resize an image
    def resize(img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0, :, :, :]

    resize = tffunc(np.float32, np.int32)(resize)

    k = np.float32([1, 4, 6, 4, 1])
    k = np.outer(k, k)
    k5x5 = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)

    def lap_split(img):
        with tf.name_scope('split'):
            lo = tf.nn.conv2d(img, k5x5, [1, 2, 2, 1], 'SAME')
            lo2 = tf.nn.conv2d_transpose(lo, k5x5 * 4, tf.shape(img), [1, 2, 2, 1])
            hi = img - lo2
        return lo, hi

    def lap_split_n(img, n):
        levels = []
        for i in range(n):
            img, hi = lap_split(img)
            levels.append(hi)
        levels.append(img)
        return levels[::-1]

    def lap_merge(levels):
        img = levels[0]
        for hi in levels[1:]:
            with tf.name_scope('merge'):
                img = tf.nn.conv2d_transpose(img, k5x5 * 4, tf.shape(hi), [1, 2, 2, 1]) + hi
        return img

    def normalize_std(img, eps=1e-10):
        with tf.name_scope('normalize'):
            std = tf.sqrt(tf.reduce_mean(tf.square(img)))
            return img / tf.maximum(std, eps)

    def lap_normalize(img, scale_n=4):
        img = tf.expand_dims(img, 0)
        tlevels = lap_split_n(img, scale_n)
        tlevels = list(map(normalize_std, tlevels))
        out = lap_merge(tlevels)
        return out[0, :, :, :]

    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))

    image = img0.copy()
    for octave in range(octave_n):
        if octave > 0:
            hw = np.float32(image.shape[:2]) * octave_scale
            image = resize(image, np.int32(hw))
        for i in range(iter_n):
            g = calc_grad_tiled(image, t_grad, sess, t_input)
            g = lap_norm_func(g)
            image += g*step

            print(i)

    return visstd(image)


def render_deepdream(img0, t_obj, t_input, iter_n, sess, step, octave_n, octave_scale):
    def tffunc(*argtypes):
        placeholders = list(map(tf.placeholder, argtypes))

        def wrap(f):
            out = f(*placeholders)

            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))

            return wrapper

        return wrap

    # Helper function that uses TF to resize an image
    def resize(img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0, :, :, :]

    resize = tffunc(np.float32, np.int32)(resize)

    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]

    img = img0.copy()
    octaves = []
    for i in range(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)

    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad, sess, t_input)
            img += g * (step / np.abs(g).mean() + 1e-7)
            print(i)

    return visstd(img)


def main():

    model_fn = '/Users/robbierolin/cv/deepdream/model/inception5h/tensorflow_inception_graph.pb'

    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    with tf.gfile.FastGFile(model_fn, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input')
    imagenet_mean = 117
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input': t_preprocessed})

    layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
    feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

    print('Number of layers: ', len(layers))
    print('Total number of feature channels ', sum(feature_nums))

    input_img = cv2.imread('stealie2.png', cv2.IMREAD_COLOR)
    input_img = cv2.resize(input_img, (0, 0), fx=1/6, fy=1/6)
    input_img = np.float32(input_img)
    # input_img += np.random.uniform(size=input_img.shape) + 100

    # Naive
    layer = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139
    img_noise = np.random.uniform(size=(244, 244, 3)) + 100
    iter_n = 20
    step = 1.0
    t_obj = graph.get_tensor_by_name('import/%s:0' % layer)[:,:,:,channel]

    # render_naive(img_noise, t_obj, t_input, iter_n, sess, step)

    octave_n = 3
    octave_scale = 1.4
    # render_multiscale(img_noise, t_obj, t_input, iter_n, sess, step, octave_n, octave_scale)

    lap_n = 4
    img = render_lapnorm(input_img, t_obj, t_input, iter_n, sess, step, octave_n, octave_scale, lap_n)

    # img = render_deepdream(input_img, t_obj, t_input, iter_n, sess, step, octave_n, octave_scale)
    save('laplacestealie.png', img)

if __name__ == '__main__':
    main()



