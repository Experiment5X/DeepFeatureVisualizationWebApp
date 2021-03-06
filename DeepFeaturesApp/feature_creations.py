import cv2
import queue
import string
import random
import threading
import numpy as np
import keras.backend as K
from keras.applications import vgg16


images_to_make = queue.Queue()


def generate_image_name():
    random_string = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(25))
    return './DeepFeaturesApp/static/DeepFeaturesApp/images/' + random_string + '.png'


class ImageParameters:
    def __init__(self, image, image_name, learning_rate, layer_index, image_std_clip, grad_std_clip, epochs,
                 total_variation, noise_count):
        self.image = image
        self.image_name = image_name
        self.learning_rate = learning_rate
        self.layer_index = layer_index
        self.image_std_clip = image_std_clip
        self.grad_std_clip = grad_std_clip
        self.epochs = epochs
        self.total_variation = total_variation
        self.noise_count = noise_count


class AsyncImageFeatureCreator(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True

    def run(self):
        image_creator = ImageFeatureCreator()
        while True:
            image_instructions = images_to_make.get()
            feature_map = image_creator.get_feature_vector(image_instructions.image, image_instructions.layer_index)
            image = image_creator.create_from_features(feature_map, image_instructions.layer_index,
                                                       learning_rate=image_instructions.learning_rate,
                                                       grad_std_clip=image_instructions.grad_std_clip,
                                                       img_std_clip=image_instructions.image_std_clip,
                                                       epochs=image_instructions.epochs,
                                                       total_variation=image_instructions.total_variation,
                                                       noise_count=image_instructions.noise_count)

            cv2.imwrite(image_instructions.image_name, image)


class ImageFeatureCreator:
    def __init__(self):
        self.model = vgg16.VGG16(weights='imagenet', include_top=True)

    def get_feature_vector(self, image, feature_layer_index):
        input_img = self.model.input
        feature_vector = self.model.layers[feature_layer_index].output

        feature_vector_func = K.function([input_img], [feature_vector])
        return feature_vector_func([image.reshape(1, 224, 224, 3)])[0]

    def clip_by_std(self, x, deviations):
        x_min = np.mean(x) - deviations * np.std(x)
        x_max = np.mean(x) + deviations * np.std(x)

        return np.clip(x, x_min, x_max)

    def random_noise(self, count):
        noise = np.zeros((224, 224, 3))
        for i in range(0, count):
            noise[random.randint(0, 223), random.randint(0, 223), random.randint(0, 2)] = random.random() - 0.5

        return noise

    def create_from_features(self, features, feature_layer_index, learning_rate=0.1, grad_std_clip=1.5,
                             img_std_clip=3, epochs=250, total_variation=0, noise_count=0, verbose=False):

        # the total number of values in the
        feature_vector = self.model.layers[feature_layer_index].output
        value_count = K.cast(K.prod(K.shape(feature_vector)), dtype='float32')

        # from https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py
        def total_variation_loss(x, img_nrows=224, img_ncols=224):
            assert K.ndim(x) == 4
            if K.image_data_format() == 'channels_first':
                a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
                b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
            else:
                a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
                b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])

            # scale by the value count so the coefficent will be same for different sized layers
            return K.sum(K.pow(a + b, 1.25))

        input_img = self.model.input

        total_variation_node = total_variation_loss(input_img)
        loss = K.sum(K.abs(feature_vector - features)) / value_count + total_variation * total_variation_node

        gradient = K.gradients(loss, input_img)[0]
        iterate = K.function([input_img], [loss, gradient, total_variation_node])

        random_noise = self.random_noise(noise_count)

        start_image = np.random.rand(1, 224, 224, 3)
        for i in range(0, epochs):
            l, grad, tot_var = iterate([start_image])

            grad_clipped = self.clip_by_std(grad, grad_std_clip)
            grad_normalized = (grad_clipped - np.min(grad_clipped))
            grad_normalized /= np.max(grad_normalized)

            start_image += learning_rate * (-grad_normalized + random_noise)
            start_image = self.clip_by_std(start_image, img_std_clip)

            random_noise = self.random_noise(noise_count)

            # in the very beginning and at the very end don't blur the image
            if int(epochs * 0.07) < i < int(epochs * 0.93):
                if i % 5 == 0:
                    start_image = cv2.GaussianBlur(start_image.reshape(224, 224, 3), (3, 3), 0, 0).reshape(1, 224, 224,
                                                                                                           3)
                if i % 10 == 7:
                    start_image = cv2.GaussianBlur(start_image.reshape(224, 224, 3), (7, 7), 0, 0).reshape(1, 224, 224,
                                                                                                           3)

            # decay the learning rate
            if i == int(epochs * 0.8):
                i /= 10
            if i == int(epochs * 0.9):
                i /= 15

            # if verbose:
            # print('Epoch ' + str(i) + ': Loss = ' + str(l) + ': Total Variation: ')

        # normalize to between 0 and 255
        start_image -= start_image.min()
        start_image /= start_image.max()
        start_image *= 255

        return start_image.reshape(224, 224, 3)
