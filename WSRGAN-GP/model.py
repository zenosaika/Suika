# [WSRGAN-GP] - SRGAN using Wasserstien distance & Gradient Penalty
# Author : H3ART (owlmen2546@gmail.com)

import os
import sys
import math
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity


OUTPUT_DIM = 3
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
N_EPOCH = 1
N_CRITIC = 5
LAMBDA = 10
GAMMA = 1e-3
FIXED_SEED = 1729

INITIAL_EPOCH = 0
INITIAL_ITER = 0

THIS_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(THIS_DIR, 'train_output/')
CHECKPOINT_DIR = os.path.join(THIS_DIR, 'checkpoint/')
STATE_FILE_PATH = os.path.join(THIS_DIR, 'state_file')
PROGRESS_FILE_PATH = os.path.join(THIS_DIR, 'progress_file')
PARENT_DIR = os.path.join(THIS_DIR, os.pardir)
DATASETS_DIR = os.path.join(PARENT_DIR, 'datasets')


sys.path.append(PARENT_DIR)
import dataset_util # import dataset_util from the parent directory


class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = self.make_generator()

    def make_generator(self):
        inputs = keras.Input(shape=(None, None, 3))
        x = keras.layers.Conv2D(64, (9, 9), strides=(1, 1), padding='same', use_bias=False)(inputs)
        x = keras.layers.PReLU(shared_axes=[1, 2])(x)
        tmp = x

        for _ in range(8):
            x = self.residual_block(x)

        x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Add()([x, tmp])

        for _ in range(2):
            x = self.upsample_block(x)

        outputs = keras.layers.Conv2D(3, (9, 9), strides=(1, 1), padding='same', use_bias=False, activation='tanh')(x)
        return keras.Model(inputs, outputs)
        
    def residual_block(self, inputs):
        x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Add()([x, inputs])
        return outputs

    def upsample_block(self, inputs):
        x = keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
        x = tf.nn.depth_to_space(x, 2)
        outputs = keras.layers.PReLU(shared_axes=[1, 2])(x)
        return outputs

    def call(self, inputs):
        return self.model(inputs)
    

class Critic(keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = keras.Sequential()

        self.model.add(keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        self.model.add(keras.layers.LeakyReLU(0.2))

        n_filters = [64, 128, 128, 256, 256, 512, 512]
        n_strides = [2, 1, 2, 1, 2, 1, 2]
        for i in range(7):
            self.model.add(keras.layers.Conv2D(n_filters[i], (3, 3), strides=(n_strides[i], n_strides[i]), padding='same', use_bias=False))
            # self.model.add(keras.layers.BatchNormalization())
            self.model.add(keras.layers.LeakyReLU(0.2))

        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(1024))
        self.model.add(keras.layers.LeakyReLU(0.2))
        self.model.add(keras.layers.Dense(1, activation='linear'))

    def call(self, inputs):
        return self.model(inputs)
    

class WSRGAN_GP(keras.Model):
    def __init__(self, generator, critic):
        super(WSRGAN_GP, self).__init__()
        self.generator = generator
        self.critic = critic
        self.vgg19 = keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        self.vgg19.trainable = False
        
        for l in self.vgg19.layers:
            l.trainable = False

        self.features_extractor = keras.Model(inputs=self.vgg19.input, outputs=self.vgg19.get_layer('block5_conv4').output)

    def compile(self, gen_optimizer, crit_optimizer, ckpt_manager):
        self.gen_optimizer = gen_optimizer
        self.crit_optimizer = crit_optimizer
        self.ckpt_manager = ckpt_manager

    def preprocess_vgg(self, img):
        # convert back to [0, 255]
        img = (img + 1) * 127.5
        
        # RGB to BGR
        img = img[..., ::-1]
        
        # apply Imagenet preprocessing : BGR mean
        mean = [103.939, 116.778, 123.68]
        IMAGENET_MEAN = tf.constant(-np.array(mean))
        img = keras.backend.bias_add(img, keras.backend.cast(IMAGENET_MEAN, keras.backend.dtype(img)))
    
        return img

    def vgg_loss(self, hr, sr):
        features_sr = self.features_extractor(self.preprocess_vgg(sr))
        features_hr = self.features_extractor(self.preprocess_vgg(hr))
        
        return 0.006 * keras.backend.mean(keras.backend.square(features_sr - features_hr), axis=-1)
    
    def save_progress(self, progress):
        if not os.path.exists(PROGRESS_FILE_PATH):
            with open(PROGRESS_FILE_PATH, 'w') as progress_file:
                progress_file.write('iteration,epoch,generator_loss,critic_loss,mse,mean_psnr,mean_ssim,content_loss,elapsed_time')
                progress_file.write('\n')

        with open(PROGRESS_FILE_PATH, 'a') as progress_file:
            for each_record in progress:
                progress_file.write(each_record)
                progress_file.write('\n')

    def MSE(self, hr_img, sr_img):
        return np.mean((hr_img-sr_img) ** 2)

    def PSNR(self, mse):
        MAX_PIXEL = 255
        return 20 * math.log10(MAX_PIXEL / math.sqrt(mse))
        
    @tf.function
    def train_step(self, img_batchs):
        # train critic n_critic times
        crit_losses = []
        for img_batch in img_batchs:
            batch_size = img_batch.shape[0]
            lr = tf.image.resize(img_batch, [32, 32])
            epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)

            with tf.GradientTape() as crit_tape:
                with tf.GradientTape() as gp_tape:
                    sr = self.generator(lr, training=True)
                    interpolated_imgs = epsilon * img_batch + (1-epsilon) * sr
                    interpolated_imgs_pred = self.critic(interpolated_imgs, training=True)

                grads = gp_tape.gradient(interpolated_imgs_pred, interpolated_imgs)
                grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
                gradient_panelty = tf.reduce_mean(tf.square(grad_norms - 1))

                fx = self.critic(img_batch, training=True)
                fgz1 = self.critic(sr, training=True)
                crit_loss = tf.reduce_mean(fgz1) - tf.reduce_mean(fx) + LAMBDA * gradient_panelty
                crit_losses.append(crit_loss)

            crit_gradients = crit_tape.gradient(crit_loss, self.critic.trainable_variables)
            self.crit_optimizer.apply_gradients(zip(crit_gradients, self.critic.trainable_variables))

        # train generator 1 time
        img_batch = random.choice(img_batchs)
        lr = tf.image.resize(img_batch, [32, 32])
        with tf.GradientTape() as gen_tape:
            sr = self.generator(lr, training=True)
            fgz2 = self.critic(sr, training=True)
            content_loss = self.vgg_loss(img_batch, sr)
            gen_loss = tf.reduce_mean(content_loss) - GAMMA * tf.reduce_mean(fgz2)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        return gen_loss, crit_losses, fx, fgz1, fgz2, content_loss

    def train(self, dataset, n_epoch, n_critic, fixed_testset):
        fixed_testset_lr = tf.image.resize(fixed_testset, [32, 32])

        dataset = list(dataset)
        n_batch = len(dataset)
        n_iter = n_batch // n_critic

        for epoch in range(1, n_epoch+1):
            iter = 0
            progress = []
            for i in range(0, n_batch, n_critic):
                iter += 1
                img_batchs = dataset[i:i+n_critic]

                start_time = time.time()
                gen_loss, crit_losses, fx, fgz1, fgz2, content_loss = self.train_step(img_batchs)
                elapsed_time_in_second = time.time() - start_time

                # prepare ground truth (HR) and Super Resolution (SR) in range 0 - 255
                hr_img = (fixed_testset + 1) * 127.5
                sr_img = (self.generator(fixed_testset_lr, training=False) + 1) * 127.5

                # calculate Mean Square Error (MSE)
                mse = self.MSE(hr_img, sr_img)

                # calculate Mean Peak Signal to Noise Ratio (PSNR)
                mpsnr = 0

                # calculate Mean Structural Similarity (SSIM)
                mssim = 0

                n_fixed_testset = fixed_testset.shape[0]
                for i in range(n_fixed_testset):
                    # PSNR
                    ith_img_mse = self.MSE(hr_img[i], sr_img[i])
                    ith_img_psnr = self.PSNR(ith_img_mse)
                    mpsnr += ith_img_psnr

                    # SSIM
                    ith_img_ssim = structural_similarity(np.array(hr_img[i]), np.array(sr_img[i]), data_range=255, channel_axis=-1)
                    mssim += ith_img_ssim
                
                mpsnr /= n_fixed_testset
                mssim /= n_fixed_testset

                # calculate Content Loss (Perceptual Loss)
                content_loss = tf.reduce_mean(self.vgg_loss(hr_img, sr_img))

                # for each training step
                values_to_save = [
                    str(iter+INITIAL_ITER),
                    str(epoch+INITIAL_EPOCH),
                    f'{gen_loss:.4f}',
                    ' '.join([f'{loss:.4f}' for loss in crit_losses]),
                    f'{mse:.4f}',
                    f'{mpsnr:.4f}',
                    f'{mssim:.4f}',
                    f'{content_loss:.4f}',
                    f'{elapsed_time_in_second:4f}',
                ]
                progress.append(','.join(values_to_save))

                if iter == n_iter: # last iteration of each epoch
                    fx = tf.reduce_mean(fx)
                    fgz1 = tf.reduce_mean(fgz1)
                    fgz2 = tf.reduce_mean(fgz2)
                    content_loss = tf.reduce_mean(content_loss)

                    print(f'epoch {epoch+INITIAL_EPOCH}/{n_epoch+INITIAL_EPOCH} \t iteration {iter}/{n_iter} \t loss_g {gen_loss:.4f} \t loss_c {np.mean(crit_losses):.4f} \t fx {fx:.4f} \t fgz1 {fgz1:.4f} \t fgz2 {fgz2:.4f} \t content {content_loss:.4f}')
                    
                    if not os.path.exists(OUTPUT_DIR):
                        os.mkdir(OUTPUT_DIR)

                    plt.figure(figsize=(10, 10))
                    for a in range(36):
                        plt.subplot(6, 6, a+1)
                        plt.xticks([])
                        plt.yticks([])
                        plt.grid(False)
                        plt.imshow(sr_img[a].numpy().astype(np.uint8))
                    plt.savefig(OUTPUT_DIR + f'epoch{epoch+INITIAL_EPOCH}_iter{iter+INITIAL_ITER}.png', bbox_inches='tight')

                    plt.close('all')

            ## for each epoch
            # save progress file
            self.save_progress(progress)

            # save state file
            state_file = open(STATE_FILE_PATH, 'wb')
            pickle.dump({
                'LATEST_EPOCH': epoch+INITIAL_EPOCH,
                'LATEST_ITER': iter+INITIAL_ITER,
            }, state_file)
            state_file.close()

            # save weight checkpoint
            self.ckpt_manager.save()
