# SRGAN using Wasserstien distance & Gradient Penalty

import os
import pickle
import random
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


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
OUTPUT_DIR = os.path.join(THIS_DIR, 'WSRGAN-GP', 'output/')
CHECKPOINT_DIR = os.path.join(THIS_DIR, 'WSRGAN-GP', 'checkpoint/')
STATE_FILE_PATH = os.path.join(THIS_DIR, 'WSRGAN-GP', 'state_file')
PROGRESS_FILE_PATH = os.path.join(THIS_DIR, 'WSRGAN-GP', 'progress_file')


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

    def compile(self, gen_optimizer, crit_optimizer):
        self.gen_optimizer = gen_optimizer
        self.crit_optimizer = crit_optimizer

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
                progress_file.write('iteration,epoch,generator_loss,critic_loss,psnr,ssim')
                progress_file.write('\n')

        with open(PROGRESS_FILE_PATH, 'a') as progress_file:
            for each_record in progress:
                progress_file.write(each_record)
                progress_file.write('\n')
        
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
        dataset = list(dataset)
        n_batch = len(dataset)
        n_iter = n_batch // n_critic

        for epoch in range(1, n_epoch+1):
            iter = 0
            progress = []
            for i in range(0, n_batch, n_critic):
                img_batchs = dataset[i:i+n_critic]
                gen_loss, crit_losses, fx, fgz1, fgz2, content_loss = self.train_step(img_batchs)
                
                # for each training step
                values_to_save = [
                    str(iter+INITIAL_ITER),
                    str(epoch+INITIAL_EPOCH),
                    f'{gen_loss:.4f}',
                    ' '.join([f'{loss:.4f}' for loss in crit_losses]),
                    '0.000', #psnr
                    '0.000', #ssim
                ]
                progress.append(','.join(values_to_save))

                if (iter%5 == 0) or (iter == n_iter-1):
                    fx = tf.reduce_mean(fx)
                    fgz1 = tf.reduce_mean(fgz1)
                    fgz2 = tf.reduce_mean(fgz2)
                    content_loss = tf.reduce_mean(content_loss)
                    print(f'epoch {epoch+INITIAL_EPOCH}/{n_epoch+INITIAL_EPOCH} \t iteration {iter+1}/{n_iter} \t loss_g {gen_loss:.4f} \t loss_c {crit_loss:.4f} \t fx {fx:.4f} \t fgz1 {fgz1:.4f} \t fgz2 {fgz2:.4f} \t content {content_loss:.4f}')
                    
                    # save image
                    generated_imgs = self.generator(fixed_testset, training=False)
                    generated_imgs = (generated_imgs + 1) * 127.5 # convert back to [0, 255]

                    plt.figure(figsize=(10, 10))
                    for a in range(36):
                        plt.subplot(6, 6, a+1)
                        plt.xticks([])
                        plt.yticks([])
                        plt.grid(False)
                        plt.imshow(generated_imgs[a].numpy().astype(np.uint8))
                    plt.savefig(OUTPUT_DIR + f'epoch{epoch+INITIAL_EPOCH}iteration{iter+1}.png', bbox_inches='tight')
                    plt.close('all')

                iter += 1

            # for each epoch
            self.save_progress(progress)
            self.save_weights(CHECKPOINT_DIR + f'wsrgan-gp_epoch{epoch+INITIAL_EPOCH}')

            # save state
            state_file = open(STATE_FILE_PATH, 'wb')
            pickle.dump({
                'LATEST_EPOCH': epoch+INITIAL_EPOCH,
                'LATEST_ITER': iter+INITIAL_ITER,
            }, state_file)
            state_file.close()


if __name__ == '__main__':

    # load state
    if os.path.exists(STATE_FILE_PATH):
        print('load state...')
        state_file = open(STATE_FILE_PATH, 'rb')
        state = pickle.load(state_file)
        if 'LATEST_EPOCH' in state:
            INITIAL_EPOCH = state['LATEST_EPOCH']
        if 'INITIAL_ITER' in state:
            INITIAL_ITER = state['INITIAL_ITER']
        state_file.close()

    generator = Generator()
    critic = Critic()
    wsrgan_gp = WSRGAN_GP(generator, critic)

    latest_cp = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_cp:
        print('load weight...')
        wsrgan_gp.load_weights(latest_cp)

    gen_optimizer = keras.optimizers.legacy.Adam(1e-4, beta_1=0.0, beta_2=0.9)
    crit_optimizer = keras.optimizers.legacy.Adam(1e-4, beta_1=0.0, beta_2=0.9)

    # load dataset & normalize to [-1, 1]
    print('load dataset...')
    dataset = keras.utils.image_dataset_from_directory(
        directory='datasets/crack/img',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    ).map(lambda imgs, _ : tf.cast(imgs, tf.float32) / 127.5 - 1)

    fixed_testset = keras.utils.image_dataset_from_directory(
        directory='datasets/crack/fixed_test',
        image_size=(64, 64),
        batch_size=36,
        shuffle=False # If set to False, sorts the data in alphanumeric order.
    ).map(lambda imgs, _ : tf.cast(imgs, tf.float32) / 127.5 - 1)
    fixed_testset = next(iter(fixed_testset)) # get first batch

    wsrgan_gp.compile(gen_optimizer, crit_optimizer)

    print('start training!')
    wsrgan_gp.train(dataset, N_EPOCH, N_CRITIC, fixed_testset)

    # tf.keras.utils.plot_model(generator.model, to_file='generator.png', show_shapes=True)
    # tf.keras.utils.plot_model(discriminator.model, to_file='discriminator.png', show_shapes=True)