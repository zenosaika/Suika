# [WSRGAN-GP] - SRGAN using Wasserstien distance & Gradient Penalty
# Author : H3ART (owlmen2546@gmail.com)

from model import *


device_name = tf.test.gpu_device_name()
print(f'[DEVICE]: {device_name if device_name else "CPU"}')


generator = Generator()
critic = Critic()
wsrgan_gp = WSRGAN_GP(generator, critic)

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

# load weight
latest_cp = tf.train.latest_checkpoint(CHECKPOINT_DIR)
if latest_cp:
    print('load weight...')
    wsrgan_gp.load_weights(latest_cp)

# if datasets not found, download from drive
if not os.path.exists(os.path.join(DATASETS_DIR, 'crack')):
    dataset_util.download('crack', DATASETS_DIR)

# load dataset & normalize to [-1, 1]
print('load dataset...')
dataset = keras.utils.image_dataset_from_directory(
    directory=os.path.join(DATASETS_DIR, 'crack/img'),
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
).map(lambda imgs, _ : tf.cast(imgs, tf.float32) / 127.5 - 1)

fixed_testset = keras.utils.image_dataset_from_directory(
    directory=os.path.join(DATASETS_DIR, 'crack/fixed_test'),
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=36,
    shuffle=False # If set to False, sorts the data in alphanumeric order.
).map(lambda imgs, _ : tf.cast(imgs, tf.float32) / 127.5 - 1)
fixed_testset = next(iter(fixed_testset)) # get first batch

gen_optimizer = keras.optimizers.legacy.Adam(1e-4, beta_1=0.0, beta_2=0.9)
crit_optimizer = keras.optimizers.legacy.Adam(1e-4, beta_1=0.0, beta_2=0.9)
wsrgan_gp.compile(gen_optimizer, crit_optimizer)

print('start training!')
wsrgan_gp.train(dataset, N_EPOCH, N_CRITIC, fixed_testset)
