# [WSRGAN-GP] - SRGAN using Wasserstien distance & Gradient Penalty
# Author : H3ART (owlmen2546@gmail.com)

from model import *
from PIL import Image

INFER_INPUT_DIR = os.path.join(THIS_DIR, 'infer_input/')
INFER_OUTPUT_DIR = os.path.join(THIS_DIR, 'infer_output/')


generator = Generator()
critic = Critic()
wsrgan_gp = WSRGAN_GP(generator, critic)

# load weight
latest_cp = tf.train.latest_checkpoint(CHECKPOINT_DIR)
if latest_cp:
    print('load weight...')
    wsrgan_gp.load_weights(latest_cp).expect_partial()

# list all file in 'infer_input' folder
input_img_filenames = [f for f in os.listdir(INFER_INPUT_DIR) if f.lower().endswith(('.png', '.jpg', 'jpeg'))]

# 4x super resolution
print('super resolution...')
for img_filename in input_img_filenames:
    img = np.array(Image.open(INFER_INPUT_DIR+img_filename))
    img = img / 127.5 - 1 # convert to [-1, 1]
    sr = wsrgan_gp.generator(np.array([img]), training=False)[0]
    sr = (sr + 1) * 127.5 # convert back to [0, 255]
    keras.utils.save_img(INFER_OUTPUT_DIR+img_filename, sr, data_format='channels_last')
    print(img_filename + ' finished!')
