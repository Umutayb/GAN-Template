from rehoboam.trainer import train
from utilities.gif_utilities import generate_gif
from utilities.gpu_utilities import get_available_gpus

if __name__ == '__main__':
    get_available_gpus()
    train(32000, batch_size=64, save_interval=100)
    generate_gif()
