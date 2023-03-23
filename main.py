from Rehoboam.trainer import train
from utilities.gif_utilities import generate_gif
from utilities.gpu_utilities import get_available_gpus

if __name__ == '__main__':
    get_available_gpus()
    train(30000, batch_size=64, save_interval=200)
    generate_gif()
