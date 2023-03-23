import glob
import imageio.v2 as imageio

anim_file = 'dcgan.gif'


def generate_gif():
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('generated_images/*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
