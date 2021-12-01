import imageio

dir_img = '/home/users/mireiffe/Documents/Python/TeethSeg/forpaper/ksiam_evolve_ex2/'

num_imgs = range(1, 150, 6)
images = []
for ni in num_imgs:
    images.append(imageio.imread(dir_img + f'ex{ni}.png'))
imageio.mimsave('/home/users/mireiffe/Documents/Python/TeethSeg/forpaper/ksiam_ex2.gif', images)