# coding: utf-8

# created by Hang Wu on 2018.10.07
# feedback: h.wu@tum.de


import cv2
import numpy as np
from numpy import random
import os
import time
# Eigen
import load_image
import generate_dict


def overlap(background, foreground, bnd_pos, image_output_path, mask_output_path, mask_bw_output_path, car_door_subcat):
    background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
    # print(foreground.shape)
    # print(background.shape)

    rows, cols = foreground.shape[:2]
    rows_b, cols_b = background.shape[:2]

    # Mask initialization
    """# solid mask
    object_mask = np.zeros([rows_b, cols_b, 3], np.uint8)
    """
    # mask with window
    object_mask_with_window = np.zeros([rows_b, cols_b, 3], np.uint8)

    # mask with window in black white
    object_mask_with_window_bw = np.zeros([rows_b, cols_b], np.uint8)

    # Range of x and y
    low_x = bnd_pos['xmin']
    low_y = bnd_pos['ymin']
    high_x = bnd_pos['xmax']
    high_y = bnd_pos['ymax']
    # Movement for random position
    move_x = int(random.randint(- low_x, cols_b - high_x, 1))
    move_y = int(random.randint(- low_y, rows_b - high_y, 1))
    # move_y = random.randint(rows_b - high_y -1, rows_b - high_y, 1)

    # print('movement x:',move_x)
    # print(high_y)
    # print('movement y:',move_y)
    
    for i in range(rows):
        for j in range(cols):
            if foreground[i,j][3] != 0:
                # Overlap images
                try:
                    background[i + move_y, j + move_x] = foreground[i,j]
                except:
                    break
                if car_door_subcat == 1:
                # Mask generating (for nomal mask with window)
                    object_mask_with_window[i + move_y, j + move_x] = [0, 0, 255]
                elif car_door_subcat == 2:
                    object_mask_with_window[i + move_y, j + move_x] = [0, 255, 0]
                # Mask in black and white
                object_mask_with_window_bw[i + move_y, j + move_x] = 1

    output_image = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)

    save_name = "{}_cat_{}_id_{:0.0f}".format(bnd_pos['filename'][:-4], car_door_subcat, time.time()*1000)
    # Path
    image_output_name = "{}.jpg".format(save_name)
    image_output_dest = os.path.join(image_output_path, image_output_name)

    # Update xml data
    ## file info
    bnd_pos['folder'] = image_output_dest.split(os.path.sep)[-2]
    bnd_pos['filename'] = image_output_dest.split(os.path.sep)[-1]
    bnd_pos['path'] = image_output_dest
    ## image info
    rows_out, cols_out, channels_out = output_image.shape
    bnd_pos['width'] = cols_out
    bnd_pos['height'] = rows_out
    bnd_pos['depth'] = channels_out
    ## x-y value
    bnd_pos['xmin'] += move_x
    bnd_pos['ymin'] += move_y
    bnd_pos['xmax'] += move_x
    bnd_pos['ymax'] += move_y
    # test
    # print(bnd_pos)


    # Save images
    # Synthetized images
    cv2.imwrite(image_output_dest, output_image)

    # Masks
    mask_output_name = "{}.png".format(save_name)
    mask_output_dest = os.path.join(mask_output_path, mask_output_name)
    cv2.imwrite(mask_output_dest, object_mask_with_window)
    # Masks in black and white
    mask_bw_output_dest = os.path.join(mask_bw_output_path, mask_output_name)
    cv2.imwrite(mask_bw_output_dest, object_mask_with_window_bw)
    
    # Test
    # cv2.imwrite('images/{}.jpg'.format(save_name), output_image)
    # cv2.imwrite('masks/{}.png'.format(save_name), object_mask)
    # cv2.imwrite('masks_with_window/{}.png'.format(save_name), object_mask_with_window)
    
    # Display
    # cv2.imshow('{}.jpg'.format(save_name), output_image)
    # cv2.imshow('mask', object_mask)
    # cv2.imshow('mask with window', object_mask_with_window)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return bnd_pos


if __name__ == '__main__':
    
    fg_list = load_image.loadim('images')
    print(fg_list)
    bg_list = load_image.loadim('background','jpg','Fabrik')
    print(bg_list)
    for fg in fg_list:
        bnd_info = generate_dict.object_dict(fg, 1)
        fg = cv2.imread(fg, -1)
        bg_path = random.choice(bg_list)
        print(bg_path)
        bg = cv2.imread(bg_path, -1)
        overlap(bg, fg, bnd_info)
