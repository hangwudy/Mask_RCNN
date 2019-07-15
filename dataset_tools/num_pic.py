import os


def loadim(image_path = '', ext = 'png', key_word = 'car_door'):
    image_list = []
    for filename in os.listdir(image_path):
        if filename.endswith(ext) and filename.find(key_word) != -1:
            current_path = os.path.abspath(image_path)
            image_abs_path = os.path.join(current_path,filename)
            image_list.append(image_abs_path)
    return image_list

for i in range(46,91):
    kw = "car_door_{}".format(i)
    im_list = loadim("/home/hangwu/Repositories/Dataset/dataset/car_door_all","jpg",kw)
    print(i, len(im_list))


for i in [50,55,62,69,78,87]:
    kw = "car_door_{}".format(i)
    for j in range(1,361):
        kw_2 = "{}_{}".format(kw,j)
        if len(loadim("/home/hangwu/Repositories/Dataset/dataset/car_door_all","jpg",kw_2)) == 0:
            print(i,j)


