import cv2
import glob
import os
import numpy as np
import threading
import queue

train2label = {255: 0, 0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 17, 6: 19, 7: 20, 8: 21, 9: 22, 10: 23, 11: 24, 12: 25,
               13: 26, 14: 27,
               15: 28, 16: 31, 17: 32, 18: 33, 19: 255}


# Function to be vectorized
def map_func(val, dictionary):
    return dictionary[val] if val in dictionary else val


# Vectorize map_func
vfunc = np.vectorize(map_func)


def convert_img(img_dir, save_dir, num_thread=30):
    paths = list(glob.glob(os.path.join(img_dir, '*png')))
    imgp = queue.Queue()
    print(len(paths))
    for p in paths:
        imgp.put(p)
    print(imgp)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    def _worker():
        while True:
            try:
                imgpath = imgp.get(timeout=2)
                img = cv2.imread(imgpath, -1)
                img_name = imgpath.split('/')[-1]
                img = vfunc(img, train2label)
                print(os.path.join(save_dir, img_name))
                img = np.uint8(img)
                print(img.shape, img.dtype)
                cv2.imwrite(os.path.join(save_dir, img_name), img)
                # print(img_name, threading.current_thread().name)
            except queue.Empty:
                break

    t_list = []
    for i in range(num_thread):
        print(i)
        t = threading.Thread(target=_worker, name='Worker %s' % i, args=())
        t.start()
        t_list.append(t)
    print('ALL start')
    for t in t_list:
        t.join()


if __name__ == '__main__':
    convert_img(img_dir='/home/lih/project/semseg/exp/cityscapes/ohem_large/result/epoch_200/test/ss/gray',
                save_dir='/home/lih/project/semseg/exp/cityscapes/ohem_large/result/epoch_200/test/ss/out/')
