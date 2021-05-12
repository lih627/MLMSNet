from PIL import Image
import os
from queue import Queue, Empty
import threading


def combine_mask_and_img(img_path, mask_path, alpha=0.4):
    """
    Add RGB mask to img
    :param img_path: path to image
    :param mask_path: path to mask
    :return:
        Image
    """
    img = Image.open(img_path)
    mask = Image.open(mask_path)

    img_ = img.convert('RGBA')
    mask_ = mask.convert('RGBA')
    image = Image.blend(img_, mask_, alpha)
    return image


def demo_frames(img_root, file_path, pred_root, save_dir, alpha=0.4,
                num_worker=10):
    """
    Generate demo video frames from demo image frame and network prediction.
    :param img_root: root to demo frame
    :param file_path: cityscapes_demo.txt, which contains demo frame paths
    :param pred_root: root to netowrk prediction RGB mask.
    :param save_dir: result saved_dir
    :param alpha: for image blend
    :param num_worder: number of threads
    """
    img_paths = Queue()
    with open(file_path) as f:
        _paths = f.readlines()
        for p in _paths:
            p = os.path.join(img_root, p.rstrip())
            print('img path: {}'.format(p))
            img_name = p.split('/')[-1]
            mask_path = os.path.join(pred_root, img_name)
            assert os.path.isfile(p), p
            assert os.path.isfile(mask_path), mask_path

            img_paths.put(p)
    if not os.path.exists(save_dir):
        print('create dir {}'.format(save_dir))
        os.makedirs(save_dir)

    def _worker():
        print('thread {} begin.'.format(threading.current_thread().name))
        while True:
            try:
                img_path = img_paths.get(timeout=1)
                img_name = img_path.split('/')[-1]
                mask_path = os.path.join(pred_root, img_name)
                ret = combine_mask_and_img(img_path, mask_path, alpha)
                save_path = os.path.join(save_dir, img_name)
                ret.save(save_path)
                print('save {} down'.format(save_path))
            except Empty:
                break
        print('thread {} dowm.'.format(threading.current_thread().name))

    thread_list = []
    for i in range(num_worker):
        t = threading.Thread(target=_worker(), name='Worker {}'.format(i + 1))
        t.start()
        thread_list.append(t)
    for t in thread_list:
        t.join()
    return


def copy_and_rename(frame_path='../demo_frame', out_dir='../demo_out'):
    import shutil
    import glob
    frame = glob.glob(os.path.join(frame_path, '*.png'))
    frame.sort()
    for i, v in enumerate(frame):
        shutil.copyfile(v, os.path.join(out_dir, '{:06d}.png'.format(i)))


if __name__ == '__main__':
    img_root = '/Volumes/LiHaoDIsk/cityscapes/leftImg8bit_demoVideo/leftImg8bit'
    file_path = '../misc/cityscapes_demo.txt'
    pred_root = '/Users/lihao/Code/semseg/ret/demo/color'
    save_dir = '../demo_frame'
    # demo_frames(img_root, file_path, pred_root, save_dir)
    # copy_and_rename()
