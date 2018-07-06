import os
import numpy as np


from glob import glob
from skimage.io import imread, imread_collection


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = './health_images/'
TEST_PATH = './stage1_test/'


seed = 545
np.random.seed = seed



def get_image_ids():
    train_ids = next(os.walk(TRAIN_PATH))[1]
    test_ids = next(os.walk(TEST_PATH))[1]
    return train_ids, test_ids


# def load_image_path():
#     image_paths = []
#     for image_id in get_image_ids():
#         image_path = os.path.join(PATH, image_id, 'images', image_id + '.png')
#         image_paths.append(image_path)
#     return image_paths


# def plot_image():
#     for img_array in image_arrays():
#         plt.imshow(img_array)
#         plt.show()


# def get_mask_path():
#     image_masks_path = {}
#     for image_id in get_image_ids():
#         image_masks_path[image_id] = glob(
#             os.path.join(PATH, image_id, 'masks', '*'))
#     return image_masks_path


# def mask_arrays(image_id):
#     masks_arrays = []
#     for mask_path in get_mask_path()[image_id]:
#         mask = imread(mask_path)
#         masks_arrays.append(mask)
#     return masks_arrays

# def mask_collection(image_id):
#     masks_arrays = []
#     mask_list = get_mask_path()[image_id]
#     for images in mask_list:
#         masks = imread(images)
#         # print(masks.shape)
#         # masks = masks.reshape(256, 256, 3)
#         masks_arrays.append(masks)

#     # plt.imshow(labels)
#     # plt.show()
    
#     return masks_arrays 


# def image_arrays():
#     image_array = []
#     for image_paths in load_image_path():
#         img_id = image_paths.split('/')[2]
#         image, masks = imread(
#             image_paths), mask_collection(img_id) 
        
#         # print('-------------image---------------------')
#         # print(image)
#         # print('-------------masks---------------------')
#         # print(masks)
#         image_array.append((image, masks))
#     return image_array

# def get_mask_arrays():
#     masks = []
#     for image_ids in get_image_ids():
#         image_mask = mask_arrays(image_ids)
#         masks += image_mask
#     return masks


# def plot_mask():
#     for masks in get_mask_arrays():
#         plt.imshow(masks)
#         plt.show()


# # print(get_image_ids())
# # print(mask_collection('003cee89357d9fe13516167fd67b609a164651b21934585648c740d2c3d86dc1'))
