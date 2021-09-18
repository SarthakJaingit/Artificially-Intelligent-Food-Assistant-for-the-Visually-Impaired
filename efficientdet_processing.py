import numpy as np
import torch
from PIL import Image
from non_max_suppression import jaccard_iou


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def simple_iou_thresh(out, iou_thresh):
    skipped_ind= list()
    for ii, pred in enumerate(out):
        iou = jaccard_iou(pred[:4].unsqueeze(0),out[:, :4])
        for jj, value in enumerate(iou[0][(ii + 1):]):
            if value > iou_thresh:
                skipped_ind.append(ii + jj + 1)

    skipped_ind, new_out, out = set(skipped_ind), list(), list(out)
    for i in range(len(out)):
        if i not in skipped_ind:
            new_out.append(out[i])

    new_out = torch.stack(new_out)
    return new_out

'''Many functions adopted from https://github.com/rwightman/efficientdet-pytorch'''

class ImageToNumpy:

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.moveaxis(np_img, 2, 0)  # HWC to CHW
        return np_img

def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR

def _size_tuple(size):
    if isinstance(size, int):
        return size, size
    else:
        assert len(size) == 2
        return size

class ResizePad:

    def __init__(self, target_size: int, interpolation: str = 'bilinear', fill_color: tuple = (0, 0, 0)):
        self.target_size = _size_tuple(target_size)
        self.interpolation = interpolation
        self.fill_color = fill_color

    def __call__(self, img):
        width, height = img.size

        img_scale_y = self.target_size[0] / height
        img_scale_x = self.target_size[1] / width
        img_scale = min(img_scale_y, img_scale_x)
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)

        new_img = Image.new("RGB", (self.target_size[1], self.target_size[0]), color=self.fill_color)
        interp_method = _pil_interp(self.interpolation)
        img = img.resize((scaled_w, scaled_h), interp_method)
        new_img.paste(img)  # pastes at 0,0 (upper-left corner)

        img_scale = 1. / img_scale  # back to original

        return new_img, img_scale

def resolve_fill_color(fill_color, img_mean=IMAGENET_DEFAULT_MEAN):
    if isinstance(fill_color, tuple):
        assert len(fill_color) == 3
        fill_color = fill_color
    else:
        try:
            int_color = int(fill_color)
            fill_color = (int_color,) * 3
        except ValueError:
            assert fill_color == 'mean'
            fill_color = tuple([int(round(255 * x)) for x in img_mean])
    return fill_color


def transforms_coco_eval(
        img,
        device, 
        img_size=512,
        interpolation='bilinear',
        use_prefetcher=False,
        fill_color='mean',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):

    fill_color = resolve_fill_color(fill_color, mean)

    transformed_img, scale = ResizePad(target_size=[img_size, img_size], interpolation=interpolation, fill_color=fill_color)(img)
    transformed_img = ImageToNumpy()(transformed_img)

    img_tensor = torch.zeros((1, *transformed_img.shape), dtype=torch.uint8)
    img_tensor[0] += torch.from_numpy(transformed_img)
    img_tensor = img_tensor.to(device)

    mean = torch.tensor([x * 255 for x in mean]).to(device).view(1, 3, 1, 1)
    std = torch.tensor([x * 255 for x in std]).to(device).view(1, 3, 1, 1)

    img_tensor = img_tensor.float().sub_(mean).div_(std)

    return img_tensor, scale
