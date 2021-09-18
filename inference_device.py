import argparse
import torch
# pip install --upgrade torchvision (Run this after installing torch)
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import os
import time
import torch.nn.parallel
from contextlib import suppress
from non_max_suppression import calculate_iou_on_label, get_labels_categ
from interface import develop_voice_over
from efficientdet_processing import simple_iou_thresh, transforms_coco_eval
import cv2

from effdet import create_model
from effdet.data import resolve_input_config
from timm.models.layers import set_layer_config
from PIL import Image

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass


classes = ["Placeholder", "Apples", "Strawberry", "Peach", "Tomato", "Bad_Spots"]
# effdet_classes = ["Apples", "Strawberry", "Peach", "Tomato", "Apple_Bad_Spot",
#                   "Strawberry_Bad_Spot", "Tomato_Bad_Spot", "Peaches_Bad_Spot"]
COLORS = [(0, 0, 0), (0, 255, 0), (0, 0 , 255), (255, 255, 0), (255, 0, 0)]

''' Setting model device'''
def set_device(input_device):
    global device
    device = torch.device(input_device)
    print("Device: {}".format(input_device))

'''Intializing Model State Dicts'''
def create_effdet():

    model_args = dict()
    model_args['num_classes'] = 8
    model_args['pretrained'] = True
    model_args['checkpoint'] = "device/effecientdet_d0/effecientdet_d0_brain.pth.tar"
    model_args['redundant_bias'] = None
    model_args['model'] = 'efficientdet_d0'
    model_args['soft_nms'] = None
    model_args['use_ema'] = True
    model_args['img_size'] = 512
    model_args['torchscript'] = True


    model_args['pretrained'] = model_args['pretrained'] or not model_args['checkpoint']  # might as well try to validate something

    # create model
    with set_layer_config(scriptable=model_args['torchscript']):
        extra_args = dict(image_size=(model_args['img_size'] ,model_args['img_size']))
        bench = create_model(
            model_args['model'],
            bench_task='predict',
            num_classes=model_args['num_classes'],
            pretrained=model_args['pretrained'],
            redundant_bias=model_args['redundant_bias'],
            soft_nms=model_args['soft_nms'],
            checkpoint_path=model_args['checkpoint'],
            checkpoint_ema=model_args['use_ema'],
            **extra_args,
        )

    model_config = bench.config
    input_config = resolve_input_config(model_args, model_config)

    param_count = sum([m.numel() for m in bench.parameters()])
    print('Model %s created, param count: %d' % (model_args['model'], param_count))
    bench = bench.to(device)
    amp_autocast = suppress
    bench.eval()

    return bench, amp_autocast

def load_torchvision_models(model_name, map_loc):
    if model_name == "ssdlite_mobilenet":
        ssd_lite = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained_backbone=True, num_classes = len(classes))
        ssd_lite.load_state_dict(torch.load("device/ssdlite_mobilenet/ssdlite_mobilenet_brain.pth", map_location=map_loc))
        print("Loaded model weights for ssdlite_mobilenet turning to eval mode")

        return ssd_lite.eval()
    else:
        mobilenet_fasterrcnn = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        mobilenet_fasterrcnn.roi_heads.box_predictor.cls_score.out_features = len(classes)
        mobilenet_fasterrcnn.roi_heads.box_predictor.bbox_pred.out_features = 4 * (len(classes))

        mobilenet_fasterrcnn.load_state_dict(torch.load("device/mobilenet_fasterrcnn/mobilenet_fasterrcnn_brain.pth", map_location = map_loc))
        print("Loaded model weights for mobilenet_fasterrcnn turning to eval model")

        return mobilenet_fasterrcnn.eval()

'''Inference functions for all models'''
def infer_effdet(model, frame, amp_autocast, nms_thresh, voice_over):

    #Preprocessing steps for every frame
    transformed_frame, img_scale = transforms_coco_eval(frame, device, 512)
    with amp_autocast():
        output = model(transformed_frame)[0]
    final_out = list()
    for ii, pred in enumerate(output):
        #Nonmax Suppression
        if pred[-2] > float(nms_thresh):
            final_out.append(pred)
        else:
            break
    if len(final_out) != 0:
        final_out = torch.stack(final_out)
        if len(final_out) > 1:
             final_out = simple_iou_thresh(final_out, 0.2)
    else:
        final_out = []

    torch_image = F.to_tensor(frame)
    for ii, out in enumerate(final_out[:, -1]):
        if final_out[:, -1][ii] > 4:
            final_out[:, -1][ii] = 5.0
    written_image = draw_boxes(final_out[:, :4] * img_scale, final_out[:, -1].to(torch.uint8), torch_image, put_text= True)

    cv2.imshow('Output', written_image)
    # Fix or remove voice over
    if voice_over:
        results = [{
        "boxes" : final_out[:, :4] * img_scale,
        "labels": final_out[:, -1].to(torch.uint8),
        "scores": final_out[:, -2]
        }]
        print(results)
        voice_over = develop_voice_over(results, classes)
        print(voice_over)
    cv2.waitKey(0)


def infer_image(image, trained_model, distance_thresh, iou_thresh, voice_over):

    torch_image = F.to_tensor(image).unsqueeze(0).to(device)
    trained_model.to(device)
    trained_model.eval()
    print("Image Size: {}".format(torch_image.size()))

    start_time = time.time()
    results = trained_model(torch_image)
    end_time = time.time() - start_time

    print("Time of Inference {:0.2f}".format(end_time))

    valid_box_count = 0
    for ii, score in enumerate(results[0]["scores"]):
        if score < distance_thresh:
          low_index_start = ii
          break
        else:
          valid_box_count += 1

    if valid_box_count == len(results[0]["scores"]):
        low_index_start = len(results[0]["scores"])

    for key in results[0]:
        results[0][key] = results[0][key][:low_index_start]

    #This is where I place the order of the list
    fruit_spot_iou_thresh, bad_spot_iou_thresh = iou_thresh

    #Update when I get more data of fruits and when running for script beware of classes.
    bad_spot_index = [ii for ii, label in enumerate(results[0]["labels"]) if label in get_labels_categ(classes, "bad_spot")]
    fruit_index = [ii for ii, _ in enumerate(results[0]["labels"]) if ii not in bad_spot_index]

    bad_spot_results, fruit_results = dict(), dict()

    for key in results[0]:
        bad_spot_results[key], fruit_results[key] = results[0][key][[bad_spot_index]], results[0][key][[fruit_index]]

    assert len(bad_spot_results["boxes"]) == len(bad_spot_results["scores"]) == len(bad_spot_results["labels"])
    assert len(fruit_results["boxes"]) == len(fruit_results["scores"]) == len(fruit_results["labels"])

    len_of_bad_spots, len_of_fruit = len(bad_spot_results["boxes"]), len(fruit_results["boxes"])

    if len_of_bad_spots > 1:
        bad_spot_results = calculate_iou_on_label(bad_spot_results, len_of_bad_spots, bad_spot_iou_thresh, device)
    if len_of_fruit > 1:
        fruit_results = calculate_iou_on_label(fruit_results, len_of_fruit, fruit_spot_iou_thresh, device)

    for key in results[0]:
        if (key == "boxes"):
            results[0]["boxes"] = torch.cat((fruit_results["boxes"], bad_spot_results["boxes"]), axis = 0)
        else:
            results[0][key] = torch.cat((fruit_results[key], bad_spot_results[key]), dim = 0)

    if device == torch.device("cuda"):
        torch_image = torch_image.cpu()

    written_image = draw_boxes(results[0]["boxes"], results[0]["labels"], torch_image.squeeze(), put_text= True)

    cv2.imshow('Output', written_image)
    if voice_over:
        voice_over = develop_voice_over(results, classes)
        print(voice_over)

    cv2.waitKey(0)


def draw_boxes(boxes, labels, image, put_text = True):
    image = image.permute(1, 2, 0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for i, box in enumerate(boxes):
        color = COLORS[labels[i] % len(COLORS)]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        if put_text:
          cv2.putText(image, classes[labels[i]], (int(box[0]), int(box[1]-5)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                      lineType=cv2.LINE_AA)
    return image

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Infers on images and can give optional voice_over")
    parser.add_argument("image_file", type = str, help = "A file path to the image")
    parser.add_argument("--device", dest = "device", required = True, help = "name of device used")
    parser.add_argument("--model_name", dest = "model_name", required = True, help = "name of models: [efficientdet_d0, mobilenet_fasterrcnn, ssdlite_mobilenet]")
    parser.add_argument("--confidence_thresh", dest = "confidence_thresh", required = True, help = "value for confidence thresholding \
    in nms")
    parser.add_argument("--voice_over", dest = "voice_over", default = False, action='store_true', help = "choice to include user interface. Default is False")
    args = parser.parse_args()

    set_device(args.device)

    pil_image = Image.open(args.image_file).convert("RGB")

    if args.model_name == "efficientdet_d0":
        model, amp_autocast = create_effdet()
        infer_effdet(model, pil_image, amp_autocast, args.confidence_thresh, args.voice_over)
    elif args.model_name == "mobilenet_fasterrcnn" or args.model_name == "ssdlite_mobilenet":
        model = load_torchvision_models(args.model_name, device)
        infer_image(pil_image, model, float(args.confidence_thresh), [0.35, 0.1], args.voice_over)
    else:
        raise ValueError("model_name can only be [efficientdet_d0, mobilenet_fasterrcnn, ssdlite_mobilenet]")
