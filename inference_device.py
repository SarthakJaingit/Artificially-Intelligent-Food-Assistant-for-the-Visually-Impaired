import argparse
import torch
import torchvision
import numpy as np
import os
import time
import torch.nn.parallel
from contextlib import suppress

from effdet import create_model
from effdet.data import resolve_input_config
from timm.models.layers import set_layer_config
from tqdm import tqdm
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

def set_device(input_device):
    global device
    device = torch.device(input_device)
    print("Device: {}".format(input_device))

def load_torchvision_models(model_name):
    if model_name == "ssdlite_mobilenet":
        ssd_lite = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained_backbone=True, num_classes = len(classes))
        ssd_lite.load_state_dict(torch.load("device/ssdlite_mobilenet/ssdlite_mobilenet_brain.pth"))
        print("Loaded model weights for ssdlite_mobilenet")

        return ssd_lite
    else:
        mobilenet_fasterrcnn = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        mobilenet_fasterrcnn.roi_heads.box_predictor.cls_score.out_features = len(classes)
        mobilenet_fasterrcnn.roi_heads.box_predictor.bbox_pred.out_features = 4 * (len(classes))

        mobilenet_fasterrcnn.load_state_dict(torch.load("device/mobilenet_fasterrcnn/mobilenet_fasterrcnn_brain.pth"))
        print("Loaded model weights for mobilenet_fasterrcnn")

        return mobilenet_fasterrcnn


def infer_image(image_file_path, trained_model, distance_thresh, iou_thresh, voice_over):


    torch_image = F.to_tensor(Image.open(image_file_path).convert("RGB")).unsqueeze(0).to(device)
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

    if show_image:
        if device == torch.device("cuda"):
            torch_image = torch_image.cpu()
        written_image = cv2.cvtColor(draw_boxes(results[0]["boxes"], results[0]["labels"], torch_image.squeeze(), infer = True, put_text= True), cv2.COLOR_BGR2RGB)
        plt.imshow(written_image)

    if voice_over:
        voice_over = develop_voice_over(results, classes)
        print(voice_over)

    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Infers on images and can give optional voice_over")
    parser.add_argument("image_file", type = str, help = "A file path to the image")
    parser.add_argument("--device", dest = "device", required = True, help = "name of device used")
    parser.add_argument("--model_name", dest = "model_name", required = True, help = "name of models: [efficientdet_d0, mobilenet_fasterrcnn, ssdlite_mobilenet]")
    parser.add_argument("--confidence_thresh", dest = "confidence_thresh", required = True, help = "value for confidence thresholding \
    in nms")
    parser.add_argument("--iou_thresh", dest = "iou_thresh", required = True, help = "value of iou thresh in nms")
    parser.add_argument("--voice_over", dest = "voice_over", required = False, help = "choice to include user interface. Default is False")
    args = parser.parse_args()

    set_device(args.device)

    if args.voice_over:
        voice_over = args.voice_over
    else:
        voice_over = False

    if args.model_name == "efficientdet_d0":
        results = stuff
        # Change this
    elif args.model_name == "mobilenet_fasterrcnn" or args.model_name = "ssdlite_mobilenet":
        model = load_torchvision_models(args.model_name)
    else:
        raise ValueError("model_name can only be [efficientdet_d0, mobilenet_fasterrcnn, ssdlite_mobilenet]")
