

# Comment on Interface

def get_label2index(model_out, classes):

  bad_spot_index = classes.index("Bad_Spots")

  label2index = dict()
  for ii, pred_label in enumerate(model_out[0]["labels"]):
    pred_label = int(pred_label)
    if pred_label != bad_spot_index or pred_label != 0:
      if pred_label in label2index:
        label2index[pred_label].append(ii)
      else:
        label2index[pred_label] = [ii]

  return label2index

def find_order_of_fruits(model_out, label2index):

  fruit_indices = list()
  for key, value in label2index.items():
    if key != 0 and key != 5:
      fruit_indices.extend(value)

  fruit_bboxes = model_out[0]["boxes"][fruit_indices]
  if len(fruit_bboxes) > 4:
    return None, ("Too Many fruits in view to give a descriptive analysis")

  ii2xmin = dict()
  ii2ymin = dict()
  for ii, bbox in enumerate(fruit_bboxes):
    other_bboxes = torch.cat((fruit_bboxes[:ii], fruit_bboxes[(ii + 1):]), 0)
    iou = jaccard_iou(bbox.unsqueeze(0), other_bboxes)
    if torch.sum(iou > 0.8):
      return None, ("Fruits are too close together to analyze. Please spread them out")
    else:
      ii2xmin[ii] = bbox[0]
      ii2ymin[ii] = bbox[1]

  hor, vert = list(ii2xmin.values()), list(ii2ymin.values())
  hor, vert = torch.stack(hor).detach().cpu().numpy(), torch.stack(vert).detach().cpu().numpy()

  chosen_rep = (ii2xmin if np.var(hor) > np.var(vert) else ii2ymin)
  return sorted(chosen_rep, key = chosen_rep.get), ("horizontal" if np.var(hor) > np.var(vert) else "vertical")

def find_bad_spot_pix_percent(model_out, fruit_order, orientation, classes):

  bad_spot_indices = np.setdiff1d(np.arange(len(model_out[0]["boxes"])), fruit_order)

  spoiled_percentages = list()
  for num in fruit_order:
    fruit_bbox = model_out[0]["boxes"][num]
    area = 0
    for bd_index in bad_spot_indices:
      bad_spot_bbox = model_out[0]["boxes"][bd_index]
      if (fruit_bbox[0] < bad_spot_bbox[0] and
          fruit_bbox[1] < bad_spot_bbox[1] and
          fruit_bbox[2] > bad_spot_bbox[2] and
          fruit_bbox[3] > bad_spot_bbox[3]):
        area += (bad_spot_bbox[2] - bad_spot_bbox[0]) * (bad_spot_bbox[3] - bad_spot_bbox[1])

    fruit_area = (fruit_bbox[2] - fruit_bbox[0]) * (fruit_bbox[3] - fruit_bbox[1])
    if area > fruit_area:
      area = fruit_area

    spoiled_percentages.append(int((area / fruit_area * 100)))

  voice_over = str()

  if len(fruit_order) == 1:
    fruit_name = classes[model_out[0]["labels"][fruit_order[0]]]
    voice_over = "The single {} has {}% of surface spoilage. ".format(
           fruit_name,
          spoiled_percentages[0]
      )
    return voice_over

  if orientation == "vertical":
    for ii, num in enumerate(fruit_order):
      fruit_name = classes[model_out[0]["labels"][num]]
      if ii == 0:
        voice_over = "The {} that is the top most fruit has {}% of surface spoilage. ".format(
            fruit_name,
            spoiled_percentages[ii]
        )
      else:
        voice_over += "The {} that is {} below the top most fruit has {}% of surface spoilage. ".format(
            fruit_name,
            ii,
            spoiled_percentages[ii]
        )
  else:
    for ii, num in enumerate(fruit_order):
      fruit_name = classes[model_out[0]["labels"][num]]
      if ii == 0:
        voice_over = "The {} that is the left most fruit has {}% of surface spoilage. ".format(
            fruit_name,
            spoiled_percentages[ii]
        )
      else:
        voice_over += "The {} that is {} right to the left most fruit has {}% of surface spoilage. ".format(
            fruit_name,
            ii,
            spoiled_percentages[ii]
        )

  return voice_over

def develop_voice_over(model_out, classes):

  if len(model_out[0]["labels"]) == 0:
    # Check if there are any predictions
    return None

  label2index = get_label2index(model_out, classes)
  fruit_order, orientation = find_order_of_fruits(model_out, label2index)

  if not fruit_order:
    # Check if you can find order
    return orientation

  voice_over = find_bad_spot_pix_percent(model_out, fruit_order, orientation, classes)
  return voice_over
