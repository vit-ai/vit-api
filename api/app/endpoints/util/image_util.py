def filter_boxes(boxes):
    """Remove boxes that are completely enveloped by larger boxes."""
    filtered_boxes = []
    for box in boxes:
        is_enveloped = False
        for other_box in boxes:
            if (other_box[0] <= box[0] and other_box[1] <= box[1] and
                    other_box[2] >= box[2] and other_box[3] >= box[3]) and box != other_box:
                is_enveloped = True
                break
        if not is_enveloped:
            filtered_boxes.append(box)
    return filtered_boxes
