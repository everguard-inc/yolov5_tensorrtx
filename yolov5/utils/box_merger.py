import numpy as np
from typing import List


def not_in_harness_check(persons,buckets,model):
    buckets_boxes = buckets[:,:4]
    person_boxes = persons[:,:4]
    iou_matrix = iou_batch_numpy(buckets_boxes,person_boxes)
    iou_matrix = np.triu(iou_matrix,0)
    matched_indices = np.c_[(iou_matrix>0).nonzero()]
    person_not_in_harness = []
    for matched in matched_indices:
        b_i = matched[0]
        p_i = matched[1]
        bucket_box = buckets_boxes[b_i]
        person_box = person_boxes[p_i]
        bucket_box_width = bucket_box[2] - bucket_box[0]
        bucket_box_height = bucket_box[3] - bucket_box[1]
        person_box_width = person_box[2] - person_box[0]
        person_box_height = person_box[3] - person_box[1]
        bucket_box[0] =  (bucket_box[0] + bucket_box_width*0.05).astype(int)
        bucket_box[2] =  (bucket_box[2] - bucket_box_width*0.05).astype(int)
        bucket_area = bucket_box_width*bucket_box_height
        person_area = person_box_width*person_box_height
        person_centr = (person_box[0]+person_box[2])/2
        bucket_centr = (bucket_box[0]+bucket_box[2])/2
        bucket_person_width_union = max(bucket_box[2],person_box[2]) - min(bucket_box[0],person_box[0])
        centr_dist_norm = abs(bucket_centr - person_centr)/bucket_person_width_union
        head_loc = 1 if person_box[1] < bucket_box[1] else 0
        legs_loc = 1 if (person_box[3] < bucket_box[3]) and \
                    (person_box[3] > bucket_box[1]) else 0
        iou = bbox_iou_numpy(bucket_box,person_box)
        iom = bbox_io_min_numpy(bucket_box,person_box)
        ['area_ratio', 'height_ratio', 'legs_loc', 'head_loc', 'iou', 'io_min',
       'centr_x_distance_norm'],
        test = {
        'area_ratio':float(person_area/bucket_area).__round__(2),
        'height_ratio':float((person_box[3]-person_box[1])/(bucket_box[3]-bucket_box[1])).__round__(2),
        'legs_loc':legs_loc,
        'head_loc':head_loc,
        'iou':float(iou).__round__(2),
        'io_min':float(iom).__round__(2),
        'centr_x_distance_norm': float(centr_dist_norm).__round__(2)
        }
        prediction = model.predict([list(test.values())])[0]
        if prediction==1:
            if not any([(persons[p_i] == a_s).all() for a_s in person_not_in_harness]):
                person_not_in_harness.append(persons[p_i])
    person_not_in_harness = np.array(person_not_in_harness)

    return person_not_in_harness
        

def iou_batch_numpy(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    wh = w * h
    iou_matrix = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return iou_matrix


def predicts_to_multilabel_numpy(
    model, predicts : np.ndarray, iou_th : float, conf_th_list : List, bucket_label : int = 9, not_in_harness_label : int = 1
    ) -> np.ndarray:
    if len(predicts) == 0:
        return []
    filtered_predicts = []
    for pred in predicts:
        conf_th = conf_th_list[int(pred[-1])]
        if pred[-2]>=conf_th:
            filtered_predicts.append(pred)
    predicts = np.array(filtered_predicts).astype(int)
    if len(predicts) == 0:
        return []
    bucket_predicts = predicts[(predicts[...,-1]==bucket_label).nonzero()[0]].astype(int)
    infraction_predicts = predicts[(predicts[...,-1]==not_in_harness_label).nonzero()[0]].astype(int)
    predicts = predicts[(predicts[...,-1]!=bucket_label).nonzero()[0]].astype(int)
    predicts = predicts[(predicts[...,-1]!=not_in_harness_label).nonzero()[0]].astype(int)
    person_not_in_harness = not_in_harness_check(infraction_predicts,bucket_predicts,model)
    if len(person_not_in_harness)>0:
        predicts = np.append(predicts,person_not_in_harness,axis=0)
    iou_matrix = iou_batch_numpy(predicts,predicts)
    iou_matrix = np.triu(iou_matrix,0)
    matched_indices = np.c_[(iou_matrix>iou_th).nonzero()]
    new_matched_indices = []
    for ids in range(len(matched_indices)):
        if len(new_matched_indices)==0:
            new_matched_indices.append(list(set(matched_indices[ids])))
        else:
            exist_flag = False
            for exist_index, exist_el in enumerate(new_matched_indices):
                if matched_indices[ids][0] in exist_el or matched_indices[ids][1] in exist_el:
                    new_unique = list(set(exist_el+list(matched_indices[ids])))
                    new_matched_indices[exist_index] = new_unique
                    exist_flag = True
                    break  
            if not exist_flag:
                new_matched_indices.append(list(set(matched_indices[ids])))
    new_predicts = []
    for ids in new_matched_indices:
        new_pr = predicts[ids]
        new_pr = np.concatenate((new_pr[0][:4],new_pr[:,5]))
        new_predicts.append(np.array(new_pr))
    for pred in bucket_predicts:
        temp = []
        for i,el in enumerate(pred):
            if i!=4:
                temp.append(el)
        new_predicts.append(np.array(temp))
    return new_predicts


def bbox_iou_numpy(bb1, bb2):
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def bbox_io_min_numpy(bb1, bb2):
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    iom = intersection_area / min(bb1_area,bb2_area)
    assert iom >= 0.0
    assert iom <= 1.0
    return iom