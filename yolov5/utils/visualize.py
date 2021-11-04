import cv2

def labels_dict_to_ints(labels : dict):
    post_labels = []
    for _,value in labels.items():
        if value:
            post_labels.append(value)
    
    return post_labels     


def plot_one_box(x, labels, img, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    label_names = ['in_harness', 'not_in_harness', 'harness_unrecognized', 'in_vest',\
        'not_in_vest','vest_unrecognized','in_hardhat','not_in_hardhat','hardhat_unrecognized','crane_bucket']
        
    labels = labels_dict_to_ints(labels)
    tl = (
        line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    box_color = (255,0,0)#color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, box_color, thickness=tl)
    labels_space = 0
    for label in labels:
        text_color = (0,255,0) if label==0 or label==3 or label==6 else (0,0,255)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label_names[label], 0, fontScale=tl / 6, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.putText(
            img,
            label_names[label],
            (c1[0], c1[1] - labels_space),
            0,
            tl / 4,
            text_color,
            thickness=tf,
            )
        labels_space += 20

    return img