import numpy as np

def nms(boxes, score, conf_thres, iou_thres= 0.25):

    # TODO
    tf = score[:] > conf_thres
    score = score[tf]

    idx = np.argsort(score)[::-1]
    keep = []

    while len(idx) > 0:

        keep.append(idx[0])
        # 需要
        overlap = np.zeros_like(idx[1:], dtype=np.float32)
        for i, j in enumerate(idx[1:]):
            bbox1 = boxes[idx[0]]
            bbox2 = boxes[j]
            out = iou(bbox1,bbox2)
            overlap[i] = out
        idx = idx[1:][overlap < conf_thres]
    return keep

def iou(bbox1,bbox2):
    x1, y1, w1, h1 = bbox1 # left up right bottom
    x2, y2, w2, h2 = bbox2
    if (x1 <= ((x2+ w2)/2) <= w1) or ((y1 <= ((y2+ h2)/2) <= h1)): # 必须加上等号
        left_top_x = max(x1, x2)
        left_top_y = max(y1, y2) 
        right_bottom_x = min(w1, w2)
        right_bottom_y = min(h1, h2)

        I = (right_bottom_x - left_top_x) * (right_bottom_y - left_top_y)
        o = (w1 - x1) * (h1- y1) + (w2 - x2) * (h2 - y2) - I

        IoU = I / o

        return IoU
    else: # 不相交
        return 0

if __name__ == '__main__':
    boxes = [
        [100, 100, 200, 200],
        [120, 110, 220, 210],
        [300, 320, 400, 400],
        [180, 100, 300, 180]
    ]
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    out = nms(boxes, scores, 0.5)
    print(out)