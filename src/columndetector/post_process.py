import numpy as np
from columndetector.utils import to_array, to_list
    
'''
パッチごとに予測したデータに対して後処理を行い、図面全体に対する予測に統合する。
'''
class PostProcess:
    def __init__(self, nms_iou_threshold=0.20, score_threshold=0.70):
        self.nms_iou_threshold = nms_iou_threshold
        self.score_threshold = score_threshold
    
    def main(self, patches):
        """[パッチごとに予測したデータに対して後処理を行い、図面全体に対する予測に統合する処理を行う]

        Args:
            patches ([list]): [図面の各パッチに対する柱の予測データ]

        Returns:
            [dict]: [prediction: 後処理後の予測データ]
        """        
        boxes = []
        scores = []
        labels = []
        for patch in patches:
            boxes_in_patch, scores_in_patch, labels_in_patch = self.post_process_to_patch(patch)
            if boxes_in_patch is None:
                continue
            boxes += boxes_in_patch
            scores += scores_in_patch
            labels += labels_in_patch
        try:  
            nms_filtered= {'boxes': boxes, 'scores' : scores, 'labels' : labels}
            nms_filtered = self.nms_method(nms_filtered, self.nms_iou_threshold)
            boxes, scores, labels = nms_filtered['boxes'], nms_filtered['scores'], nms_filtered['labels']
        except:
            pass
        
        prediction = {}
        prediction['boxes'] = boxes
        prediction['scores'] = scores
        prediction['labels'] = labels
        return prediction
        

    
    def post_process_to_patch(self, patch):
        """[パッチに対して後処理を行う]

        Args:
            patch ([dict]): [1つのパッチに対する予測データ]

        Returns:
            [list]: [boxes: 後処理後のパッチに対応する柱の矩形座標]
            [list]: [scores: 後処理後のパッチに対応する柱の信頼度]
            [list]: [labels: 後処理後のパッチに対応する柱のクラスラベル]
        """        
        window = patch['window']
        boxes = np.asarray(patch['boxes'])
        scores = np.asarray(patch['scores'])
        labels = np.asarray(patch['labels'])
        try:
            boxes, scores, labels = self.filter_by_scores(boxes, scores, labels, self.score_threshold)
        except:
            pass
        if len(boxes) == 0:
            return None, None, None
        
        boxes = self.window2entire(boxes, window)
        boxes = to_list(boxes)
        scores = to_list(scores)
        labels = to_list(labels)
        return boxes, scores, labels
    
    
    def filter_by_scores(self, boxes, scores, labels, score_threshold=0.70):
        """[信頼度(score)が低い柱を除外する処理を行う]

        Args:
            boxes ([np.ndarray]): [柱の矩形座標]
            scores ([np.ndarray]): [柱の信頼度]
            labels ([np.ndarray]): [柱のクラスラベル]
            score_threshold (float, optional): [信頼度の閾値]. Defaults to 0.70.

        Returns:
            [np.ndarray]: [boxes: 柱の矩形座標]
            [np.ndarray]: [scores: 柱の信頼度]
            [np.ndarray]: [labels: 柱のクラスラベル]
        """        
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        boxes = boxes[score_mask]
        labels = labels[score_mask]
        return boxes, scores, labels
    
    def window2entire(self, boxes, window):
        """[windowの左上を0とした座標から、全体の画像の左上を0とした座標に変換する]

        Args:
            lines ([list]): [直線のリスト]
            window ([list]): [ウィンドウの座標]

        Returns:
            [list]: [変換後の直線のリスト]
        """    
        w_start, h_start = window[:2]
        boxes = to_array(boxes).reshape(-1, 4) + np.array([[w_start, h_start, w_start, h_start]])
        boxes = to_list(boxes)
        return boxes
    
    
    def nms_method(self, entire_predictions, iou_thr=0.2):
        """[nms処理を行う]

        Args:
            entire_predictions ([dict]): [図面全体に対応する予測データ]
            iou_thr (float, optional): [IoUの閾値]. Defaults to 0.2.

        Returns:
            [dict]: [nms処理後の図面全体に対応する予測データ]
        """        
        boxes = np.array(entire_predictions['boxes'])
        labels = np.array(entire_predictions['labels'])
        scores = np.array(entire_predictions['scores'])
        unique_labels = np.unique(labels)
        final_boxes = []
        final_scores = []
        final_labels = []
        for l in unique_labels:
            condition = (labels == l)
            boxes_by_label = boxes[condition]
            scores_by_label = scores[condition]
            labels_by_label = np.array([l] * len(boxes_by_label))
            # Use faster function
            keep = self.nms_fast(boxes_by_label, scores_by_label, thresh=iou_thr)

            final_boxes.append(boxes_by_label[keep])
            final_scores.append(scores_by_label[keep])
            final_labels.append(labels_by_label[keep])
        final_boxes = np.concatenate(final_boxes)
        final_scores = np.concatenate(final_scores)
        final_labels = np.concatenate(final_labels)

        entire_predictions_apl_nms = {}
        entire_predictions_apl_nms['boxes'] = final_boxes
        entire_predictions_apl_nms['labels'] = final_labels
        entire_predictions_apl_nms['scores'] = final_scores
        return entire_predictions_apl_nms




    def nms_fast(self, dets, scores, thresh):
        """
        # It's different from original nms because we have float coordinates on range [0; 1]
        :param dets: numpy array of boxes with shape: (N, 5). Order: x1, y1, x2, y2, score. All variables in range [0; 1]
        :param thresh: IoU value for boxes
        :return: index of boxes to keep


        ovrは、iで抜き出したboxを除いた全てのbboxのIOUの結果。
        ovrのうち、重なりがあるboxが削除された結果がinds。indsはovr配列に対応する
        indsは、ovr配列に対応しているため、インデックス一つ分orderより小さいため、1を追加して、orderに対応させる。
        この処理で、orderから、iouが小さいboxを削除

        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep