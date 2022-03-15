import numpy as np
from girderbeamdetector.utils import to_array, to_list
    
'''
パッチごとに予測したデータに対して後処理を行い、図面全体に対する予測に統合する。
'''
class PostProcess:
    def __init__(self, nms_iou_threshold=0.20, breadth_threshold=0.70, score_threshold=0.70):
        self.nms_iou_threshold = nms_iou_threshold
        self.breadth_threshold = breadth_threshold
        self.score_threshold = score_threshold
    
    def main(self, patches):
        """[パッチごとに予測したデータに対して後処理を行い、図面全体に対する予測に統合する処理を行う]

        Args:
            patches ([list]): [図面の各パッチに対する梁の予測データ]

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
            boxes, scores, labels = self.concat_by_breadth(boxes, scores, labels, self.breadth_threshold)
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
            [list]: [boxes: 後処理後のパッチに対応する梁の矩形座標]
            [list]: [scores: 後処理後のパッチに対応する梁の信頼度]
            [list]: [labels: 後処理後のパッチに対応する梁のクラスラベル]
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
        try:
            nms_filtered= {'boxes': boxes, 'scores' : scores, 'labels' : labels}
            nms_filtered = self.nms_method(nms_filtered, self.nms_iou_threshold)
            boxes, scores, labels = nms_filtered['boxes'], nms_filtered['scores'], nms_filtered['labels']
        except:
            pass
        boxes = self.window2entire(boxes, window)
        boxes = to_list(boxes)
        scores = to_list(scores)
        labels = to_list(labels)
        return boxes, scores, labels
    
    
    def filter_by_scores(self, boxes, scores, labels, score_threshold=0.70):
        """[信頼度(score)が低い梁を除外する処理を行う]

        Args:
            boxes ([np.ndarray]): [梁の矩形座標]
            scores ([np.ndarray]): [梁の信頼度]
            labels ([np.ndarray]): [梁のクラスラベル]
            score_threshold (float, optional): [信頼度の閾値]. Defaults to 0.70.

        Returns:
            [np.ndarray]: [boxes: 梁の矩形座標]
            [np.ndarray]: [scores: 梁の信頼度]
            [np.ndarray]: [labels: 梁のクラスラベル]
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


    def filter_by_breadth(self, box, boxes, scores, breadth_threshold):
        """[梁の幅の重なり度が閾値以下の矩形座標を除外する]

        Args:
            boxes ([np.ndarray]): [梁の矩形座標]
            scores ([np.ndarray]): [梁の信頼度]
            labels ([np.ndarray]): [梁のクラスラベル]
            breadth_threshold ([type]): [幅の重なりの閾値]

        Returns:
            [np.ndarray]: [boxes: 梁の矩形座標]
            [np.ndarray]: [scores: 梁の信頼度]
        """        
        eps = 1e-5
        box_height = box[3] - box[1]
        box_width = box[2] - box[0]
        if box_height > box_width:
            direction = 0
        else:
            direction = 1
            
        boxes = to_array(boxes).reshape(-1, 4)
        scores = to_array(scores)
        boxes_height = boxes[:, 3] - boxes[:, 1]
        boxes_width = boxes[:, 2] - boxes[:, 0]
        if direction == 0:
            direction_mask = boxes_height > boxes_width
        else:
            direction_mask = boxes_height < boxes_width
        
        boxes = boxes[direction_mask]
        scores = scores[direction_mask]
        if len(boxes) == 0:
            return boxes, scores
        
        
        cliped_boxes = np.copy(boxes)
        if direction == 0:
            
            cliped_boxes[:, 0::2] = np.clip(boxes[:, 0::2], box[0], box[2])
            cliped_boxes_width = cliped_boxes[:, 2] - cliped_boxes[:, 0]
            breadth_rates = cliped_boxes_width / (box_width + eps)
            
        else:
            cliped_boxes[:, 1::2] = np.clip(boxes[:, 1::2], box[1], box[3])
            cliped_boxes_height = cliped_boxes[:, 3] - cliped_boxes[:, 1]
            breadth_rates = cliped_boxes_height / (box_height + eps)
            
        breadth_mask = breadth_rates > breadth_threshold
        boxes = boxes[breadth_mask]
        scores = scores[breadth_mask]
        return boxes, scores

    '''
    IoUがプラスで、梁の幅が閾値以上重なっていれば結合する。
    '''
    def concat_by_breadth(self, boxes, scores, labels, breadth_threshold):
        """[梁の幅の重なりが閾値以上であれば梁の矩形同士を結合する処理を行う]

        Args:
            boxes ([np.ndarray]): [梁の矩形座標]
            scores ([np.ndarray]): [梁の信頼度]
            labels ([np.ndarray]): [梁のクラスラベル]
            breadth_threshold ([type]): [幅の重なりの閾値]

        Returns:
            [np.ndarray]: [all_new_boxes: 結合後の梁の矩形座標]
            [np.ndarray]: [all_new_scores: 結合後の梁の信頼度]
            [np.ndarray]: [all_new_labels: 結合後の梁のクラスラベル]
        """        
        all_new_boxes = []
        all_new_scores = []
        all_new_labels = []
        labels = to_array(labels)
        scores = to_array(scores)
        boxes = to_array(boxes).reshape(-1, 4)
        unique_labels = np.unique(labels)
        for unique_label in unique_labels:
            label_mask = labels == unique_label
            boxes_i = boxes[label_mask].tolist()
            scores_i = scores[label_mask].tolist()
            
            new_boxes = []
            new_scores = []
            while len(boxes_i) != 0:
                box = boxes_i.pop(0)
                score = scores_i.pop(0)
                boxes_i_cpy = np.copy(boxes_i)
                ious = self.IoUs(box, boxes_i_cpy)
                iou_mask = ious > 0
                boxes_i_iou_masked = np.array(boxes_i)[iou_mask]
                scores_i_iou_masked = np.array(scores_i)[iou_mask]
                boxes_fill_by_breadth, scores_fill_by_breadth = self.filter_by_breadth(box, boxes_i_iou_masked, scores_i_iou_masked, breadth_threshold=0.70)
                if len(boxes_fill_by_breadth) != 0:
                    # 結合
                    concated_boxes = [box]
                    concated_scores = [score]
                    concated_boxes += to_list(boxes_fill_by_breadth)
                    concated_scores += to_list(scores_fill_by_breadth)
                    concated_box = self.concat_boxes(concated_boxes)
                    concated_score = max(concated_scores)
                    boxes_i, scores_i = self.del_items(boxes_i, scores_i, concated_boxes)
                    boxes_i.append(concated_box)
                    scores_i.append(concated_score)
                else:
                    # 非結合
                    new_boxes.append(box)
                    new_scores.append(score)
                    boxes_i, scores_i = self.del_items(boxes_i, scores_i, [box])
                
            new_labels = np.ones(len(new_boxes)) * unique_label
            new_labels = to_list(new_labels)
            all_new_boxes += new_boxes
            all_new_scores += new_scores
            all_new_labels += new_labels
             
        return all_new_boxes, all_new_scores, all_new_labels

    def del_items(self, deled_boxes, deled_scores, subject_boxes):
        """[指定した矩形座標を削除する処理を行う]

        Args:
            deled_boxes ([np.ndarray]): [削除対象となる矩形座標の配列]
            deled_scores ([np.ndarray]): [削除対象となる矩形座標の信頼度]
            subject_boxes ([np.ndarray]): [削除する矩形座標]

        Returns:
            [list]: [deled_boxes: 削除後の矩形座標の配列]
            [list]: [deled_scores: 削除後の矩形座標の信頼度]
        """        
        deled_boxes = to_array(deled_boxes).reshape(-1, 4).tolist()
        subject_boxes = to_array(subject_boxes).reshape(-1, 4).tolist()
        del_idxes = []
        for subject_box in subject_boxes:
            if subject_box in deled_boxes:
                del_idx = deled_boxes.index(subject_box)
            else:
                continue
            del_idxes.append(del_idx)
            
        deled_boxes = np.delete(deled_boxes, del_idxes, 0)
        deled_scores = np.delete(deled_scores, del_idxes)
        deled_boxes = to_list(deled_boxes)
        deled_scores = to_list(deled_scores)
        return deled_boxes, deled_scores
        

    def IoUs(self, coord1, coords):
        """[coord1とcoordsの各IoUを取得する処理を行う]

        Args:
            coord1 ([np.ndarray]): [矩形座標]
            coords ([np.ndarray]): [矩形座標の配列]

        Returns:
            [type]: [description]
        """        
        coords = to_array(coords).reshape(-1, 4)
        eps=1e-5
        coords_cpy = np.copy(coords)
        coord1_area = (coord1[2] - coord1[0]) * (coord1[3] - coord1[1])
        coords_areas = (coords[:, 2] - coords[:, 0]) * (coords[:, 3] - coords[:, 1])
        np.clip(coords_cpy[:, 0::2], coord1[0], coord1[2], out=coords_cpy[:, 0::2])
        np.clip(coords_cpy[:, 1::2], coord1[1], coord1[3], out=coords_cpy[:, 1::2])
        inters = (coords_cpy[:, 2] - coords_cpy[:, 0]) * (coords_cpy[:, 3] - coords_cpy[:, 1])
        unions = coords_areas + coord1_area - inters
        IoUs = inters / (unions + eps)
        return IoUs

    def concat_boxes(self, boxes):
        """[矩形座標の結合処理を行う]

        Args:
            boxes ([np.ndarray): [矩形座標の配列]

        Returns:
            [list]: [結合後の矩形座標]
        """        
        boxes = to_array(boxes).reshape(-1, 4)
        x_min, y_min = np.min(boxes[:, 0]), np.min(boxes[:, 1])
        x_max, y_max = np.max(boxes[:, 2]), np.max(boxes[:, 3])
        concated_box = [x_min, y_min, x_max, y_max]
        return concated_box    
    
    
    
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