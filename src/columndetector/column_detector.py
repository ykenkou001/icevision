import numpy as np

from columndetector.model import Model
from columndetector.post_process import PostProcess
from columndetector.utils import to_lists


'''
柱を物体検出モデルを使用して取得するクラス
'''
class ColumnDetector:
    def __init__(self, model_path):
        
        self.size = 256
        self.stride_rate = 0.80
        self.shift_rate = 0
        
        num_classes = 2
        self.model = Model(model_path, num_classes)
        
        nms_iou_threshold=0.20
        score_threshold=0.70
        self.post_process = PostProcess(nms_iou_threshold, score_threshold)
        
    def main(self, img):
        """[柱の矩形座標を物体検出モデルを使用して取得する処理を行う]

        Args:
            img ([np.ndarray]): [図面の画像]

        Returns:
            [list]: [boxes: 取得した柱の矩形座標]
        """        
        try:
            windows = self.get_windows(img.shape, self.size, self.stride_rate, self.shift_rate)
            patches = self.predict(img, windows)
            prediction = self.post_process.main(patches)
            boxes = prediction['boxes']
            boxes = to_lists(boxes)
        except:
            boxes = []
            
        return boxes    
        
    def predict(self, img, windows):
        """[図面全体のモデルによる予測結果を取得する]

        Args:
            img ([np.ndarray]): [図面の画像]
            windows ([list]): [モデルによる予測を適用する範囲]

        Returns:
            [list]: [各パッチに対するモデルの予測結果]
        """        
        #全てのパッチに対する予測をentire_predictionsとして取得する
        patches = []
        for window in windows:
            try:
                patch = self.model.predict_by_patch(img, window)
                patches.append(patch)
            except:
                pass
            
        return patches
        
        
        
        
    def get_windows(self, img_shape, size, stride_rate, shift_rate=0):
        """[画像をタイリングした際の各窓の座標を取得する]

        Args:
            img_shape ([tuple]): [画像サイズ]
            size ([int]): [ウィンドウの幅]
            stride_rate ([float]): [ストライドのsizeを基準とした比率]

        Returns:
            [list]: [各窓の座標のリスト]
        """    
    
        if shift_rate == 0 or shift_rate is None:
            shift_start_pt = 0
        else:
            shift_start_pt = int(size * shift_rate * stride_rate)
        
        if len(img_shape) == 3:
            H, W, _ = img_shape
        else:
            H, W = img_shape
        
        windows = []
            
        for j in np.arange(shift_start_pt, W, int(size * stride_rate)):
            w_start = j
            w_end = min(j + size, W)
            if w_end == W:
                w_start = w_start - (j+size - W)

            for i in np.arange(shift_start_pt, H, int(size * stride_rate)):
                h_start = i
                h_end = min(i + size, H)
                if h_end == H:
                    h_start = h_start - (i+size - H)
                window = [w_start, h_start, w_end, h_end]
                windows.append(window)
                if h_end == H:
                    break

            if w_end == W:
                    break
        return windows
        