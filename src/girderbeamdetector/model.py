import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead

from girderbeamdetector.utils import to_list

'''
保存したモデルを読み込んで使用するラッパークラス。
'''
class Model:
    
    def __init__(self, model_path, num_classes, anchor_sizes=None, anchor_aspect_ratios=None):
        self.model = self._load_net(model_path, num_classes, anchor_sizes, anchor_aspect_ratios)
        
        
    def _get_model_instance_customized_anchor(self, num_classes, anchor_sizes, anchor_aspect_ratios):
        """[アンカーをカスタマイズしたモデルのインスタンスを取得する]

        Args:
            num_classes ([int]): [予測するクラスの数]
            anchor_sizes ([tuple]): [アンカーサイズを格納したタプル]
            anchor_aspect_ratios ([tuple]): [アンカーの縦横比率を格納したタプル]

        Returns:
            [Faster R-CNNクラス]: [model 取得したモデルのインスタンス]
        """        
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # create an anchor_generator for the FPN
        # which by default has 5 outputs
        anchor_generator = AnchorGenerator(
            sizes=tuple([anchor_sizes for _ in range(5)]),
            aspect_ratios=tuple([anchor_aspect_ratios for _ in range(5)]))
        model.rpn.anchor_generator = anchor_generator

        # 256 because that's the number of features that FPN returns
        model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
        
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    
    def _get_model_instance(self, num_classes):
        """[モデルのインスタンスを取得する]

        Args:
            num_classes ([int]): [予測するクラスの数]

        Returns:
            [Faster R-CNNクラス]: [model 取得したモデルのインスタンス]
        """        
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model


    def _load_net(self, model_path, num_classes, anchor_sizes=None, anchor_aspect_ratios=None):
        """[学習済みモデルの読み込みを行う]

        Args:
            model_path ([str]): [モデルを読み込むファイルのパス]
            num_classes ([int]): [予測するクラスの数]
            anchor_sizes ([tuple]): [アンカーサイズを格納したタプル]. Defaults to None.
            anchor_aspect_ratios ([tuple]): [アンカーの縦横比率を格納したタプル]. Defaults to None.

        Returns:
            [Faster R-CNNクラス]: [model: 学習済みモデル]
        """        
        # train on the GPU or on the CPU, if a GPU is not available
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # get the model
        if anchor_sizes is not None:
            model = self._get_model_instance_customized_anchor(num_classes, anchor_sizes, anchor_aspect_ratios)
        else:
            model = self._get_model_instance(num_classes)
        # move model to the right device
        model.to(device)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        model.to(device)
        model.eval()
        return model
    
    
    '''
    複数枚同時に処理しない場合は、一枚のパッチに対する予測をpredictionsとして返す

    '''
    def predict_by_patch(self, image, window):
        """[パッチごとにモデルによる予測を行う]

        Args:
            image ([np.ndarray]): [図面の画像]
            window ([list]): [パッチの範囲となる矩形座標]

        Returns:
            [dict]: [prediction: 予測結果]
        """        
        patch = image[window[1] : window[3], window[0] : window[2]]
        sample = {'image': patch.astype(np.uint8)}
        transform = self._get_transforms()
        sample = transform(**sample)
        if torch.cuda.is_available():
            patch = sample['image'].unsqueeze(0).cuda().float()
        else:
            patch = sample['image'].unsqueeze(0).float()

        with torch.no_grad():
            pred = self.model(patch)[0]

        boxes = to_list(pred['boxes'])
        scores = to_list(pred['scores'])
        labels = to_list(pred['labels'])
        prediction = {
            'window' : window,
            'boxes': boxes,
            'scores': scores,
            'labels' : labels
        }
        return prediction
    
    
    
    def _get_transforms(self):
        """[モデルにデータをインプットする前の前処理を行うクラスのインスタンスを生成する]

        Returns:
            [A.Composeクラス]: [モデルにデータをインプットする前の前処理を行うクラスのインスタンス]
        """        
        return A.Compose(
            [
                ToTensor()
            ], 
            p=1.0, 
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0, 
                min_visibility=0
            )
        )