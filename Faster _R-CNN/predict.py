import os
import time
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from GhostNet.model.GhostNet import ghostnet
from network_files import FasterRCNN, AnchorsGenerator
from draw_box_utils import draw_objs
from train_coco_dataset.backbone import LastLevelMaxPool, BackboneWithFPN


def create_model(num_classes):
    import torchvision
    from torchvision.models.feature_extraction import create_feature_extractor
    # GhostNet
    backbone = ghostnet()
    backbone.load_state_dict(torch.load("GhostNet.pth", map_location="cpu"))
    return_layers = {"blocks.4": "0",   # stride 8
                     "blocks.6": "1",  # stride 16
                     "blocks.9": "2"}  # stride 32
    in_channels_list = [40, 112, 960]
    new_backbone = create_feature_extractor(backbone, return_layers)
    # print(backbone)
    # exit()
    # --- mobilenet_v3_large fpn backbone --- #
    # backbone = torchvision.models.mobilenet_v3_large(pretrained=True)
    # # print(backbone)
    # return_layers = {"features.6": "0",   # stride 8
    #                  "features.12": "1",  # stride 16
    #                  "features.16": "2"}  # stride 32
    # # 提供给fpn的每个特征层channel
    # in_channels_list = [40, 112, 960]
    # new_backbone = create_feature_extractor(backbone, return_layers)
    # img = torch.randn(1, 3, 224, 224)
    # outputs = new_backbone(img)
    # [print(f"{k} shape: {v.shape}") for k, v in outputs.items()]
    backbone_with_fpn = BackboneWithFPN(new_backbone,
                                        return_layers=return_layers,
                                        in_channels_list=in_channels_list,
                                        out_channels=256,
                                        extra_blocks=LastLevelMaxPool(),
                                        re_getter=False)

    anchor_sizes = ((64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorsGenerator(sizes=anchor_sizes,
                                        aspect_ratios=aspect_ratios)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2'],  # 在哪些特征层上进行RoIAlign pooling
                                                    output_size=[7, 7],  # RoIAlign pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone_with_fpn,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model



def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    num_classes = 90  # 不包含背景
    model = create_model(num_classes=num_classes + 1)

    # load train weights
    weights_path = "./save_weights/model_25.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    label_json_path = './coco91_indices.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        category_index = json.load(f)

    # load image
    original_img = Image.open("./test.jpg")

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        plot_img = draw_objs(original_img,
                             predict_boxes,
                             predict_classes,
                             predict_scores,
                             category_index=category_index,
                             box_thresh=0.5,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        plt.imshow(plot_img)
        plt.show()
        # 保存预测的图片结果
        plot_img.save("test_result.jpg")


if __name__ == '__main__':
    main()

