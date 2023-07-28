import torch
from thop import profile

from train_coco_dataset.backbone import LastLevelMaxPool, BackboneWithFPN
from train_coco_dataset.backbone.GhostNet import ghostnet
from train_coco_dataset.network_files import FasterRCNN, AnchorsGenerator


def create_model(num_classes):
    import torchvision
    from torchvision.models.feature_extraction import create_feature_extractor

    # GhostNet
    # backbone = ghostnet()
    # backbone.load_state_dict(torch.load("GhostNet.pth", map_location="cpu"))
    # return_layers = {"blocks.4": "0",   # stride 8
    #                  "blocks.6": "1",  # stride 16
    #                  "blocks.9": "2"}  # stride 32
    # in_channels_list = [40, 112, 960]
    # new_backbone = create_feature_extractor(backbone, return_layers)
    # print(backbone)
    # exit()
    # --- mobilenet_v3_large fpn backbone --- #
    backbone = torchvision.models.mobilenet_v2(pretrained=True)
    print(backbone)
    return_layers = {"features.6": "0",   # stride 8
                     "features.12": "1",  # stride 16
                     "features.16": "2"}  # stride 32
    # 提供给fpn的每个特征层channel
    in_channels_list = [32, 96, 160]
    new_backbone = create_feature_extractor(backbone, return_layers)
    img = torch.randn(1, 3, 224, 224)
    outputs = new_backbone(img)
    [print(f"{k} shape: {v.shape}") for k, v in outputs.items()]

    # --- efficientnet_b0 fpn backbone --- #
    # backbone = torchvision.models.efficientnet_b0(pretrained=True)
    # # print(backbone)
    # return_layers = {"features.3": "0",  # stride 8
    #                  "features.4": "1",  # stride 16
    #                  "features.8": "2"}  # stride 32
    # # 提供给fpn的每个特征层channel
    # in_channels_list = [40, 80, 1280]
    # new_backbone = create_feature_extractor(backbone, return_layers)
    # # img = torch.randn(1, 3, 224, 224)
    # # outputs = new_backbone(img)
    # # [print(f"{k} shape: {v.shape}") for k, v in outputs.items()]

    backbone_with_fpn = BackboneWithFPN(new_backbone,
                                        return_layers=return_layers,
                                        in_channels_list=in_channels_list,
                                        out_channels=256,
                                        extra_blocks=LastLevelMaxPool(),
                                        re_getter=False)

    # anchor_sizes = ((64,), (128,), (256,), (512,))
    # aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    # anchor_generator = AnchorsGenerator(sizes=anchor_sizes,
    #                                     aspect_ratios=aspect_ratios)
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2'],  # 在哪些特征层上进行RoIAlign pooling
    #                                                 output_size=[7, 7],  # RoIAlign pooling输出特征矩阵尺寸
    #                                                 sampling_ratio=2)  # 采样率
    #
    # model = FasterRCNN(backbone=backbone_with_fpn,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    return backbone_with_fpn


def modelInfo(model, verbose=False):
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients

    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    # 计算MACs或FLOPs时要手动修改profile种的inputs尺寸
    try:
        # FLOPS:注意全大写，是floating point operations per second的缩写，意指每秒浮点运算次数，理解为计算速度。
        # FLOPs:注意s小写，是floating point operations的缩写（s表复数），意指浮点运算数，理解为计算量。可以用来衡量算法/模型的复杂度。
        macs, _ = profile(model, inputs=(torch.zeros(1, 3, 224, 224),), verbose=False)
        ms = ', %f GMACs' % (macs / 1E9)
        fs = ', %f GFLOPs' % (macs / 1E9 * 2)
    except:
        ms = ''
        fs = ''

    print('Model Summary: %g layers, %g parameters, %g gradients%s%s' % (len(list(model.parameters())), n_p, n_g, ms, fs))

if __name__ == '__main__':
    # model = ghostnet()
    # modelInfo(model)

    model = create_model(90)
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))

