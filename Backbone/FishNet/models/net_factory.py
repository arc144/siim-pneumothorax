from .fishnet import fish
import torch


def fishnet99(pretrained=True, **kwargs):
    """

    :return:
    """
    net_cfg = {
        #  input size:   [224, 56, 28,  14  |  7,   7,  14,  28 | 56,   28,  14]
        # output size:   [56,  28, 14,   7  |  7,  14,  28,  56 | 28,   14,   7]
        #                  |    |    |   |     |    |    |    |    |     |    |
        'network_planes': [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600],
        'num_res_blks': [2, 2, 6, 2, 1, 1, 1, 1, 2, 2],
        'num_trans_blks': [1, 1, 1, 1, 1, 4],
        'num_cls': 1000,
        'num_down_sample': 3,
        'num_up_sample': 3,
    }
    cfg = {**net_cfg, **kwargs}
    net = fish(**cfg)
    if pretrained:
        dic = torch.load('/media/hdd/Kaggle/Pneumothorax/Backbone/FishNet/checkpoints/fishnet99_ckpt.tar')['state_dict']
        new_dic = {}
        for k, v in dic.items():
            new_dic[k.lstrip('module.')] = v
        net.load_state_dict(new_dic)
        print('Pretrained weights loaded!')
    return net


def fishnet150(pretrained=True, **kwargs):
    """

    :return:
    """
    net_cfg = {
        #  input size:   [224, 56, 28,  14  |  7,   7,  14,  28 | 56,   28,  14]
        # output size:   [56,  28, 14,   7  |  7,  14,  28,  56 | 28,   14,   7]
        #                  |    |    |   |     |    |    |    |    |     |    |
        'network_planes': [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600],
        'num_res_blks': [2, 4, 8, 4, 2, 2, 2, 2, 2, 4],
        'num_trans_blks': [2, 2, 2, 2, 2, 4],
        'num_cls': 1000,
        'num_down_sample': 3,
        'num_up_sample': 3,
    }
    cfg = {**net_cfg, **kwargs}
    net = fish(**cfg)
    if pretrained:
        dic = torch.load('/media/hdd/Kaggle/Pneumothorax/Backbone/FishNet/checkpoints/fishnet150_ckpt.tar')['state_dict']
        new_dic = {}
        for k, v in dic.items():
            new_dic[k.lstrip('module.')] = v
        net.load_state_dict(new_dic)
        print('Pretrained weights loaded!')
    return net


def fishnet201(pretrained=True, **kwargs):
    """

    :return:
    """
    net_cfg = {
        #  input size:   [224, 56, 28,  14  |  7,   7,  14,  28 | 56,   28,  14]
        # output size:   [56,  28, 14,   7  |  7,  14,  28,  56 | 28,   14,   7]
        #                  |    |    |   |     |    |    |    |    |     |    |
        'network_planes': [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600],
        'num_res_blks': [3, 4, 12, 4, 2, 2, 2, 2, 3, 10],
        'num_trans_blks': [2, 2, 2, 2, 2, 9],
        'num_cls': 1000,
        'num_down_sample': 3,
        'num_up_sample': 3,
    }
    cfg = {**net_cfg, **kwargs}
    net = fish(**cfg)
    if pretrained:
        net.load_state_dict(
            torch.load(
                '/media/hdd/Kaggle/Pneumothorax/Backbone/FishNet/checkpoints/fishnet201_ckpt_welltrain.tar')['state_dict'])
        print('Pretrained weights loaded!')
    return net
