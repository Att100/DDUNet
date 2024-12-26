import paddle
import paddle.nn.functional as F
from PIL import Image
import numpy as np
from paddle.io import DataLoader

from utils.dataset import SWINySEG
from utils.metric import get_metrics


def run_all_experiments():
    # quantitative
    from models.ddunet import DDUNet
    model = DDUNet(3, 8, 2)
    model.set_state_dict(paddle.load("./weights/ddunet_c8_full_e100.pdparam"))
    model.eval()

    for data_split in ['all', 'd', 'n']:
        testset = SWINySEG("./dataset/SWINySEG", data_split, 'val')
        loader = DataLoader(testset, batch_size=1, shuffle=False)
        metrics = get_metrics(model, loader)
        print("Dataset: SWINySEG-{}".format(data_split))
        for key, val in metrics.items():
            print("{}: {:.5f}".format(key, val))
                
    # ablation study
    testset = SWINySEG("./dataset/SWINySEG", 'all', 'val')
    loader = DataLoader(testset, batch_size=1, shuffle=False)

    from models.ddunet_baseline import DDUNet
    model = DDUNet(3, 8, 2)
    model.set_state_dict(paddle.load("./weights/ddunet_c8_baseline_e100.pdparam"))
    model.eval()

    metrics = get_metrics(model, loader)
    print("Model: baseline")
    for key, val in metrics.items():
        print("{}: {:.5f}".format(key, val))


    testset = SWINySEG("./dataset/SWINySEG", 'all', 'val')
    loader = DataLoader(testset, batch_size=1, shuffle=False)

    from models.ddunet_dmsc import DDUNet
    model = DDUNet(3, 8, 2)
    model.set_state_dict(paddle.load("./weights/ddunet_c8_dmsc_e100.pdparam"))
    model.eval()

    metrics = get_metrics(model, loader)
    print("Model: baseline+dmsc")
    for key, val in metrics.items():
        print("{}: {:.5f}".format(key, val))


if __name__ == "__main__":
    run_all_experiments()

    
