import paddle
from paddle.io import DataLoader
import paddle.nn.functional as F
import paddle.optimizer as optim
import argparse

from models.loss import _bce_loss, _bce_loss_with_aux, \
    _bce_iou_loss, _bce_iou_loss_with_aux
from utils.dataset import SWINySEG
from utils.progressbar import ProgressBar
from utils.metric import get_cfm


def bce_loss(pred, target):
    return _bce_loss(pred, paddle.unsqueeze(target, 1))

def bce_iou_loss(pred, target):
    return _bce_iou_loss(pred, paddle.unsqueeze(target, 1))

def bce_loss_with_aux(pred, target, weight=[1, 0.4, 0.2]):
    return _bce_loss_with_aux(pred, paddle.unsqueeze(target, 1), weight)

def bce_iou_loss_with_aux(pred, target, weight=[1, 0.4, 0.2], weight2=1):
    return _bce_iou_loss_with_aux(pred, paddle.unsqueeze(target, 1), weight, weight2)

def get_accuracy(pred, label):
    pred_t = F.sigmoid(paddle.squeeze(pred[0] if isinstance(pred, tuple) else pred, 1))
    return float(
        paddle.mean(((pred_t>0.5).astype('int64')==label).astype('float32')))


def train(args):
    print("# =============== Training Configuration =============== #")
    print("# config: ddunet_"+args.config)
    print("# base_channels: "+str(args.base_channels))
    print("# iou loss: "+str(args.iou))
    print("# learning rate: "+str(args.lr))
    print("# epochs: "+str(args.epochs))
    print("# dataset: SWINySEG ("+args.dataset_split+")")
    print("# evaluation interval: "+str(args.eval_interval))
    print("# ====================================================== #")

    paddle.seed(999)

    model_name = "ddunet_c{}_{}_{}".format(
        args.base_channels, 
        args.config,
        "iou" if args.iou else "")

    train_set = SWINySEG(args.dataset_path, args.dataset_split, 'train', aug=False)
    test_set = SWINySEG(args.dataset_path, args.dataset_split, 'val', aug=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    train_record_file = open('./results/{}_train_log.csv'.format(model_name), 'w')
    eval_record_file = open('./results/{}_eval_log.csv'.format(model_name), 'w')
    eval_record_file.write("epoch,iteration,precision,recall,f-measure,error-rate,miou\n")
    train_record_file.write("epoch,iteration,loss,acc\n")

    if args.config == 'baseline':
        from models.ddunet_baseline import DDUNet
    elif args.config == 'dmsc':
        from models.ddunet_dmsc import DDUNet
    elif args.config == 'full':
        from models.ddunet import DDUNet
    else:
        raise Exception("Config name not found!!")

    model = DDUNet(3, args.base_channels, 2)

    scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=args.lr, gamma=0.95, verbose=True)
    optimizer = optim.Adam(scheduler, parameters=model.parameters(), weight_decay=5e-4)

    train_steps = len(train_loader)
    test_steps = len(test_loader)

    for e in range(args.epochs):
        train_loss = 0
        train_acc = 0

        bar = ProgressBar(maxStep=train_steps)
        model.train()

        for i, (image, label) in enumerate(train_loader()):
            optimizer.clear_grad()
            pred = model(image)

            if not isinstance(pred, tuple):
                if args.iou:
                    loss = bce_iou_loss(pred, label)
                else:
                    loss = bce_loss(pred, label)
            else:
                if args.iou:
                    loss = bce_iou_loss_with_aux(pred, label)
                else:
                    loss = bce_loss_with_aux(pred, label)

            loss.backward()
            optimizer.step()

            batch_loss = float(loss)
            batch_acc = get_accuracy(pred, label)
            train_loss += batch_loss
            train_acc += batch_acc

            if i != train_steps-1:
                bar.updateBar(
                        i+1, headData={'Epoch':e+1, 'Status':'training'}, 
                        endData={
                            'Train loss': "{:.5f}".format(train_loss/(i+1)),
                            'Train Acc': "{:.5f}".format(train_acc/(i+1))})
            else:
                bar.updateBar(
                        i+1, headData={'Epoch':e+1, 'Status':'finished'}, 
                        endData={
                            'Train loss': "{:.5f}".format(train_loss/(i+1)),
                            'Train Acc': "{:.5f}".format(train_acc/(i+1))})
            
            train_record_file.write(
                "{},{},{},{}\n".format(e+1, e*train_steps+i+1, batch_loss, batch_acc))

        if (e+1) % args.eval_interval == 0:
            bar = ProgressBar(maxStep=test_steps)
            model.eval()

            accuracy, precision, recall, f_measure, error_rate, miou = 0, 0, 0, 0, 0, 0

            for i, (image, label) in enumerate(test_loader()):
                pred = model(image)

                pred_t = F.interpolate(
                    F.sigmoid(pred[0] if isinstance(pred, tuple) else pred), 
                    (300, 300), mode='bilinear', align_corners=True)
                pred_t = (pred_t[0][0] > 0.5).astype('int32')
                label_t = label[0]

                tp, fn, fp, tn = get_cfm(pred_t, label_t)

                accuracy += float(paddle.mean((pred_t==label_t).astype('float32')))
                precision += int(tp)/(int(tp)+int(fp)) if int(tp)+int(fp) !=0 else 0
                recall += int(tp)/(int(tp)+int(fn))
                miou += ((int(tp)/(int(tp)+int(fp)+int(fn)))+(int(tn)/(int(tn)+int(fp)+int(fn))))/2

                bar.updateBar(i+1, headData={}, endData={})

            accuracy /= test_steps
            precision /= test_steps
            recall /= test_steps
            f_measure = (2 * precision * recall) / (precision + recall)
            error_rate = 1 - accuracy
            miou /= test_steps

            if i != test_steps-1:
                bar.updateBar(i+1, headData={'Epoch (Test)':e+1, 'Status':'testing'}, endData={})
            else:
                bar.updateBar(i+1, headData={'Epoch (Test)':e+1, 'Status':'finished'}, endData={})

            for key, val in zip(
                    ['precision', 'recall', 'f-measure', 'error-rate', 'miou'], 
                    [precision, recall, f_measure, error_rate, miou]):
                print("{}: {:.5f}".format(key, val))

            eval_record_file.write(
                "{},{},{},{},{},{},{}\n".format(
                e+1, (e+1)*train_steps, 
                precision, recall, f_measure, error_rate, miou))
        
        scheduler.step()

    paddle.save(
        model.state_dict(), 
        "./weights/{}_e{}.pdparam".format(model_name, args.epochs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='full', 
        help="the config of model (default: full)")
    parser.add_argument(
        '--base_channels', type=int, default=8, 
        help="the base_channels value of model (default: 8)")
    parser.add_argument(
        '--batch_size', type=int, default=16, 
        help="batchsize for model training (default: 16)")
    parser.add_argument(
        '--lr', type=float, default=5e-4, 
        help="the learning rate for training (default: 5e-4)")
    parser.add_argument(
        '--epochs', type=int, default=100, 
        help="number of training epochs (default: 100)")
    parser.add_argument(
        '-iou', action="store_true", default=False, 
        help="use iou loss (default: False)")
    parser.add_argument(
        '--dataset_split', type=str, default='all',
        help="split of SWINySEG dataset, ['all', 'd', 'n'] (default: all)")
    parser.add_argument(
        '--dataset_path', type=str, default='./dataset/SWINySEG', 
        help="path of training dataset (default: ./dataset/SWINySEG)")
    parser.add_argument(
        '--eval_interval', type=int, default=5, 
        help="interval of model evaluation during training (default: 5)"
    )
    
    args = parser.parse_args()
    
    train(args)