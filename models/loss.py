import paddle
import paddle.nn.functional as F


def _bce_loss(pred, target):
    return F.binary_cross_entropy(F.sigmoid(pred), target)

def _iou_loss(pred, target, smooth=1):
    intersection = paddle.sum(target * pred, axis=[1,2,3])
    union = paddle.sum(target, axis=[1,2,3]) + paddle.sum(pred, axis=[1,2,3]) - intersection
    iou = paddle.mean((intersection + smooth) / (union + smooth), axis=0)
    return 1 - iou

def _bce_loss_with_aux(pred, target, weight=[1, 0.4, 0.2]):
    # pred = (x1, x2, x4)
    pred, pred_sub2, pred_sub4 = tuple(pred)

    target_2x = F.interpolate(
        target, pred_sub2.shape[2:], mode='bilinear', align_corners=True)
    target_4x = F.interpolate(
        target, pred_sub4.shape[2:], mode='bilinear', align_corners=True)
    
    _1x_loss = F.binary_cross_entropy(F.sigmoid(pred), target)
    _2x_loss = F.binary_cross_entropy(F.sigmoid(pred_sub2), target_2x)
    _4x_loss = F.binary_cross_entropy(F.sigmoid(pred_sub4), target_4x)

    loss = weight[0] * _1x_loss + weight[1] * _2x_loss + weight[2] * _4x_loss
    return loss

def _bce_iou_loss(pred, target, weight=1):
    pred = F.sigmoid(pred)
    return F.binary_cross_entropy(pred, target) + weight * _iou_loss(pred, target)

def _bce_iou_loss_with_aux(pred, target, weight=[1, 0.4, 0.2], weight2=1):
    # pred = (x1, x2, x4)
    pred, pred_sub2, pred_sub4 = tuple(pred)

    target_2x = F.interpolate(
        target, pred_sub2.shape[2:], mode='bilinear', align_corners=True)
    target_4x = F.interpolate(
        target, pred_sub4.shape[2:], mode='bilinear', align_corners=True)
    
    pred, pred_sub2, pred_sub4 = F.sigmoid(pred), F.sigmoid(pred_sub2), F.sigmoid(pred_sub4)
    
    _1x_loss = F.binary_cross_entropy(pred, target) + weight2 * _iou_loss(pred, target)
    _2x_loss = F.binary_cross_entropy(pred_sub2, target_2x) + weight2 * _iou_loss(pred_sub2, target_2x)
    _4x_loss = F.binary_cross_entropy(pred_sub4, target_4x) + weight2 * _iou_loss(pred_sub4, target_4x)

    loss = weight[0] * _1x_loss + weight[1] * _2x_loss + weight[2] * _4x_loss
    return loss


