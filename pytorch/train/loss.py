import math
import torch
import torch.nn as nn

class CIoULoss(nn.Module):
    def __init__(self):
        super(CIoULoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        targets = torch.sigmoid(targets)
        input_width = torch.exp(inputs[:, 2])
        input_height = torch.exp(inputs[:, 3])
        target_width = torch.exp(targets[:, 2])
        target_height = torch.exp(targets[:, 3])

        input_area = input_width * input_height
        target_area = target_width * target_height

        input_center_x = inputs[:, 0]
        input_center_y = inputs[:, 1]
        target_center_x = targets[:, 0]
        target_center_y = targets[:, 1]

        left_wall = torch.max(input_center_x - input_width / 2, target_center_x
                - target_width / 2)
        right_wall = torch.min(input_center_x + input_width / 2, target_center_x
                + target_width / 2)
        upper_wall = torch.max(input_center_y - input_height / 2,
                target_center_y - target_height / 2)
        lower_wall = torch.min(input_center_y + input_height / 2,
                target_center_y + target_height / 2)

        intersection_area = (torch.clamp((right_wall - left_wall),min=0) *
                torch.clamp((lower_wall - upper_wall), min=0))

        union = input_area + target_area - intersection_area
        iou = intersection_area / union

        x_left = torch.min(input_center_x - input_width / 2, target_center_x -
                target_width / 2)
        x_right = torch.max(input_center_x + input_width / 2, target_center_x +
                target_width / 2)
        y_top = torch.min(input_center_y - input_height / 2, target_center_y -
                target_height / 2)
        y_lower = torch.max(input_center_y + input_height / 2, target_center_y +
                target_height / 2)

        c = torch.clamp((x_right - x_left), min=0)**2 + torch.clamp((y_lower -
            y_top), min=0)**2
        d = (input_center_x - target_center_x)**2 + (input_center_y -
                target_center_y)**2
        D = d / c

        v = (4 / (math.pi**2)) * torch.pow((torch.atan(target_width /
            target_height) - torch.atan(input_width / input_height)), 2)

        with torch.no_grad():
            r = (iou>0.5).float()
            alpha = r*v/(1-iou+v)
            self.avg_iou = torch.mean(iou)

        CIoU = iou - D - alpha * v
        CIoU = torch.clamp(CIoU,min=-1.0,max = 1.0)
        return torch.mean(1-CIoU)
