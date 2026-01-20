import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_BCE_and_MSE_loss(nn.Module):
    """
    Combined loss for vector field segmentation:
    - Dice + BCE loss for binary mask (channel 0 of prediction and target)
    - MSE loss for vector components (channels 1-3 of prediction and target)

    Expected input shapes:
        net_output: (B, 4, D, H, W) - channel 0: mask logits, channels 1-3: vector predictions
        target: (B, 4, D, H, W) - channel 0: mask (0 or 1), channels 1-3: vector ground truth
    """

    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_dice=1, weight_bce=1, weight_mse=1,
                 apply_mask_to_vectors=True, dice_class=MemoryEfficientSoftDiceLoss):
        """
        Args:
            bce_kwargs: kwargs for BCEWithLogitsLoss
            soft_dice_kwargs: kwargs for Dice loss
            weight_dice: weight for Dice loss on mask
            weight_bce: weight for BCE loss on mask
            weight_mse: weight for MSE loss on vectors
            apply_mask_to_vectors: if True, only compute vector MSE where target mask > 0.5
            dice_class: Dice loss class to use
        """
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.weight_mse = weight_mse
        self.apply_mask_to_vectors = apply_mask_to_vectors

        self.bce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        Args:
            net_output: (B, 4, D, H, W) - mask logits + vector predictions
            target: (B, 4, D, H, W) - mask + vector ground truth
        Returns:
            Combined loss value
        """
        # Split predictions and targets
        pred_mask = net_output[:, 0:1]  # (B, 1, D, H, W)
        pred_vectors = net_output[:, 1:4]  # (B, 3, D, H, W)

        target_mask = target[:, 0:1]  # (B, 1, D, H, W)
        target_vectors = target[:, 1:4]  # (B, 3, D, H, W)

        # Dice loss on mask
        dc_loss = self.dc(pred_mask, target_mask) if self.weight_dice != 0 else 0

        # BCE loss on mask
        bce_loss = self.bce(pred_mask, target_mask.float()) if self.weight_bce != 0 else 0

        # MSE loss on vectors
        if self.weight_mse != 0:
            if self.apply_mask_to_vectors:
                # Only compute vector loss where ground truth mask > 0.5 (foreground)
                fg_mask = (target_mask > 0.5).float()
                vector_mse = self.mse(pred_vectors, target_vectors)
                # Apply mask: average over foreground voxels only
                masked_mse = vector_mse * fg_mask
                num_fg_voxels = fg_mask.sum()
                # Average over 3 vector channels and foreground voxels
                mse_loss = masked_mse.sum() / torch.clip(num_fg_voxels * 3, min=1e-8)
            else:
                mse_loss = self.mse(pred_vectors, target_vectors).mean()
        else:
            mse_loss = 0

        result = self.weight_dice * dc_loss + self.weight_bce * bce_loss + self.weight_mse * mse_loss
        return result
