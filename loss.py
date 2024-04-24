import torch
import torch.nn as nn
from torch.nn import functional as F
from pprint import pprint
import sys


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.5):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_feature = features
        anchor_feature = contrast_feature
        anchor_count = 2  # two views if is_waug is True, one view if is_waug is False

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        # it produces 1 for the non-matching places and 0 for matching places i.e its opposite of mask
        mask = mask * logits_mask
        # compute log_prob with logsumexp
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        # log_prob = x - max(x1,..,xn) - logsumexp(x1,..,xn) the equation
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum.detach()
        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.mean()

        return loss


class LCL(nn.Module):
    def __init__(self, temperature=0.07):
        super(LCL, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, weights=None, mask=None):
        """
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        batch_size = features.shape[0]
        weights = F.softmax(weights, dim=1)
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_feature = features
        anchor_feature = contrast_feature
        anchor_count = 2

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)

        ## it produces 0 for the non-matching places and 1 for matching places and neg mask does the opposite
        mask = mask * logits_mask
        weighted_mask = torch.zeros_like(logits_mask).float().to(device)
        for i,val in enumerate(labels):
            for j,jval in enumerate(labels):
                weighted_mask[i,j] = weights[i,jval]
        weighted_mask = weighted_mask * logits_mask
        pos_weighted_mask = weighted_mask * mask
        # compute log_prob with logsumexp
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

        logits = anchor_dot_contrast - logits_max.detach()
        # print(logits)

        exp_logits = torch.exp(logits) * weighted_mask

        ## log_prob = x - max(x1,..,xn) - logsumexp(x1,..,xn) the equation
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum.detach()

        # loss
        loss = -1 * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        loss = loss.mean()
        return loss


class RDrop(nn.Module):
    """
    R-Drop for classification tasks.
    Example:
        criterion = RDrop()
        logits1 = model(input)  # model: a classification model instance. input: the input data
        logits2 = model(input)
        loss = criterion(logits1, logits2, target)     # target: the target labels. len(loss_) == batch size
    Notes: The model must contains `dropout`. The model predicts twice with the same input, and outputs logits1 and logits2.
    """

    def __init__(self, kl_weight=5.0):
        super(RDrop, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.kld = nn.KLDivLoss(reduction='none')
        self.kl_weight = kl_weight

    def forward(self, logits1, logits2, target):
        """
        Args:
            logits1: One output of the classification model.
            logits2: Another output of the classification model.
            target: The target labels.
            kl_weight: The weight for `kl_loss`.

        Returns:
            loss: Losses with the size of the batch size.
        """
        ce_loss = (self.ce(logits1, target) + self.ce(logits2, target)) / 2
        kl_loss1 = self.kld(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1)).sum(-1)
        kl_loss2 = self.kld(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1)).sum(-1)
        kl_loss = (kl_loss1 + kl_loss2) / 2
        loss = ce_loss + self.kl_weight * kl_loss
        return loss.mean()

    # def simcse_loss(y_true, y_pred):
    #     """用于SimCSE训练的loss
    #     """
    #     # 构造标签
    #     idxs = K.arange(0, K.shape(y_pred)[0])
    #     idxs_1 = idxs[None, :]
    #     idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    #     y_true = K.equal(idxs_1, idxs_2)
    #     y_true = K.cast(y_true, K.floatx())
    #     # 计算相似度
    #     y_pred = K.l2_normalize(y_pred, axis=1)
    #     similarities = K.dot(y_pred, K.transpose(y_pred))
    #     similarities = similarities - torch.eye(K.shape(y_pred)[0]) * 1e12
    #     similarities = similarities * 20
    #     loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    #     return K.mean(loss)
