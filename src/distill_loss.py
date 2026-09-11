"""Multi-label knowledge distillation losses for Charades VideoMAE.

Charades is multi-label, so we use BCE (not softmax CE/KL from the DeiT PDF).
Soft targets = sigmoid(teacher_logits / T); student is trained with
BCEWithLogits on those soft labels, scaled by T^2 (Hinton temperature trick,
Hinton et al. 2015, "Distilling the Knowledge in a Neural Network").

Feature terms:
  - alpha_feat: point-wise (FitNet-style) feature matching. Requires a
    `projector` to bridge the student/teacher embedding-dim mismatch.
  - alpha_rkd_d / alpha_rkd_a: relational KD (Park et al. 2019, CVPR),
    matching pairwise-distance and triplet-angle structure between the
    teacher's and student's own embedding spaces instead of raw feature
    values. No projector needed: relations are dimensionless, computed
    separately in each model's native space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise_dist(e, eps=1e-12):
    """[N, D] -> [N, N] Euclidean distance matrix (zero diagonal)."""
    sq = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    d = (sq.unsqueeze(1) + sq.unsqueeze(0) - 2 * prod).clamp(min=eps).sqrt()
    return d.clone().fill_diagonal_(0)


def rkd_distance_loss(teacher_feats, student_feats):
    """Distance-wise RKD loss (Park et al. 2019, Eq. 5/7)."""
    with torch.no_grad():
        t_d = _pairwise_dist(teacher_feats)
        t_d = t_d / t_d[t_d > 0].mean()

    s_d = _pairwise_dist(student_feats)
    s_d = s_d / s_d[s_d > 0].mean()
    return F.smooth_l1_loss(s_d, t_d)


def rkd_angle_loss(teacher_feats, student_feats):
    """Angle-wise RKD loss (Park et al. 2019, Eq. 9/10)."""
    with torch.no_grad():
        td = teacher_feats.unsqueeze(0) - teacher_feats.unsqueeze(1)
        t_angle = torch.bmm(F.normalize(td, dim=2), F.normalize(td, dim=2).transpose(1, 2))

    sd = student_feats.unsqueeze(0) - student_feats.unsqueeze(1)
    s_angle = torch.bmm(F.normalize(sd, dim=2), F.normalize(sd, dim=2).transpose(1, 2))
    return F.smooth_l1_loss(s_angle, t_angle)


class MultiLabelDistillLoss(nn.Module):
    def __init__(self, alpha_kd=0.5, alpha_feat=0.0, alpha_rkd_d=0.0,
                 alpha_rkd_a=0.0, temperature=3.0, pos_weight=None):
        super().__init__()
        self.alpha_kd = float(alpha_kd)
        self.alpha_feat = float(alpha_feat)
        self.alpha_rkd_d = float(alpha_rkd_d)
        self.alpha_rkd_a = float(alpha_rkd_a)
        self.T = float(temperature)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, student_logits, teacher_logits, labels,
                student_feats=None, teacher_feats=None, projector=None):
        loss_hard = self.bce(student_logits, labels)
        loss_kd = student_logits.new_zeros(())
        loss_feat = student_logits.new_zeros(())
        loss_rkd_d = student_logits.new_zeros(())
        loss_rkd_a = student_logits.new_zeros(())

        if self.alpha_kd > 0:
            if teacher_logits is None:
                raise ValueError("teacher_logits required when alpha_kd > 0")
            T = self.T
            soft = torch.sigmoid(teacher_logits.detach() / T)
            loss_kd = F.binary_cross_entropy_with_logits(
                student_logits / T, soft, reduction="mean",
            ) * (T * T)

        loss = (1.0 - self.alpha_kd) * loss_hard + self.alpha_kd * loss_kd

        have_feats = student_feats is not None and teacher_feats is not None
        if self.alpha_feat > 0 and have_feats:
            s = student_feats
            if projector is not None:
                s = projector(s)
            s = F.normalize(s, dim=-1)
            t = F.normalize(teacher_feats.detach(), dim=-1)
            loss_feat = F.mse_loss(s, t)
            loss = loss + self.alpha_feat * loss_feat

        if self.alpha_rkd_d > 0 and have_feats:
            loss_rkd_d = rkd_distance_loss(teacher_feats.detach(), student_feats)
            loss = loss + self.alpha_rkd_d * loss_rkd_d

        if self.alpha_rkd_a > 0 and have_feats:
            loss_rkd_a = rkd_angle_loss(teacher_feats.detach(), student_feats)
            loss = loss + self.alpha_rkd_a * loss_rkd_a

        return loss, {
            "hard": float(loss_hard.detach()),
            "kd": float(loss_kd.detach()),
            "feat": float(loss_feat.detach()),
            "rkd_d": float(loss_rkd_d.detach()),
            "rkd_a": float(loss_rkd_a.detach()),
        }
