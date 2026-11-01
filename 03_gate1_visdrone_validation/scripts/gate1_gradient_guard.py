"""Two-halfspace Euclidean projection and bounded auxiliary gradient.

Only instantaneous auxiliary directions are constrained, not momentum, TP or AP.
"""
from __future__ import annotations
import torch
import gate1_native_target_correction as N

P = N.P
RHO = .25
DOUBLE_FEASIBILITY = 2e-10
FLOAT32_FEASIBILITY = 2e-6


def project_two_halfspaces(auxiliary, base, classification):
    """Project onto {u:b.u>=0,c.u>=0}; active-set enumeration in float64."""
    if not (auxiliary.shape == base.shape == classification.shape and auxiliary.ndim == 1):
        raise ValueError("Gradient shape mismatch")
    a, b, c = (x.detach().to(torch.float64) for x in (auxiliary, base, classification))
    if not all(bool(torch.isfinite(x).all()) for x in (a, b, c)):
        raise ValueError("Nonfinite gradients")
    normals = [x / torch.linalg.vector_norm(x) for x in (b, c) if float(torch.linalg.vector_norm(x)) > 0]
    scale = max(1., float(torch.linalg.vector_norm(a)))
    tolerance = DOUBLE_FEASIBILITY * scale
    feasible = lambda x: all(float(torch.dot(n, x)) >= -tolerance for n in normals)
    if not normals or feasible(a):
        return a.clone(), "already_feasible"
    candidates = []
    for index, n in enumerate(normals):
        coefficient = -float(torch.dot(n, a))
        if coefficient >= -tolerance:
            u = a + coefficient * n
            if feasible(u):
                candidates.append((float(torch.sum((u-a).square())), index, u, f"plane_{index}"))
    if len(normals) == 2:
        n0, n1 = normals
        correlation = torch.dot(n0, n1)
        residual = n1 - correlation * n0
        residual_norm = torch.linalg.vector_norm(residual)
        if float(residual_norm) > 1e-14:
            orthogonal = residual / residual_norm
            first = torch.dot(a, n0)
            second = torch.dot(a, orthogonal)
            multiplier1 = -second / residual_norm
            multiplier0 = -first - correlation * multiplier1
            if min(float(multiplier0), float(multiplier1)) >= -tolerance:
                u = a - first * n0 - second * orthogonal
                if feasible(u):
                    candidates.append((float(torch.sum((u-a).square())), 2, u, "both_planes"))
    if not candidates:
        raise ArithmeticError("No valid active-set solution; do not silently zero/retry")
    _, _, projected, active = min(candidates, key=lambda x: (x[0], x[1]))
    return projected, active


def guarded_gradient(base, classification, auxiliary):
    """Return a float32 final gradient plus independently checkable Gram data."""
    if any(x.dtype != torch.float32 for x in (base, classification, auxiliary)):
        raise ValueError("Accumulated training gradients must be FP32")
    b, c, a = (x.detach().to(torch.float64) for x in (base, classification, auxiliary))
    nb, nc, na = (float(torch.linalg.vector_norm(x)) for x in (b, c, a))
    if nb == 0 or na == 0:
        u = torch.zeros_like(a)
        active, factor = "zero_base_or_auxiliary", 0.
    else:
        u, active = project_two_halfspaces(a, b, c)
        nu = float(torch.linalg.vector_norm(u))
        factor = min(1., RHO * nb / nu) if nu > 0 else 0.
        u = u * factor
    vectors = torch.stack((b, c, a, u))
    gram = (vectors @ vectors.T).cpu().tolist()
    final = (b + u).to(torch.float32)
    actual_aux = final.to(torch.float64) - b
    actual_norm = float(torch.linalg.vector_norm(actual_aux))
    dot_b, dot_c = float(torch.dot(actual_aux, b)), float(torch.dot(actual_aux, c))
    if actual_norm > RHO * nb + FLOAT32_FEASIBILITY * max(1., nb):
        raise ArithmeticError("Guard cap failed after FP32 conversion")
    if dot_b < -FLOAT32_FEASIBILITY * max(1., nb * actual_norm) or dot_c < -FLOAT32_FEASIBILITY * max(1., nc * actual_norm):
        raise ArithmeticError("Guard direction failed after FP32 conversion")
    return final, {"active_set": active, "cap_factor": factor, "rho": RHO,
                   "gram_order": ["base", "classification", "auxiliary_before", "auxiliary_after_double"],
                   "gram_float64": gram, "base_norm": nb, "classification_norm": nc, "auxiliary_before_norm": na,
                   "actual_FP32_auxiliary_norm": actual_norm, "actual_FP32_dot_base": dot_b,
                   "actual_FP32_dot_classification": dot_c,
                   "tolerance": {"float64_projection_relative": DOUBLE_FEASIBILITY, "float32_cast_relative": FLOAT32_FEASIBILITY},
                   "guarantee_scope": "local additional gradient only; before clip/momentum/weight decay"}


def flat_gradient(loss, parameters, *, retain_graph):
    values = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    vector = torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(values, parameters)])
    if not bool(torch.isfinite(vector).all()):
        raise ArithmeticError("Nonfinite component gradient")
    return vector


def assign_gradient(parameters, vector):
    position = 0
    for p in parameters:
        n = p.numel()
        p.grad = vector[position:position+n].view_as(p).clone()
        position += n
    if position != vector.numel():
        raise ValueError("Parameter gradient width differs")


class GuardedCriterion(N.NativeCorrectiveLoss):
    """Original frozen scalar objective split for window-level gradient guarding."""
    def components(self, preds, batch, specs):
        self.native_targets = None
        assignment, base, _ = P.IgnoreAwareDetectionLoss.get_assigned_targets_and_loss(self, preds, batch)
        fg, gt_index, _, anchors, stride = assignment
        fg = fg.bool()
        scores = preds["scores"].permute(0, 2, 1).contiguous()
        if self.native_targets is None or self.native_targets.shape != scores.shape:
            raise ValueError("Native targets were not captured")
        keep = N.build_negative_keep_mask((anchors * stride).detach(), fg, batch["ignore_boxes_xyxy"], tuple(batch["img"].shape[-2:]))
        boxes = self.bbox_decode(anchors, preds["boxes"].permute(0, 2, 1).contiguous()) * stride
        zero = scores.sum() * 0
        loc, errors = [], []
        active = ignored = foreground = 0
        for b, spec in enumerate(specs):
            ids = torch.where(batch["batch_idx"] == b)[0]
            for local_gt in spec["positive_gt"]:
                selected = fg[b] & (gt_index[b] == local_gt)
                if not selected.any():
                    continue
                t = batch["bboxes"][ids[local_gt]] * 640
                truth = torch.cat((t[:2] - t[2:] / 2, t[:2] + t[2:] / 2))
                loc.append(self.hyp.box * (1 - N.aligned_iou(boxes[b, selected], truth)).mean())
            for e in spec["errors"]:
                anchor, cls = e["anchor"], e["category"]
                if not bool(keep[b, anchor]):
                    ignored += 1
                    continue
                logit, target = scores[b, anchor, cls], self.native_targets[b, anchor, cls]
                errors.append(self.hyp.cls * N.overconfidence_divergence(logit, target))
                active += int(logit.detach().sigmoid() > target)
                foreground += int(fg[b, anchor])
        loc_loss = torch.stack(loc).mean() if loc else zero
        error_loss = torch.stack(errors).mean() if errors else zero
        aux = .25 * (loc_loss + error_loss)
        self.last_aux = {"positive_objects": len(loc), "selected_error_channels_kept": len(errors),
                         "active_error_channels": active, "ignored_error_channels": ignored,
                         "error_anchors_currently_foreground": foreground, "localization_loss": float(loc_loss.detach()),
                         "overconfidence_loss": float(error_loss.detach()), "auxiliary_loss": float(aux.detach()),
                         "extra_positive_classification_BCE": False}
        return base.sum(), base[1], aux
