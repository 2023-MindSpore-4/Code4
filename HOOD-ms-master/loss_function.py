import mindspore as ms
from mindspore import nn, ops

from bypass_bn import disable_running_stats, enable_running_stats


def cal_stats(out):
    mu = ops.ReduceMean(out, axis=0)
    logvar = ops.log(ms.Tensor.var(out, axis=0))
    return mu, logvar


def compute_rec(reconstructed_inputs, original_inputs):
    return nn.MSELoss(reconstructed_inputs, original_inputs)


def cal_entropy(out):
    # entropy
    return - ((out.softmax(1) * nn.LogSoftmax(out, 1)).sum(1)).mean()


def cal_vae_loss(mu, logvar):
    return - 0.01 * ops.ReduceMean(1 + logvar - mu.pow(2) - logvar.exp())


def cal_content_classification(content_pred, class_label, mask=None):
    if mask:
        content_cls_loss = (nn.CrossEntropyLoss(content_pred, class_label, reduction="none",
                                                ignore_index=-1) * mask.float()).mean()
    else:
        content_cls_loss = nn.CrossEntropyLoss(content_pred, class_label)

    return content_cls_loss


def cal_style_classification(style_pred, aug_label):
    style_cls_loss = nn.CrossEntropyLoss(style_pred, aug_label)

    return style_cls_loss


def cal_reconstruction(args, rec_content, rec_style, rec_target):
    content_rec_loss = args.rec_coef * (compute_rec(rec_content, rec_target))
    style_rec_loss = args.rec_coef * (compute_rec(rec_style, rec_target))

    return content_rec_loss, style_rec_loss


def cal_ssl(args, logits_all, mask_probs):
    logits_w, logits_s = logits_all.chunk(2)
    pseudo_label = ops.Softmax(logits_w.detach() / args.T, axis=-1)
    max_probs, max_target = ops.ArgMaxWithValue(pseudo_label, axis=-1)
    fix_mask = max_probs.ge(args.threshold).float()
    ssl_consistency_loss = (nn.CrossEntropyLoss(logits_s, max_target, reduction='none') * fix_mask).mean()
    mask_probs.update(fix_mask.mean().item())

    return ssl_consistency_loss


def cal_ood(args, logits_open_all, targets_l, b_size=None, negative=False):
    logits_open_all = logits_open_all.view(logits_open_all.size(0), 2, -1)
    logits_open_all = ops.Softmax(logits_open_all, 1)

    if not negative:
        open_target = ops.Zeros((targets_l.size(0), logits_open_all.size(2))).to(targets_l.device)
        label_range = ms.numpy.arange(0, targets_l.size(0)).long()
        open_target[label_range, targets_l] = 1
        open_target_nega = 1 - open_target

        if logits_open_all.size(0) > 2 * b_size:
            logits_open_l = logits_open_all[:2 * b_size]
            logits_open_u_w1, logits_open_u_w2 = logits_open_all[2 * b_size:].chunk(2)
            # ood consistency
            ood_loss = args.lambda_socr * ops.ReduceMean(ops.ReduceSum(ops.ReduceSum(ops.abs(
                logits_open_u_w1 - logits_open_u_w2) ** 2, 1), 1))

            # entropy minimization
            ood_loss += 0.5 * args.lambda_oem * (ops.ReduceMean(
                ops.ReduceMean(ops.ReduceSum(-logits_open_u_w1 * ops.Log(logits_open_u_w1 + 1e-8), 1), 1)) + \
                                                 ops.ReduceMean(ops.ReduceMean(
                                                     ops.ReduceSum(-logits_open_u_w2 * ops.Log(logits_open_u_w2 + 1e-8),
                                                                   1), 1)))

            # open classification
            ood_loss += ops.ReduceMean(ops.ReduceSum(-ops.Log(logits_open_l[:, 1, :] + 1e-8) * open_target, 1) +
                                       ops.ArgMaxWithValue(-ops.Log(logits_open_l[:, 0, :] + 1e-8) * open_target_nega,
                                                           1)[0])
        else:
            # open classification
            ood_loss = ops.ReduceSum(ops.ReduceSum(-ops.Log(logits_open_all[:, 1, :] + 1e-8) * open_target, 1) +
                                     ops.ArgMaxWithValue(-ops.Log(logits_open_all[:, 0, :] + 1e-8) * open_target_nega,
                                                         1)[0])

    else:
        open_target = ops.Ones((targets_l.size(0), logits_open_all.size(2))).to(targets_l.device)
        # negative detection
        ood_loss = ops.ReduceSum(ops.ArgMaxWithValue(-ops.Log(logits_open_all[:, 0, :] + 1e-8) * open_target, 1)[0])

    return ood_loss


def augmentation(args, mean, std, model_c, model_s, inputs, class_label, domain_label=None, aug_type='benign',
                 domain_targeted=False):
    """
        This function computs the adversarial examples by augmenting content and style.
    """
    mean = ms.Tensor(mean).view(3, 1, 1)
    std = ms.Tensor(std).view(3, 1, 1)
    upper_limit = ((1 - mean) / std).to(args.device)
    lower_limit = ((0 - mean) / std).to(args.device)

    disable_running_stats(model_c)
    disable_running_stats(model_s)
    model_c.eval()
    model_s.eval()

    perturbations = ops.ZerosLike(inputs)
    perturbations.uniform_(-0.01, 0.01)
    perturbations.data.clamp_(lower_limit - inputs, upper_limit - inputs)
    perturbations.requires_grad = True

    with ops.stop_gradient():
        init_cls, _, init_content, _ = model_c(inputs)
        init_cls_hat = model_s.disentangle(init_content)
        init_dom, _, _, _ = model_s(inputs)

    init_cls_target = init_cls.max(axis=1)[1]
    init_dom_target = init_dom.max(axis=1)[1]

    content_losses = 0
    style_losses = 0
    # adv augmentation
    for i in range(args.adv_step):
        step_size = \
            args.adv_magnitude * 0.1 ** (i / args.adv_step)
        if perturbations.grad is not None:
            perturbations.grad.data.zero_()

        adv_inputs = inputs + perturbations

        cls_pred, _, adv_content, _ = model_c(adv_inputs)
        cls_pred_hat = model_s.disentangle(adv_content)
        target_adv = ops.Log((ops.GatherD(cls_pred.softmax(1), 1, init_cls_target.view(-1, 1))).squeeze(axis=1))

        if aug_type == 'malign':
            # change content
            content_loss = - nn.CrossEntropyLoss(cls_pred, init_cls_target) + nn.MSELoss(cls_pred_hat, init_cls_hat)
        else:
            # maintain content
            content_loss = nn.MSELoss(adv_content, init_content) - nn.CrossEntropyLoss(cls_pred_hat, init_dom_target)
        content_losses += content_loss.item()
        content_loss.backward()

        grad = perturbations.grad.data
        grad_norm = grad / (grad.reshape(grad.size()[0], -1).norm(axis=1)[:, None, None, None] + 1e-8)
        with ops.stop_gradient():
            target_after = ops.Log(ops.GatherD(ops.softmax(
                model_c(inputs + perturbations - grad_norm * 0.1)[0], axis=1),
                1, init_cls_target.view(-1, 1)).squeeze(axis=1))
            dist_grads = ops.abs(target_adv - target_after) / 0.1
            norm = step_size / (dist_grads + 1e-4)
        perturbation_update = -grad_norm * norm[:, None, None, None]
        perturbations.data = (perturbations + perturbation_update).clamp_(lower_limit - inputs, upper_limit - inputs)

    adv_inputs = inputs + perturbations

    enable_running_stats(model_c)
    enable_running_stats(model_s)
    model_c.train()
    model_s.train()
    return adv_inputs.detach(), content_losses, style_losses