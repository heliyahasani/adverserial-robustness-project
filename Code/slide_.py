import torch
import torch.nn as nn

from torchattacks.attack import Attack
import rep_transformations as rt


class SLIDE_(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, transf=None, percentile=0.01, eps=0.3,
                 alpha=2/255, steps=40, random_start=True):
        if transf is None:
            self.transf = rt.Identity()
        else:
            self.transf = transf
        super().__init__("SLIDE_"+str(percentile), model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
        self.percentile = percentile

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images_transf = self.transf(images).clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images_transf, labels)

        loss = nn.CrossEntropyLoss()

        adv_images_transf = images_transf.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            delta = torch.empty_like(adv_images_transf).uniform_(-self.eps, self.eps)
            delta_shape = delta.shape
            delta_flat = torch.reshape(delta, (delta_shape[0], delta_shape[1], -1))
            delta_norm1 = torch.sum(torch.abs(delta_flat), dim=-1, keepdim=True)
            delta_flat = delta_flat/delta_norm1
            delta = torch.reshape(delta_flat, delta_shape).detach()
            adv_images_transf = adv_images_transf + delta
            adv_images_transf = torch.clamp(adv_images_transf, min=0, max=1)

        for _ in range(self.steps):
            adv_images_transf.requires_grad = True
            adv_images = torch.clamp(self.transf.inv(adv_images_transf), min=0, max=1)
            outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images_transf,
                                       retain_graph=False, create_graph=False)[0]
            
            #SLIDE algorithm
            grad_shape = grad.shape
            grad_flat = torch.reshape(grad, (grad_shape[0], grad_shape[1], -1))
            grad_norm1 = torch.sum(grad_flat, dim=-1, keepdim=True)
            grad_normalized = grad_flat/grad_norm1
            grad_threshold_flat = grad_normalized.sgn()*(torch.abs(grad_normalized)>self.percentile)
            grad_threshold = torch.reshape(grad_threshold_flat, grad_shape)

            adv_images_transf = adv_images_transf.detach() + self.alpha*grad_threshold
            delta = (adv_images_transf - images_transf)
            delta_flat = torch.reshape(delta, (delta_shape[0], delta_shape[1], -1))
            delta_norm1 = torch.sum(torch.abs(delta_flat), dim=-1, keepdim=True)
            delta_flat = delta_flat/delta_norm1
            delta = torch.reshape(delta_flat, delta_shape)
            adv_images_transf = (images_transf + delta).detach()
            
        adv_images = torch.clamp(self.transf.inv(adv_images_transf), min=0, max=1).detach()

        return adv_images.detach()