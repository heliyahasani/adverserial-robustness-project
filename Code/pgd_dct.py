import torch
import torch.nn as nn

from torchattacks.attack import Attack
import torch_dct as dct


class PGD_DCT(Attack):
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
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, eps=0.3,
                 alpha=2/255, steps=40, random_start=True):
        super().__init__("PGD_DCT", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images_dct = dct.dct(images).clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images_dct, labels)

        loss = nn.CrossEntropyLoss()

        adv_images_dct = images_dct.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images_dct = adv_images_dct + torch.empty_like(adv_images_dct).uniform_(-self.eps, self.eps).detach()
            adv_images_dct = adv_images_dct.detach()

        for _ in range(self.steps):
            adv_images_dct.requires_grad = True
            adv_images = torch.clamp(dct.idct(adv_images_dct), min=0, max=1)
            outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images_dct,
                                       retain_graph=False, create_graph=False)[0]

            adv_images_dct = adv_images_dct.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images_dct - images_dct, min=-self.eps, max=self.eps)
            adv_images_dct = (images_dct + delta).detach()

        return torch.clamp(dct.idct(adv_images), min=0, max=1).detach()