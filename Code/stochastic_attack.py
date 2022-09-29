import torch

from torchattacks.attack import Attack
import rep_transformations as rt


class STOCHASTIC_ATTACK(Attack):
    r"""
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = STOCHASTIC_ATTACK(model, eps=8/255)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, transf=None, eps=8/255):
        if transf is None:
            self.transf = rt.Identity()
        else:
            self.transf = transf
        super().__init__("RAND_"+self.transf.name, model)
        self.eps = eps
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images_transf = self.transf(images).clone().detach().to(self.device)

        adv_images_transf = images_transf + torch.randn_like(images_transf).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(self.transf.inv(adv_images_transf), min=0, max=1).detach()

        return adv_images