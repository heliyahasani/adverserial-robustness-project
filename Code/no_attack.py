from torchattacks.attack import Attack


class NO_ATTACK(Attack):
    r"""
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`.
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = NO_ATTACK(model, eps=8/255)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, eps=None):
        super().__init__("NO_ATTACK", model)
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        adv_images = images.clone().detach().to(self.device)

        return adv_images