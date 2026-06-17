"""Base class for optimizers"""


class Optimizer:
    """Base optimizer with optional per-group learning rates.

    ``params`` may be a flat list of tensors or a list of dicts::

        optimizer = Adam(model.parameters(), learning_rate=1e-3)
        optimizer = Adam([
            {"params": head_params, "lr": 1e-3},
            {"params": body_params, "lr": 1e-4},
        ])
    """

    def __init__(self, params, *, learning_rate=0.01, lr=None):
        default_lr = lr if lr is not None else learning_rate
        if params and isinstance(params[0], dict):
            self.param_groups = []
            for group in params:
                if "params" not in group:
                    raise ValueError("each param group must include 'params'")
                lr_value = group.get("lr", group.get("learning_rate", default_lr))
                entry = {"params": list(group["params"]), "lr": lr_value}
                if "weight_decay" in group:
                    entry["weight_decay"] = float(group["weight_decay"])
                self.param_groups.append(entry)
        else:
            self.param_groups = [{"params": list(params), "lr": default_lr}]

        self.params = [param for group in self.param_groups for param in group["params"]]

    @property
    def learning_rate(self):
        """Public function learning_rate."""
        return self.param_groups[0]["lr"]

    @learning_rate.setter
    def learning_rate(self, value):
        """Public function learning_rate."""
        for group in self.param_groups:
            group["lr"] = value

    def get_lr(self):
        """Current learning rate for each param group."""
        return [group["lr"] for group in self.param_groups]

    def set_lr(self, lrs):
        """Set learning rates, one value per param group."""
        if len(lrs) != len(self.param_groups):
            raise ValueError(
                f"expected {len(self.param_groups)} learning rates, got {len(lrs)}"
            )
        for group, lr in zip(self.param_groups, lrs):
            group["lr"] = lr

    def step(self):
        """Public function step."""
        raise NotImplementedError("Optimizer.step must be implemented by subclasses.")

    def zero_grad(self, set_to_none: bool = False):
        """Clear parameter gradients."""
        for param in self.params:
            param.zero_grad(set_to_none=set_to_none)
