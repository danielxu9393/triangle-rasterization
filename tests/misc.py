from dataclasses import asdict

import torch


# If PyTorch were cooler, this would be pytree_allclose...
def dataclass_allclose(a, b, rtol=1e-5) -> bool:
    # First, check if the keys are identical.
    if set(asdict(a)) != set(asdict(b)):
        return False

    # Then, check all of the float and int values.
    for key, value_a in asdict(a).items():
        value_b = getattr(b, key)

        if isinstance(value_a, torch.Tensor):
            # Check equality for floats and complex numbers.
            if value_a.is_floating_point() or value_a.is_complex():
                if not torch.allclose(value_a, value_b, rtol=rtol):
                    return False

            # Check equality for integers.
            elif not (value_a == value_b).all():
                return False

    return True


def grad_allclose(a, b, **kwargs) -> bool:
    # None is equivalent to all zeros.
    if a is None:
        return b is None or (b == 0).all()
    if b is None:
        return a is None or (a == 0).all()

    return torch.allclose(a, b, **kwargs)
