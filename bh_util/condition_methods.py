from abc import ABC, abstractmethod
import torch

__CONDITIONING_METHOD__ = {}


def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls

    return wrapper


def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser

    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == "gaussian":
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        elif self.noiser.__name__ == "poisson":
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement - Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError

        return norm_grad, norm

    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass


@register_conditioning_method(name="vanilla")
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t


@register_conditioning_method(name="projection")
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name="mcg")
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(
        self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs
    ):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(
            x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs
        )
        x_t -= norm_grad * self.scale

        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm


@register_conditioning_method(name="ps")
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(
            x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs
        )
        x_t -= norm_grad * self.scale
        return x_t, norm


@register_conditioning_method(name="ps+")
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get("num_sampling", 5)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling

        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm


@register_conditioning_method(name="VLBI")
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser="gaussian")
        self.scale = kwargs.get("scale", 0)
        self.NORM = []

    def grad_and_value(self, x_prev, x_0_hat, **kwargs):
        cp_model, lc_model = self.operator.forward(x_0_hat, **kwargs)
        diff_cp = (
            torch.conj(
                self.operator.cov_cp
                @ (torch.exp(1j * self.operator.cphase) - torch.exp(1j * cp_model))
            ).T
            @ (
                self.operator.cov_cp
                @ (torch.exp(1j * self.operator.cphase) - torch.exp(1j * cp_model))
            )
            / 2
        )
        # diff_lc = -(self.operator.camp - lc_model).T @ self.operator.cov_lc\
        #          @ (self.operator.camp - lc_model)/2
        diff_lc = (
            -(self.operator.cov_lc @ (self.operator.camp - lc_model)).T
            @ (self.operator.cov_lc @ (self.operator.camp - lc_model))
            / 2
        )
        norm = diff_cp + diff_lc
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        self.NORM.append(norm.detach().cpu().data)
        return norm_grad, norm

    def conditioning(self, x_prev, x_t, x_0_hat, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, **kwargs)
        x_t -= norm_grad * self.scale
        return x_t, norm


@register_conditioning_method(name="+VLBI")
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser="gaussian")
        self.num_sampling = kwargs.get("num_sampling", 5)
        self.scale = kwargs.get("scale", 1.0)

    def grad_and_value(self, x_prev, x_0_hat, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            cp_model, lc_model = self.operator.forward(x_0_hat_noise, **kwargs)
            diff_cp = (
                (
                    -torch.exp(
                        -1j * (self.operator.cphase - cp_model)
                        + torch.exp(1j * (self.operator.cphase - cp_model))
                    )
                ).T
                @ self.operator.cov_cp
                @ (
                    -torch.exp(
                        -1j * (self.operator.cphase - cp_model)
                        + torch.exp(1j * (self.operator.cphase - cp_model))
                    )
                    / 2
                ).to(torch.float64)
            )
            diff_lc = (
                -(self.operator.camp - lc_model).T
                @ self.operator.cov_lc
                @ (self.operator.camp - lc_model)
                / 2
            )
            norm += (diff_cp + diff_lc) / self.num_sampling
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        return norm_grad, norm

    def conditioning(self, x_prev, x_t, x_0_hat, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, **kwargs)
        x_t -= norm_grad * self.scale
        return x_t, norm
