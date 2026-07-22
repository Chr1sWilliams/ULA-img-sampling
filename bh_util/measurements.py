"""This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n."""

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch

# from motionblur.motionblur import Kernel
import numpy as np

# from util.resizer import Resizer
# from util.img_utils import Blurkernel, fft2_m

# =================
# Operation classes
# =================

__OPERATOR__ = {}


def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls

    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass

    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name="noise")
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device

    def forward(self, data):
        return data

    def transpose(self, data):
        return data

    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data)


@register_operator(name="interferometer")
class InterferometerOperator(NonLinearOperator):
    def __init__(self, uvfile, img_size, device, psize=4.848136811094e-12):
        self.device = device
        (
            self.cphase,
            self.camp,
            self.cptraj,
            self.cltraj,
            self.dmat_cp,
            self.dmat_lc,
            self.nufft_ob,
            self.cov_cp,
            self.cov_lc,
        ) = self.prepare_VLBI_interferometer(uvfile, img_size, device, psize)

    def prepare_VLBI_interferometer(self, uvfile, img_size, device, psize):
        """
        VLBI interferometry requires external codes.
        """
        from .closure_constructions import get_minimal_cphases, get_minimal_logcamps
        import ehtim as eh
        import torchkbnufft as tkbn
        # psize = 4.848136811094e-12#7.575213767334451e-12

        # load obs file
        obs = eh.obsdata.load_uvfits(uvfile)
        obs.add_scans()
        obs = (
            obs.avg_coherent(0.0, scan_avg=True)
            .add_fractional_noise(0.01)
            .flag_uvdist(uv_min=0.1e9)
        )  # .add_fractional_noise(0.01).flag_uvdist(uv_min=0.1e9)
        # construct log closures and amplitudes
        cp_data, dmat_cp, uvcp, cov_cp = get_minimal_cphases(obs)
        lc_data, dmat_lc, uvlc, cov_lc = get_minimal_logcamps(obs)

        # convert numpy object to torch
        cptraj = torch.tensor(uvcp.T * psize * 2 * np.pi).to(torch.float).to(device)
        cltraj = torch.tensor(uvlc.T * psize * 2 * np.pi).to(torch.float).to(device)

        dmat_cp = sparse_numpy_to_tensor(dmat_cp, device)
        dmat_lc = sparse_numpy_to_tensor(dmat_lc, device)

        cphase = torch.tensor(cp_data["cphase"], device=device)
        camp = torch.tensor(lc_data["camp"], device=device)

        # cov_cp = torch.tensor(np.linalg.inv(np.linalg.cholesky(cov_cp))).to(torch.complex128).to(device)
        # cov_lc = torch.tensor(np.linalg.inv(cov_lc)).to(device)
        # cov_lc = torch.tensor(np.linalg.inv(np.linalg.cholesky(cov_lc))).to(device)
        # cov_cp = torch.tensor(cov_cp).to(torch.complex128).to(device)
        # cov_lc = torch.tensor(cov_lc).to(device)

        cov_cp = (
            (torch.linalg.cholesky(torch.from_numpy(cov_cp)))
            .to(torch.complex128)
            .to(device)
        )
        cov_lc = (torch.linalg.cholesky(torch.from_numpy(cov_lc))).to(device)

        # construct non-uniform FFT object
        nufft_ob = tkbn.KbNufft(im_size=(img_size, img_size)).to(device)
        return cphase, camp, cptraj, cltraj, dmat_cp, dmat_lc, nufft_ob, cov_cp, cov_lc

    def forward(self, data, **kwargs):
        img = data.to(torch.complex64)
        viscp = self.nufft_ob(img, self.cptraj)

        # print('viscp',viscp.shape,viscp )
        vislc = self.nufft_ob(img, self.cltraj)

        # viscp = torch.conj(viscp)
        # vislc = torch.conj(vislc)

        # print(self.dmat_lc.shape, vislc.shape)

        # print(self.dmat_cp.shape,torch.transpose(self.dmat_cp,0,1).shape )
        # print(self.dmat_cp.shape, torch.angle(viscp.squeeze(1)).shape )

        # print( self.dmat_cp.shape, torch.transpose(torch.angle(viscp.squeeze(1)),0,1).shape)

        cp_model = torch.matmul(
            self.dmat_cp, torch.transpose(torch.angle(viscp.squeeze(1)), 0, 1)
        )
        lc_model = torch.matmul(
            self.dmat_lc, torch.transpose(torch.log(torch.abs(vislc.squeeze(1))), 0, 1)
        )

        # old and works
        # cp_model = torch.matmul(self.dmat_cp, torch.angle(viscp.squeeze()))
        # lc_model = torch.matmul(self.dmat_lc, torch.log(torch.abs(vislc.squeeze())))
        return torch.transpose(cp_model, 0, 1), torch.transpose(lc_model, 0, 1)


def sparse_numpy_to_tensor(ar, device):
    # construct torch closure sparse matrix
    values = ar.data
    indices = np.vstack((ar.row, ar.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = ar.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense().to(device)


# =============
# Noise classes
# =============


__NOISE__ = {}


def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls

    return wrapper


def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser


class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)

    @abstractmethod
    def forward(self, data):
        pass


@register_noise(name="clean")
class Clean(Noise):
    def forward(self, data):
        return data


@register_noise(name="gaussian")
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma

    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name="poisson")
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        """
        Follow skimage.util.random_noise.
        """

        # TODO: set one version of poisson

        # version 3 (stack-overflow)
        import numpy as np

        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(
            np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate
        )
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0

        # return data.clamp(low_clip, 1.0)
