"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood

###
from scipy.interpolate import PchipInterpolator
###


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 50/ num_diffusion_timesteps#1000 / num_diffusion_timesteps
        beta_start = 0.001 #scale * 0.0001
        beta_end = 0.3 #scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=True,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.beta0 = 0.01

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas

        self.betas[0] = self.beta0

        assert len(betas.shape) == 1, "betas must be 1-D"
        # assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        self.schedule_updates = 0

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    ###########################################################################################################

        self.omega = np.cumsum(self.betas)

        self.omega_start = self.beta0 #self.omega[0]
        self.omega_end = self.omega[-1]
        
        self.lambda_increments = th.zeros(self.num_timesteps)
        self.energy_increments = th.zeros(self.num_timesteps)

        self.Energy = 0.0
        self.Lambda = 0.0

        self.space_error = 0.0

        self.D = th.zeros(self.num_timesteps)
        self.D2 = th.zeros(self.num_timesteps)
        self.n_time = th.zeros(self.num_timesteps,dtype = int)
        self.n_time[0] = 1


    def reset_counters(self):

        self.D = th.zeros(self.num_timesteps)
        self.D2 = th.zeros(self.num_timesteps)
        self.n_time = th.zeros(self.num_timesteps,dtype = int)
        self.n_time[0] = 1
    
    def resample_omega(self, dists, tau = 0.05):

        def project_omega_derivative(w, M = 1, w0 = 0.1, wN = 1.0):

            # Boundary conditions
            f_0 = w0
            f_N = wN

            assert M * (len(w) - 1) >= f_N, 'no feasiable solution'

            w_dash = np.diff(w)

            w_dash[w_dash < 0 ] = 1e-16
            w_dash[(w_dash > M )] = M

            w_dash = np.hstack([np.array([w0]),w_dash])

            w_new = np.cumsum(w_dash)

            w_new[-1] = wN

            return w_new, w_dash


        def project_omega(w, M = 1, w0 = 0.1, wN = 1.0):
            from scipy.optimize import minimize

            # Boundary conditions
            f_0 = w0
            f_N = wN

            assert M * (len(w) - 1) >= f_N, 'no feasiable solution'

            # Define the objective function (squared Euclidean norm)
            def objective(f):
                return np.sum((f - w)**2)

            # Define the constraint functions
            def boundary_0(f):
                return f[0] - f_0

            def boundary_N(f):
                return f[-1] - f_N

            def derivative_constraint(f):
                # Calculate the difference between adjacent values (finite difference)
                return -(np.max(np.abs(np.diff(f))) - M)
            
            def derivative_constraint_positive(f):
                # Calculate the difference between adjacent values (finite difference)
                return np.min(np.diff(f)) - 1e-6
            
            def constraint_positive(f):
                # Calculate the difference between adjacent values (finite difference)
                return np.min(f)

            # Initial guess for f
            f_init = w

            # Define constraints as a dictionary
            constraints = [
                {'type': 'eq', 'fun': boundary_0},  # f_0 = f_0
                {'type': 'eq', 'fun': boundary_N},  # f_N = f_N
                {'type': 'ineq', 'fun': constraint_positive} , # f'(x) <= M,
                {'type': 'ineq', 'fun': derivative_constraint} , # f'(x) <= M,
                {'type': 'ineq', 'fun': derivative_constraint_positive}  # f'(x) <= M
            ]

            # Solve the optimization problem
            result = minimize(objective, f_init, constraints=constraints, method='SLSQP')
            derivative = np.hstack([np.array([w0]),np.diff(result.x)])
            derivative[derivative < 0 ] = 1e-16
            derivative[derivative > M ] = M
            omegas = np.cumsum(derivative)


            return omegas, derivative


        print('omega resample!')

        with th.no_grad():
        
            # Step 1: Calculate the cumulative distribution of dists
            cumulative_dists = np.cumsum(dists.double())
    
            # Step 2: Generate an equally spaced new cumulative distribution
            equal_cumulative_dists = np.linspace(0, cumulative_dists[-1], len(dists), dtype = np.float64)
            
            # Step 3: Interpolate beta across the new distribution
            # Ensure beta is a numpy array for interpolation
            omega_np = self.omega.cpu().numpy() if isinstance(self.omega, th.Tensor) else self.omega
            
            # Interpolate beta values on the new, equally spaced cumulative distribution
            interpolation_function = PchipInterpolator(cumulative_dists, omega_np)
            new_omega_values = interpolation_function(equal_cumulative_dists)

            # new_omega_values, new_betas = project_omega(new_omega_values, M = 0.99, w0 = self.omega_start, wN = self.omega_end)
            new_omega_values, new_betas = project_omega_derivative(new_omega_values, M = 0.99, w0 = self.omega_start, wN = self.omega_end)
            
            # new_omega_values[0]  = self.omega_start
            # new_omega_values[-1] = self.omega_end
            
            # Update the beta tensor in the class
            self.omega = tau*th.tensor(new_omega_values, dtype=th.float64) + (1-tau)*self.omega
            self.omega = self.omega.detach().numpy()

            self.betas = tau*th.tensor(new_betas, dtype=th.float64) + (1-tau)*self.betas
            self.betas = self.betas.detach().numpy()


        
    
    def project_betas(self, max_beta = 0.999):

        def FD_central(arr):
            h = 1 #len(arr)-1
            derivative = np.zeros_like(arr)
            derivative[1:-1] = (arr[2:] - arr[:-2]) / (2*h )
            derivative[0] = (arr[1] - arr[0])/h  
            derivative[-1] = (arr[-1] - arr[-2])/h 
            return derivative
        
        self.betas = FD_central(self.omega)
        self.betas[0] = self.beta0

        n_step = len(self.betas)
        n_cut = np.sum(self.betas>max_beta)



        # omega_np = self.omega.cpu().numpy() if isinstance(self.omega, th.Tensor) else self.omega
        # n_step = len(omega_np)
        # n_cut = np.sum(self.betas>max_beta)
        if n_cut > 0:

            self.betas[self.betas>max_beta] = max_beta
            betas_left = PchipInterpolator(np.linspace(1,n_step,n_step), self.betas)(np.linspace(1,n_step-n_cut+min(2,n_cut),n_step))
            betas_right = PchipInterpolator(np.linspace(1,n_step,n_step), self.betas)(np.linspace(1,n_step-n_cut+min(1,n_cut),n_step))
            self.betas = 0.5*betas_left + 0.5*betas_right

            # self.omega = PchipInterpolator(np.linspace(1,n_step,n_step), omega_np)(np.linspace(1,n_step-n_cut,n_step))
            # self.omega_end = self.omega[-1]
            # self.betas = FD_central(self.omega)
            # self.betas[0] = self.beta0

            self.omega = np.cumsum(self.betas)




        # self.betas[self.betas>max_beta] = max_beta

        # # omega_np = self.omega.cpu().numpy() if isinstance(self.omega, th.Tensor) else self.omega
        # # n_step = len(omega_np)
        # # n_cut = len(self.betas>max_beta)

        # # interpolation_function = PchipInterpolator(np.linspace(1,n_step,n_step), omega_np)(np.linspace(1,n_step-n_cut,n_step))
    
        # self.omega = np.cumsum(self.betas)
        # self.omega = (self.omega_end/self.omega[-1])*self.omega

        # self.omega[0] = self.omega_start
        # self.omega[0] = 0.5*self.omega[0] + 0.5*self.omega_start
        # self.omega[-1] = 0.5*self.omega[-1] + 0.5*self.omega_end

    # def project_omega_beta(self):
    #     cons = []
    
    def update_alpha(self):
        """Recalculate `alpha` and `alpha_bar` based on the current `beta` values in double precision."""
        # Convert beta to double precision if it's not already
        # Use float64 for accuracy.
        betas = self.betas

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )


    # def update_schedule(self,stm1_xt,st_xt,stm1_xtm1,st_xtm1,t,tau=0.5,n_l_min=1):
        
    #     # xt = xt[t>0]
    #     # st_xt = st_xt[t>0,:]
    #     # t = t[t>0]
    
    #     # stm1_xt = model(xt,self.beta[t-1].float())

    #     #print(stm1_xt.shape, st_xt.shape)
    
    #     v_fwd = (stm1_xt - st_xt)**2
    #     v_bwd = (stm1_xtm1 - st_xtm1)**2
    #     v = 0.5*v_fwd.mean(dim=(1, 2, 3)) + 0.5*v_bwd.mean(dim=(1, 2, 3))

    #     self.D.scatter_add_(0,t.to(self.D.device),v.reshape(-1).to(self.D.device))
    #     #self.D2.scatter_add_(0,t.to(self.D2.device),(v**2).reshape(-1).to(self.D2.device))
    #     self.n_time.scatter_add_(0,t.to(self.n_time.device),th.ones_like(t).to(self.n_time.device))
    
    #     if th.all(self.n_time[1:]>n_l_min) and th.all(self.D[1:]>0) : 

    #         # print(self.D,self.n_time)

    #         # print('schedule-updat!')

    #         self.schedule_updates += 1

    #         self.D = self.D[self.n_time>0]
    #         #self.D2 = self.D[self.n_time>0]
    #         beta_interp = self.betas[self.n_time>0]
    #         self.n_time = self.n_time[self.n_time>0]

            

    #         # print(self.n_time)

    #         # print(th.sqrt(self.D/self.n_time)[1:].shape,self.betas[1:].shape,self.betas[:-1].shape)

    #         self.D = th.sqrt(self.D/self.n_time) #* th.cat([th.zeros(1),th.tensor(self.betas[1:] - self.betas[:-1])])

    #         #self.D2 = th.sqrt( (self.D2**2 - (self.D/self.n_time)**2)/ self.n_time)

    #         #self.D2[th.isnan(self.D2)] = 0 
            
    #         # self.D = torch.cat([torch.zeros(1),(self.betas[1:] - self.betas[:-1])])
    
    #         self.lambda_increments = self.D #+ self.D2

    #         self.Lambda = (1-tau)*self.Lambda + tau*th.sum(self.D)

    #         L_bar = th.sum(self.D)/(len(self.D)-1)

    #         self.space_error = (1-tau)*self.space_error + tau*th.max(th.abs(self.D[1:-1] - L_bar)/L_bar)
            
    #         self.resample_beta(self.lambda_increments,tau = tau)
    #         self.update_alpha()
    #         self.reset_counters()
    
    #         # self.D = th.zeros(self.n_steps)
    #         # self.n_time = th.zeros(self.n_steps,dtype = int)
    #         # self.n_time[0] = 1


    
    def update_schedule(self,stm1_xt,st_xt,t,tau=0.05,n_l_min=1):

        # print(stm1_xt.device)

        # print(self.sqrt_one_minus_alphas_cumprod[t.cpu().numpy()  ])
        # print(self.sqrt_one_minus_alphas_cumprod[t.cpu().numpy()  ].shape)


        stm1_xt = -stm1_xt/th.tensor(self.sqrt_one_minus_alphas_cumprod[t.cpu().numpy()-1],device = stm1_xt.device).view(-1,1,1,1)
        st_xt   = -st_xt  /th.tensor(self.sqrt_one_minus_alphas_cumprod[t.cpu().numpy()  ],device = st_xt.device).view(-1,1,1,1)
    
        sig2 = th.tensor( 1-self.alphas_cumprod[t.cpu().numpy()-1],device = stm1_xt.device).view(-1,1,1,1)

        #sig2 = th.tensor(self.posterior_variance[t.cpu().numpy()],device = stm1_xt.device).view(-1,1,1,1)

        v = sig2 * (stm1_xt - st_xt)**2

        v = v.view(v.size(0), -1).sum(1).detach().float().to(self.D.device)

        t = t.to(self.D.device)

        self.D.scatter_add_(0,t,v.reshape(-1))
        self.n_time.scatter_add_(0,t,th.ones_like(t))
    
        if th.all(self.n_time[1:] > n_l_min) and th.all(self.D[1:] > 0):
    
            self.energy_increments = self.D/self.n_time

            self.Energy = th.sum(self.energy_increments )
            
            self.D = th.sqrt(self.D/self.n_time)
    
            self.lambda_increments = self.D
    
            self.Lambda = th.sum(self.D) #(1-tau)*self.Lambda + tau*th.sum(self.D)
        
            self.resample_omega(self.lambda_increments,tau = tau)
            # self.betas = FD_central(self.omega)
            # self.betas[0] = self.beta0
            # self.project_betas()
            self.update_alpha() ###schedule LEARNHACK
    
            self.D = th.zeros(self.num_timesteps)
            self.n_time = th.zeros(self.num_timesteps,dtype = int)
            self.n_time[0] = 1
    

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    # def _scale_timesteps(self, t):
    #     if self.rescale_timesteps:
    #         return t.float() #* (1000.0 / self.num_timesteps)
    #     return t

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            #print(self.betas, t)
            # return t.float()
            return _extract_into_tensor(self.omega, t, t.shape) #(_extract_into_tensor(self.omega, t, t.shape) - self.omega[0])/(self.omega[-1] - self.omega[0]) * 1000
            #return t.float() * (1000.0 / self.num_timesteps)
        return t


    # def _scale_timesteps(self, t):
    #     if self.rescale_timesteps:
    #         #print(self.betas, t)
    #         return (_extract_into_tensor(self.betas, t, t.shape) - self.betas[0])/(self.betas[-1] - self.betas[0]) * 1000
    #         #return t.float() * (1000.0 / self.num_timesteps)
    #     return t

    def p_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to nam the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            #print(x_t.shape)
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms, x_t, model_output

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
