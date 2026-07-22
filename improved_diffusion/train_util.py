import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from scipy import interpolate

import torchvision.transforms as transforms

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

import matplotlib.pyplot as plt
import wandb

from io import BytesIO
from PIL import Image

print('is dev1')
print(dist_util.dev() == th.device('cuda:0'))
print(dist_util.dev())

if dist_util.dev() == th.device('cuda:0'):

    wandb.init(
        # set the wandb project where this run will be logged
        project="diffusion_schedules",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.02,
        "architecture": "U-Net",
        "dataset": "cifar",
        "epochs": 10,
        }
    )

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        schedule_tune = False,
        schedule_lr = 0.01,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.schedule_tune = schedule_tune
        self.schedule_lr = schedule_lr

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()


        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            print(dist_util.dev())
            print(th.cuda.device_count())
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            #####################################################################
            to_gray = transforms.Grayscale(num_output_channels=1)
            batch = to_gray(batch) #MNIST
            # batch = th.log(batch + 2)
            #####################################################################

            self.run_step(batch, cond)
            if self.step % self.log_interval == 0 and dist_util.dev() == th.device('cuda:0'):
                # print(self.diffusion.D)
                # print(self.diffusion.Lambda)

                betas = self.diffusion.betas
                data_table = wandb.Table(columns=["x", "y"])
                for i, beta in enumerate(betas):
                    data_table.add_data(i, beta)
                wandb.log({"betas_plot": wandb.plot.line(data_table, "x", "y", title="Betas Vector")})

                omega = self.diffusion.omega
                data_table = wandb.Table(columns=["x", "y"])
                for i, om in enumerate(omega):
                    data_table.add_data(i, om)
                wandb.log({"omega_plot": wandb.plot.line(data_table, "x", "y", title="Omega Vector")})



                lambda_incre = self.diffusion.lambda_increments
                data_table = wandb.Table(columns=["x", "y"])
                for i, lam in enumerate(lambda_incre):
                    data_table.add_data(i, lam)

                wandb.log({"lambda_plot": wandb.plot.line(data_table, "x", "y", title="Lambda Vector")})

                energy_incre = self.diffusion.energy_increments
                data_table = wandb.Table(columns=["x", "y"])
                for i, lam in enumerate(energy_incre):
                    data_table.add_data(i, lam)

                
                wandb.log({"energy_plot": wandb.plot.line(data_table, "x", "y", title="Energy Vector")})

                # Log the line plot of betas using the wandb.Table
                wandb.log({"lambda": self.diffusion.Lambda})
                wandb.log({"energy": self.diffusion.Energy})
                wandb.log({"lambda max relative error": self.diffusion.space_error})
                # logger.dumpkvs()

                if self.step % self.save_interval == 0:

                          
                    with th.no_grad():
                        self.model.eval()  # Set the model to evaluation mode
                        samples = self.generate_samples(use_ddim=True)#, device=dist_util.dev(), num_samples=10, image_size=args.image_size, class_cond=args.class_cond, use_ddim=args.use_ddim)
                        self.model.train()  # Set back to training mode

                        samples = samples.cpu().numpy()

                        # samples = np.exp(sample) - 2 

                        print(samples.shape)

                        for i in range(len(samples)):

                            wandb.log({f"generated_samples_{i}":wandb.Image(samples[i])} )


                        # images = wandb.Image(samples.cpu().numpy())
                        # wandb.log({"examples": images})

                        # images_to_log = [wandb.Image(sample.cpu().numpy(), caption=f"Sample {i}") for i, sample in enumerate(samples)]
                        # wandb.log({"generated_samples": images_to_log})

                

                if self.diffusion.schedule_updates > 10 and False:

                    self.diffusion.schedule_updates = 0

                    with th.no_grad():
                        i = int(np.log2(self.diffusion.num_timesteps))

                        if self.diffusion.space_error < 0.1:
                            i += 1

                        nstep = 2**i




                        interpolation = interpolate.interp1d(np.linspace(self.diffusion.betas[0],self.diffusion.betas[-1],len(self.diffusion.betas)), self.diffusion.betas, kind='linear')
                        betas = interpolation(np.linspace(self.diffusion.betas[0],self.diffusion.betas[-1],nstep))
                        self.diffusion.betas = np.array(betas, dtype = np.float64)

                        # assert len(betas.shape) == 1, "betas must be 1-D"
                        # assert (betas > 0).all() and (betas <= 1).all()



                        self.diffusion.num_timesteps = int(betas.shape[0])

                        print('beta size is now', self.diffusion.num_timesteps)

                        self.diffusion.update_alpha()
                        self.diffusion.reset_counters()



                
                # images = []
                # for out in self.diffusion.ddim_sample_loop_progressive(self.ddp_model,(1,)+batch.shape[1:] ):  # fill in your arguments
                #     # Convert the tensor to a PIL image. You might need to adjust normalization and tensor shape
                #     img = out["sample"].squeeze(0).cpu().detach()  # Assuming batch size of 1 for simplicity
                #     img = (img + 1) / 2  # Assuming output in [-1, 1], normalize to [0, 1]
                #     img = img.permute(1, 2, 0).numpy()  # Assuming img is CHW, convert to HWC for image
                #     img = (img * 255).astype(np.uint8)
                #     pil_img = Image.fromarray(img)
                #     images.append(pil_img)

                # image_dict = {}
                # for i, img in enumerate(images):
                #     # Convert PIL image to wandb.Image
                #     image_dict[f"step_{i}"] = wandb.Image(img)

                # wandb.log(image_dict)

            if self.step % self.save_interval == 0:
                # Assuming self.diffusion.num_timesteps is accessible and indicates the total timesteps
                # total_steps = self.diffusion.num_timesteps
                # sample_interval = max(1, total_steps // 10)

                # # Initialize the list for selected images
                # selected_images = []

                # # Generate images and select 10 uniformly
                # step_counter = 0
                # for out in self.diffusion.ddim_sample_loop_progressive(self.ddp_model, (1,) + batch.shape[1:]):  # Fill in your arguments
                #     if step_counter % sample_interval == 0 or step_counter == total_steps - 1:
                #         # Convert the tensor to a PIL image. Adjust normalization and tensor shape as needed
                #         img = out["sample"].squeeze(0).cpu().detach()  # Assuming batch size of 1 for simplicity
                #         img = (img + 1) / 2  # Normalize to [0, 1] if output is in [-1, 1]
                #         img = img.permute(1, 2, 0).numpy()  # Convert CHW to HWC for image
                #         img = (img * 255).astype(np.uint8)
                #         pil_img = Image.fromarray(img)
                #         selected_images.append(pil_img)
                #         if len(selected_images) >= 10:  # Break after collecting enough images
                #             break
                #     step_counter += 1

                # # Log selected images to wandb
                # image_dict = {}
                # for i, img in enumerate(selected_images):
                #     image_dict[f"step_{i}"] = wandb.Image(img)

                # wandb.log(image_dict)
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses,xt,st_xt = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses,xt,st_xt = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()

            if dist_util.dev() == th.device('cuda:0') and self.step % self.log_interval == 0:
                #wandb.log({"loss": loss})
                wandb.log({"loss": loss}, step=self.step)
                log_loss_dict(
                    self.diffusion, t, {k: v * weights for k, v in losses.items()}
                )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

                # print(self.diffusion.sqrt_one_minus_alphas_cumprod[t.cpu().numpy()])
                # print(t)
            if self.step > 10 and self.schedule_tune:
                with th.no_grad():

                    xt = xt[t>0]
                    st_xt = st_xt[t>0,:]
                    micro = micro[t>0]
                    t = t[t>0]

                    stm1_xt = self.ddp_model(xt, self.diffusion._scale_timesteps(t-1), **micro_cond)
                    #st_xt_test = self.ddp_model(xt, self.diffusion._scale_timesteps(t), **micro_cond)


                    #print(xt.shape, st_xt.shape, stm1_xt.shape, st_xt_test.shape)

                    #model(x_t, self._scale_timesteps(t), **model_kwargs)



                    #xtm1 = self.diffusion.q_sample(micro, t-1, noise=th.randn_like(micro))
                
                    #stm1_xt = self.ddp_model(xt, self.diffusion._scale_timesteps(t-1), **micro_cond)

                    #st_xtm1 = self.ddp_model(xtm1, self.diffusion._scale_timesteps(t), **micro_cond)
                    #stm1_xtm1 = self.ddp_model(xtm1, self.diffusion._scale_timesteps(t-1), **micro_cond)

                    #self.diffusion.update_schedule(stm1_xt,st_xt,stm1_xtm1,st_xtm1,t,tau=0.001,n_l_min=5)
                    self.diffusion.update_schedule(stm1_xt,st_xt,t,tau=self.schedule_lr,n_l_min=24)

    def generate_samples(self, device = None, num_samples=9, image_size=128, class_cond=False, use_ddim=False):
        
        if device is None:
            device = next(self.model.parameters()).device
        
        model_kwargs = {}
        if class_cond:
            # Assuming you've defined NUM_CLASSES somewhere
            classes = th.randint(low=0, high=NUM_CLASSES, size=(num_samples,), device=device)
            model_kwargs["y"] = classes
        sample_fn = self.diffusion.p_sample_loop if not use_ddim else self.diffusion.ddim_sample_loop
        samples = sample_fn(
            self.model,
            #####################################################################
            #(num_samples, 3, image_size, image_size), #for colour
            (num_samples, 1, image_size, image_size), #MNIST
            #####################################################################
            clip_denoised=True,
            model_kwargs=model_kwargs,
        )
        samples = ((samples + 1) * 127.5).clamp(0, 255).to(th.uint8)
        samples = samples.permute(0, 2, 3, 1)  # Change to NHWC for easier handling
        return samples.contiguous()

    
    
    
    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    
    # def save(self):
    #     def save_checkpoint(rate, params):
    #         state_dict = self._master_params_to_state_dict(params)

    #         if dist.get_rank() == 0:
    #             logger.log(f"saving model {rate}...")
    #             if not rate:
    #                 model_filename = f"model{(self.step+self.resume_step):06d}.pt"
    #                 betas_filename = f"betas{(self.step+self.resume_step):06d}.npy"
    #             else:
    #                 model_filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
    #                 betas_filename = f"ema_betas_{rate}_{(self.step+self.resume_step):06d}.npy"
                
    #             # Save the PyTorch model state dict
    #             with bf.BlobFile(bf.join(get_blob_logdir(), model_filename), "wb") as f:
    #                 th.save(state_dict, f)
                
    #             # Save the betas NumPy array
    #             betas_path = bf.join(get_blob_logdir(), betas_filename)
    #             with bf.BlobFile(betas_path, "wb") as f:
    #                 np.save(f, self.ddpm.betas)

    #             logger.log(f"Saved model to {model_filename} and betas to {betas_filename}")

    #     dist.barrier()
            


    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        # Save model and EMA checkpoints
        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # Save optimizer state
        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

            # Save ddpm.betas as a separate file
            betas_filename = f"ddpm_betas{(self.step+self.resume_step):06d}.npy"
            with bf.BlobFile(bf.join(get_blob_logdir(), betas_filename), "wb") as f:
                # Use numpy.save to write the array directly to the file
                np.save(f, self.diffusion.betas)
            logger.log(f"Saved ddpm.betas to {betas_filename}")
             # Save ddpm.betas as a separate file
            omegas_filename = f"ddpm_omegas{(self.step+self.resume_step):06d}.npy"
            with bf.BlobFile(bf.join(get_blob_logdir(), omegas_filename), "wb") as f:
                # Use numpy.save to write the array directly to the file
                np.save(f, self.diffusion.omega)
            logger.log(f"Saved ddpm.omega to {omegas_filename}")

        # Ensure synchronization before proceeding
        dist.barrier()

    # def save(self):
    #     def save_checkpoint(rate, params):
    #         state_dict = self._master_params_to_state_dict(params)
    #         if dist.get_rank() == 0:
    #             logger.log(f"saving model {rate}...")
    #             if not rate:
    #                 filename = f"model{(self.step+self.resume_step):06d}.pt"
    #             else:
    #                 filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
    #             with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
    #                 th.save(state_dict, f)

    #     save_checkpoint(0, self.master_params)
    #     for rate, params in zip(self.ema_rate, self.ema_params):
    #         save_checkpoint(rate, params)

    #     if dist.get_rank() == 0:
    #         with bf.BlobFile(
    #             bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
    #             "wb",
    #         ) as f:
    #             th.save(self.opt.state_dict(), f)

    #     dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
