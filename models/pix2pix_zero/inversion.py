import sys
import numpy as np
import torch
import torch.nn.functional as F
from random import randrange
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffusers import DDIMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
sys.path.insert(0, "src/utils")
from models.pix2pix_zero.base_pipeline import BasePipeline
from models.pix2pix_zero.cross_attention import prep_unet
from .attention_control import register_attention_control
import torch.nn.functional as nnf
from torch.optim.adam import Adam
from utils.utils import slerp_tensor, image2latent, latent2image
from tqdm import tqdm


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class DDIMInversion(BasePipeline):

    def auto_corr_loss(self, x, random_shift=True):
        B,C,H,W = x.shape
        assert B==1
        x = x.squeeze(0)
        # x must be shape [C,H,W] now
        reg_loss = 0.0
        for ch_idx in range(x.shape[0]):
            noise = x[ch_idx][None, None,:,:]
            while True:
                if random_shift: roll_amount = randrange(noise.shape[2]//2)
                else: roll_amount = 1
                reg_loss += (noise*torch.roll(noise, shifts=roll_amount, dims=2)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=roll_amount, dims=3)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        return reg_loss
    
    def kl_divergence(self, x):
        _mu = x.mean()
        _var = x.var()
        return _var + _mu**2 - 1 - torch.log(_var+1e-7)


    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_inversion_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        img=None, # the input image as a PIL image
        torch_dtype=torch.float32,

        # inversion regularization parameters
        lambda_ac: float = 20.0,
        lambda_kl: float = 20.0,
        num_reg_steps: int = 5,
        num_ac_rolls: int = 5,
    ):
        
        # 0. modify the unet to be useful :D
        self.unet = prep_unet(self.unet)

        # set the scheduler to be the Inverse DDIM scheduler
        # self.scheduler = MyDDIMScheduler.from_config(self.scheduler.config)

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        self.scheduler.set_timesteps(num_inversion_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Encode the input image with the first stage model
        x0 = np.array(img)/255
        x0 = torch.from_numpy(x0).type(torch_dtype).permute(2, 0, 1).unsqueeze(dim=0).repeat(1, 1, 1, 1).to(device)
        x0 = (x0 - 0.5) * 2.
        with torch.no_grad():
            x0_enc = self.vae.encode(x0).latent_dist.sample().to(device, torch_dtype)
        latents = x0_enc = 0.18215 * x0_enc

        # Decode and return the image
        with torch.no_grad():
            x0_dec = self.decode_latents(x0_enc.detach())
        image_x0_dec = self.numpy_to_pil(x0_dec)

        with torch.no_grad():
            prompt_embeds = self._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt).to(device)
        extra_step_kwargs = self.prepare_extra_step_kwargs(None, eta)
        
        latent_list=[latents]

        # Do the inversion
        num_warmup_steps = len(timesteps) - num_inversion_steps * self.scheduler.order # should be 0?
        with self.progress_bar(total=num_inversion_steps) as progress_bar:
            # noted: we edit the code here to solve the problem of zero-shot can only do 48 steps of inversion
            for i, t in enumerate(timesteps.flip(0)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input,t,encoder_hidden_states=prompt_embeds,cross_attention_kwargs=cross_attention_kwargs,).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # regularization of the noise prediction
                e_t = noise_pred
                for _outer in range(num_reg_steps):
                    if lambda_ac>0:
                        for _inner in range(num_ac_rolls):
                            _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                            l_ac = self.auto_corr_loss(_var)
                            l_ac.backward()
                            _grad = _var.grad.detach()/num_ac_rolls
                            e_t = e_t - lambda_ac*_grad
                    if lambda_kl>0:
                        _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                        l_kld = self.kl_divergence(_var)
                        l_kld.backward()
                        _grad = _var.grad.detach()
                        e_t = e_t - lambda_kl*_grad
                    e_t = e_t.detach()
                noise_pred = e_t

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, reverse=True, **extra_step_kwargs).prev_sample
                latent_list.append(latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        


        # 8. Post-processing
        image = self.decode_latents(latents.detach())
        image = self.numpy_to_pil(image)
        return latent_list, image, image_x0_dec

class NullInversion:
    
    def prev_step(self, model_output, timestep: int, sample):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output, timestep: int, sample):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, guidance_scale, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], 
            padding="max_length", 
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()

        for i in tqdm(range(self.num_ddim_steps), desc="DDIM Inversion ..."):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon, guidance_scale):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        for i in tqdm(range(self.num_ddim_steps), desc="Null Optimization ..."):
            uncond_embeddings = uncond_embeddings.clone().detach()
            t = self.model.scheduler.timesteps[i]
            if num_inner_steps!=0:
                uncond_embeddings.requires_grad = True
                optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
                latent_prev = latents[len(latents) - i - 2]
                with torch.no_grad():
                    noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
                for j in range(num_inner_steps):
                    noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                    loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_item = loss.item()
                    if loss_item < epsilon + i * 2e-5:
                        break
                
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, guidance_scale, False, context)
        return uncond_embeddings_list
    
    def invert(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5, ours=False):
        self.init_prompt(prompt)
        if ours is False:
            register_attention_control(self.model, None)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale)
        return image_gt, image_rec, ddim_latents, uncond_embeddings
    
    def __init__(self, model,num_ddim_steps):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.prompt = None
        self.context = None
        self.num_ddim_steps=num_ddim_steps

class NegativePromptInversion:
    
    def prev_step(self, model_output, timestep, sample):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output, timestep, sample):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    @torch.no_grad()
    def init_prompt(self, prompt):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        print("DDIM Inversion ...")
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents, latent

    def invert(self, image_gt, prompt, npi_interp=0.0):
        """
        Get DDIM Inversion of the image
        
        Parameters:
        image_gt - the gt image with a size of [512,512,3], the channel follows the rgb of PIL.Image. i.e. RGB.
        prompt - this is the prompt used for DDIM Inversion
        npi_interp - the interpolation ratio among conditional embedding and unconditional embedding
        num_ddim_steps - the number of ddim steps
        
        Returns:
            image_rec - the image reconstructed by VAE decoder with a size of [512,512,3], the channel follows the rgb of PIL.Image. i.e. RGB.
            image_rec_latent - the image latent with a size of [64,64,4]
            ddim_latents - the ddim inversion latents 50*[64,4,4], the first latent is the image_rec_latent, the last latent is noise (but in fact not pure noise)
            uncond_embeddings - the fake uncond_embeddings, in fact is cond_embedding or a interpolation among cond_embedding and uncond_embedding
        """
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        image_rec, ddim_latents, image_rec_latent = self.ddim_inversion(image_gt)
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        if npi_interp > 0.0: # do vector interpolation among cond_embedding and uncond_embedding
            cond_embeddings = slerp_tensor(npi_interp, cond_embeddings, uncond_embeddings)
        uncond_embeddings = [cond_embeddings] * self.num_ddim_steps
        return image_rec, image_rec_latent, ddim_latents, uncond_embeddings

    def __init__(self, model,num_ddim_steps):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.prompt = None
        self.context = None
        self.num_ddim_steps=num_ddim_steps