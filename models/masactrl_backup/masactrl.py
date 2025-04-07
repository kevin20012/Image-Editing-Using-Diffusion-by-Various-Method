import os

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from einops import rearrange

from diffusers import DDIMScheduler

from models.p2p.inversion import DirectInversion, NullInversion, NegativePromptInversion
from .proximal_guidance_forward import proximal_guidance_forward
from models.p2p.scheduler_dev import DDIMSchedulerDev

from .masactrl_utils import AttentionBase
from .diffuser_utils import MasaCtrlPipeline
from .masactrl_utils import AttentionBase
from .masactrl_utils import register_attention_editor_diffusers
# from .masactrl import MutualSelfAttentionControl

from torchvision.utils import save_image
from torchvision.io import read_image
from utils.utils import load_512,txt_draw, latent2image

def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image


class MutualSelfAttentionControl(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        print("MasaCtrl at denoising steps: ", self.step_idx)
        print("MasaCtrl at U-Net layers: ", self.layer_idx)

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u = self.attn_batch(qu, ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c = self.attn_batch(qc, kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
        out = torch.cat([out_u, out_c], dim=0)

        return out


class MutualSelfAttentionControlUnion(MutualSelfAttentionControl):
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"):
        """
        Mutual self-attention control for Stable-Diffusion model with unition source and target [K, V]
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps, model_type)

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        qu_s, qu_t, qc_s, qc_t = q.chunk(4)
        ku_s, ku_t, kc_s, kc_t = k.chunk(4)
        vu_s, vu_t, vc_s, vc_t = v.chunk(4)
        attnu_s, attnu_t, attnc_s, attnc_t = attn.chunk(4)

        # source image branch
        out_u_s = super().forward(qu_s, ku_s, vu_s, sim, attnu_s, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_s = super().forward(qc_s, kc_s, vc_s, sim, attnc_s, is_cross, place_in_unet, num_heads, **kwargs)

        # target image branch, concatenating source and target [K, V]
        out_u_t = self.attn_batch(qu_t, torch.cat([ku_s, ku_t]), torch.cat([vu_s, vu_t]), sim[:num_heads], attnu_t, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_t = self.attn_batch(qc_t, torch.cat([kc_s, kc_t]), torch.cat([vc_s, vc_t]), sim[:num_heads], attnc_t, is_cross, place_in_unet, num_heads, **kwargs)

        out = torch.cat([out_u_s, out_u_t, out_c_s, out_c_t], dim=0)

        return out


class MutualSelfAttentionControlMask(MutualSelfAttentionControl):
    def __init__(self,  start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50,  mask_s=None, mask_t=None, mask_save_dir=None, model_type="SD"):
        """
        Maske-guided MasaCtrl to alleviate the problem of fore- and background confusion
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            mask_s: source mask with shape (h, w)
            mask_t: target mask with same shape as source mask
            mask_save_dir: the path to save the mask image
            model_type: the model type, SD or SDXL
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps, model_type)
        self.mask_s = mask_s  # source mask with shape (h, w)
        self.mask_t = mask_t  # target mask with same shape as source mask
        print("Using mask-guided MasaCtrl")
        if mask_save_dir is not None:
            os.makedirs(mask_save_dir, exist_ok=True)
            save_image(self.mask_s.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask_s.png"))
            save_image(self.mask_t.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask_t.png"))

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if kwargs.get("is_mask_attn") and self.mask_s is not None:
            print("masked attention")
            mask = self.mask_s.unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask, (H, W)).flatten(0).unsqueeze(0)
            mask = mask.flatten()
            # background
            sim_bg = sim + mask.masked_fill(mask == 1, torch.finfo(sim.dtype).min)
            # object
            sim_fg = sim + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min)
            sim = torch.cat([sim_fg, sim_bg], dim=0)
        attn = sim.softmax(-1)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u_source = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_source = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        out_u_target = self.attn_batch(qu[-num_heads:], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, is_mask_attn=True, **kwargs)
        out_c_target = self.attn_batch(qc[-num_heads:], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, is_mask_attn=True, **kwargs)

        if self.mask_s is not None and self.mask_t is not None:
            out_u_target_fg, out_u_target_bg = out_u_target.chunk(2, 0)
            out_c_target_fg, out_c_target_bg = out_c_target.chunk(2, 0)

            mask = F.interpolate(self.mask_t.unsqueeze(0).unsqueeze(0), (H, W))
            mask = mask.reshape(-1, 1)  # (hw, 1)
            out_u_target = out_u_target_fg * mask + out_u_target_bg * (1 - mask)
            out_c_target = out_c_target_fg * mask + out_c_target_bg * (1 - mask)

        out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)
        return out


class MutualSelfAttentionControlMaskAuto(MutualSelfAttentionControl):
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, thres=0.1, ref_token_idx=[1], cur_token_idx=[1], mask_save_dir=None, model_type="SD"):
        """
        MasaCtrl with mask auto generation from cross-attention map
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            thres: the thereshold for mask thresholding
            ref_token_idx: the token index list for cross-attention map aggregation
            cur_token_idx: the token index list for cross-attention map aggregation
            mask_save_dir: the path to save the mask image
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps, model_type)
        print("Using MutualSelfAttentionControlMaskAuto")
        self.thres = thres
        self.ref_token_idx = ref_token_idx
        self.cur_token_idx = cur_token_idx

        self.self_attns = []
        self.cross_attns = []

        self.cross_attns_mask = None
        self.self_attns_mask = None

        self.mask_save_dir = mask_save_dir
        if self.mask_save_dir is not None:
            os.makedirs(self.mask_save_dir, exist_ok=True)

    def after_step(self):
        self.self_attns = []
        self.cross_attns = []

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if self.self_attns_mask is not None:
            # binarize the mask
            mask = self.self_attns_mask
            thres = self.thres
            mask[mask >= thres] = 1
            mask[mask < thres] = 0
            sim_fg = sim + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min)
            sim_bg = sim + mask.masked_fill(mask == 1, torch.finfo(sim.dtype).min)
            sim = torch.cat([sim_fg, sim_bg])

        attn = sim.softmax(-1)

        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def aggregate_cross_attn_map(self, idx):
        attn_map = torch.stack(self.cross_attns, dim=1).mean(1)  # (B, N, dim)
        B = attn_map.shape[0]
        res = int(np.sqrt(attn_map.shape[-2]))
        attn_map = attn_map.reshape(-1, res, res, attn_map.shape[-1])
        image = attn_map[..., idx]
        if isinstance(idx, list):
            image = image.sum(-1)
        image_min = image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        image_max = image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        image = (image - image_min) / (image_max - image_min)
        return image

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross:
            # save cross attention map with res 16 * 16
            if attn.shape[1] == 16 * 16:
                self.cross_attns.append(attn.reshape(-1, num_heads, *attn.shape[-2:]).mean(1))

        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u_source = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_source = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        if len(self.cross_attns) == 0:
            self.self_attns_mask = None
            out_u_target = self.attn_batch(qu[-num_heads:], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_target = self.attn_batch(qc[-num_heads:], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
        else:
            mask = self.aggregate_cross_attn_map(idx=self.ref_token_idx)  # (2, H, W)
            mask_source = mask[-2]  # (H, W)
            res = int(np.sqrt(q.shape[1]))
            self.self_attns_mask = F.interpolate(mask_source.unsqueeze(0).unsqueeze(0), (res, res)).flatten()
            if self.mask_save_dir is not None:
                H = W = int(np.sqrt(self.self_attns_mask.shape[0]))
                mask_image = self.self_attns_mask.reshape(H, W).unsqueeze(0)
                save_image(mask_image, os.path.join(self.mask_save_dir, f"mask_s_{self.cur_step}_{self.cur_att_layer}.png"))
            out_u_target = self.attn_batch(qu[-num_heads:], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_target = self.attn_batch(qc[-num_heads:], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        if self.self_attns_mask is not None:
            mask = self.aggregate_cross_attn_map(idx=self.cur_token_idx)  # (2, H, W)
            mask_target = mask[-1]  # (H, W)
            res = int(np.sqrt(q.shape[1]))
            spatial_mask = F.interpolate(mask_target.unsqueeze(0).unsqueeze(0), (res, res)).reshape(-1, 1)
            if self.mask_save_dir is not None:
                H = W = int(np.sqrt(spatial_mask.shape[0]))
                mask_image = spatial_mask.reshape(H, W).unsqueeze(0)
                save_image(mask_image, os.path.join(self.mask_save_dir, f"mask_t_{self.cur_step}_{self.cur_att_layer}.png"))
            # binarize the mask
            thres = self.thres
            spatial_mask[spatial_mask >= thres] = 1
            spatial_mask[spatial_mask < thres] = 0
            out_u_target_fg, out_u_target_bg = out_u_target.chunk(2)
            out_c_target_fg, out_c_target_bg = out_c_target.chunk(2)

            out_u_target = out_u_target_fg * spatial_mask + out_u_target_bg * (1 - spatial_mask)
            out_c_target = out_c_target_fg * spatial_mask + out_c_target_bg * (1 - spatial_mask)

            # set self self-attention mask to None
            self.self_attns_mask = None

        out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)
        return out

class MasaCtrlEditor:
    def __init__(self, method_list, device, num_ddim_steps=50) -> None:
        self.device=device
        self.method_list=method_list
        self.num_ddim_steps=num_ddim_steps
        # init model
        self.scheduler1 = DDIMScheduler(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="scaled_linear",
                                    clip_sample=False,
                                    set_alpha_to_one=False)
        self.scheduler2 = DDIMSchedulerDev(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="scaled_linear",
                                    clip_sample=False,
                                    set_alpha_to_one=False)
        self.model = MasaCtrlPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", scheduler=self.scheduler1).to(device)
        self.neg_model = MasaCtrlPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", scheduler=self.scheduler2).to(device)
        self.model.scheduler.set_timesteps(self.num_ddim_steps)
        self.neg_model.scheduler.set_timesteps(self.num_ddim_steps)
    
    def __call__(self, 
                edit_method,
                image_path,
                prompt_src,
                prompt_tar,
                guidance_scale,
                step=4,
                layper=10):
        if edit_method=="ddim+masactrl":
            return self.edit_image_ddim_MasaCtrl(image_path,prompt_src,prompt_tar,guidance_scale,step=step,layper=layper)
        elif edit_method=="directinversion+masactrl":
            return self.edit_image_directinversion_MasaCtrl(image_path,prompt_src,prompt_tar,guidance_scale,step=step,layper=layper)
        elif edit_method=="null-text-inversion+masactrl":
            return self.edit_image_null_text_inversion_MasaCtrl(image_path,prompt_src,prompt_tar,guidance_scale,step=step,layper=layper)
        elif edit_method=="negative-prompt-inversion+masactrl":
            return self.edit_image_negative_prompt_inversion_MasaCtrl(image_path,prompt_src,prompt_tar,guidance_scale,step=step,layper=layper)
        elif edit_method=="inversion-free-editing+masactrl":
            return self.edit_image_inversion_free_editing_MasaCtrl(image_path,prompt_src,prompt_tar,guidance_scale,step=step,layper=layper)
        else:
            raise NotImplementedError(f"No edit method named {edit_method}")
    #To-Do: Implement the following methods
    def edit_image_null_text_inversion_MasaCtrl(self,image_path,prompt_src,prompt_tar,guidance_scale,step=4,layper=10):
        source_image=load_image(image_path, self.device)
        image_gt = load_512(image_path)
        
        prompts = [prompt_src, prompt_tar]
        
        null_inversion = NullInversion(model=self.model,
                                    num_ddim_steps=self.num_ddim_steps)
        
        _, _, x_stars, uncond_embeddings = null_inversion.invert(
            image_gt=image_gt, prompt=prompt_src,guidance_scale=guidance_scale)
        x_t = x_stars[-1]
        
        # results of direct synthesis
        editor = AttentionBase()
        register_attention_editor_diffusers(self.model, editor)
        image_fixed = self.model([prompt_tar],
                            latents=x_t,
                            num_inference_steps=self.num_ddim_steps,
                            guidance_scale=guidance_scale,
                            unconditioning=uncond_embeddings)
        
        # hijack the attention module
        editor = MutualSelfAttentionControl(step, layper)
        register_attention_editor_diffusers(self.model, editor)

        # inference the synthesized image
        image_masactrl = self.model(prompts,
                            latents= x_t.expand(len(prompts), -1, -1, -1),
                            guidance_scale=guidance_scale,
                            unconditioning=uncond_embeddings)
        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        out_image=np.concatenate((
                                np.array(image_instruct),
                                (image_gt),
                                (image_masactrl[0].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8),
                                (image_masactrl[-1].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)),1)
        
        return Image.fromarray(out_image)
    #To-Do
    def edit_image_negative_prompt_inversion_MasaCtrl(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5, 
        proximal=None,
        quantile=0.7,
        use_reconstruction_guidance=False,
        recon_t=400,
        recon_lr=0.1,
        npi_interp=0, 
        use_inversion_guidance=False,
        dilate_mask=1,
        step=4,
        layper=10
    ):
        
        source_image=load_image(image_path, self.device)
        image_gt = load_512(image_path)
        
        prompts = [prompt_src, prompt_tar]
        
        null_inversion = NegativePromptInversion(model=self.neg_model,
                                                num_ddim_steps=self.num_ddim_steps)
        
        _, image_enc_latent, x_stars, uncond_embeddings = null_inversion.invert(
            image_gt=image_gt, prompt=prompt_src, npi_interp=npi_interp)
        x_t = x_stars[-1]
        
        # results of direct synthesis
        editor = AttentionBase()
        # register_attention_editor_diffusers(self.model, editor)
        # image_fixed = self.model([prompt_tar],
        #                     latents=x_t,
        #                     num_inference_steps=self.num_ddim_steps,
        #                     guidance_scale=guidance_scale,
        #                     noise_loss_list=None)
        
        reconstruct_latent, _ = proximal_guidance_forward(
                model=self.neg_model,
                prompt=[prompt_src],
                controller=editor,
                latent=x_t,
                guidance_scale=guidance_scale,
                generator=None,
                uncond_embeddings=uncond_embeddings,
                edit_stage=False,
                prox=None,
                quantile=quantile,
                image_enc=None,
                recon_lr=recon_lr,
                recon_t=recon_t,
                inversion_guidance=False,
                x_stars=None,
                dilate_mask=dilate_mask)
        reconstruct_image = latent2image(model=self.model.vae, latents=reconstruct_latent)[0]
        # hijack the attention module
        editor = MutualSelfAttentionControl(step, layper)
        # register_attention_editor_diffusers(self.model, editor)

        # inference the synthesized image
        # image_masactrl = self.model(prompts,
        #                     latents= x_t.expand(len(prompts), -1, -1, -1),
        #                     guidance_scale=guidance_scale,
        #                     noise_loss_list=noise_loss_list)
        latents_masactrl, _ = proximal_guidance_forward(
                        model=self.neg_model,
                        prompt=prompts,
                        controller=editor,
                        latent=x_t.expand(len(prompts), -1, -1, -1),
                        guidance_scale=guidance_scale,
                        generator=None,
                        uncond_embeddings=uncond_embeddings,
                        edit_stage=True,
                        prox=proximal,
                        quantile=quantile,
                        image_enc=image_enc_latent if use_reconstruction_guidance else None,
                        recon_lr=recon_lr
                            if use_reconstruction_guidance or use_inversion_guidance else 0,
                        recon_t=recon_t
                            if use_reconstruction_guidance or use_inversion_guidance else 1000,
                        x_stars=x_stars,
                        dilate_mask=dilate_mask)
        image_masactrl = latent2image(model=self.model.vae, latents=latents_masactrl)
        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        out_image=Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,image_masactrl[-1]),axis=1))
        
        return out_image


    def edit_image_directinversion_MasaCtrl(self,image_path,prompt_src,prompt_tar,guidance_scale,step=4,layper=10):
        source_image=load_image(image_path, self.device)
        image_gt = load_512(image_path)
        
        prompts=["", prompt_tar]
        
        null_inversion = DirectInversion(model=self.model,
                                                num_ddim_steps=self.num_ddim_steps)
        
        _, image_enc_latent, x_stars, noise_loss_list = null_inversion.invert(
            image_gt=image_gt, prompt=prompts, guidance_scale=guidance_scale)
        x_t = x_stars[-1]
        
        # results of direct synthesis
        editor = AttentionBase()
        register_attention_editor_diffusers(self.model, editor)
        image_fixed = self.model([prompt_tar],
                            latents=x_t,
                            num_inference_steps=self.num_ddim_steps,
                            guidance_scale=guidance_scale,
                            noise_loss_list=None)
        
        # hijack the attention module
        editor = MutualSelfAttentionControl(step, layper)
        register_attention_editor_diffusers(self.model, editor)

        # inference the synthesized image
        image_masactrl = self.model(prompts,
                            latents= x_t.expand(len(prompts), -1, -1, -1),
                            guidance_scale=guidance_scale,
                            noise_loss_list=noise_loss_list)
        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        out_image=np.concatenate((
                                np.array(image_instruct),
                                (image_gt),
                                (image_masactrl[0].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8),
                                (image_masactrl[-1].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)),1)
        
        return Image.fromarray(out_image)
    
    def edit_image_ddim_MasaCtrl(self, image_path,prompt_src,prompt_tar,guidance_scale,step=4,layper=10):
        source_image=load_image(image_path, self.device)
        image_gt = load_512(image_path)
        
        prompts=["", prompt_tar]
        
        start_code, latents_list = self.model.invert(source_image,
                                            "",
                                            guidance_scale=guidance_scale,
                                            num_inference_steps=self.num_ddim_steps,
                                            return_intermediates=True)
        start_code = start_code.expand(len(prompts), -1, -1, -1)
        
        # results of direct synthesis
        editor = AttentionBase()
        register_attention_editor_diffusers(self.model, editor)
        image_fixed = self.model([prompt_tar],
                            latents=start_code[-1:],
                            num_inference_steps=self.num_ddim_steps,
                            guidance_scale=guidance_scale)
        
        # hijack the attention module
        editor = MutualSelfAttentionControl(step, layper)
        register_attention_editor_diffusers(self.model, editor)

        # inference the synthesized image
        image_masactrl = self.model(prompts,
                            latents=start_code,
                            guidance_scale=guidance_scale)
        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        out_image=np.concatenate((
                                np.array(image_instruct),
                                (image_gt),
                                (image_masactrl[0].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8),
                                (image_masactrl[-1].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)),1)
        
        return Image.fromarray(out_image)