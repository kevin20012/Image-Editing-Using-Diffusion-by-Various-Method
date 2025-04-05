import torch

import numpy as np
import random
from PIL import Image
from .register import register_attention_control

from lavis.models import load_model_and_preprocess
# from models.p2p.inversion import DirectInversion, NullInversion, NegativePromptInversion

from models.pix2pix_zero.inversion import DDIMInversion, NullInversion, NegativePromptInversion
from models.pix2pix_zero.base_pipeline import BasePipeline
from models.pix2pix_zero.scheduler import DDIMInverseScheduler
from models.pix2pix_zero.edit_directions import construct_direction
from models.pix2pix_zero.edit_pipeline import EditingPipeline
from diffusers import StableDiffusionPipeline
from models.p2p.scheduler_dev import DDIMSchedulerDev
from diffusers import StableDiffusionPipeline, DDIMScheduler
from utils.utils import txt_draw, load_512

from diffusers import DDIMScheduler

class Pix2PixZeroEditor:
    def __init__(self, method_list, device, num_ddim_steps=50, xa_guidance=0.1) -> None:
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
            'cpu')
        
        self.method_list = method_list
        
        self.num_ddim_steps = num_ddim_steps
        self.xa_guidance = xa_guidance

        # load the BLIP model
        model_blip, vis_processors, _ = load_model_and_preprocess(name="blip_caption", 
                                                                model_type="base_coco", 
                                                                is_eval=True, 
                                                                device=torch.device(device))
        
        self.model_blip = model_blip
        self.vis_processors = vis_processors

        # make the DDIM inversion pipeline
        self.pipe = DDIMInversion.from_pretrained('CompVis/stable-diffusion-v1-4').to(device)
        self.pipe.scheduler = DDIMInverseScheduler.from_config(self.pipe.scheduler.config)

        self.edit_pipe = EditingPipeline.from_pretrained('CompVis/stable-diffusion-v1-4').to(device)
        self.edit_pipe.scheduler = DDIMScheduler.from_config(self.edit_pipe.scheduler.config)

        # for null text inversion
        self.ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
        self.ldm_stable.scheduler = DDIMScheduler.from_config(self.ldm_stable.scheduler.config)
        self.ldm_stable.scheduler.set_timesteps(num_ddim_steps)
        
    def __call__(self, edit_method, image_path, prompt_src, prompt_tar, guidance_scale=7.5, image_size=[512,512]):
        if edit_method=="ddim+pix2pix_zero":
            return self.edit_image_ddim_pix2pix_zero(image_path, prompt_src, prompt_tar, guidance_scale, image_size)
        elif edit_method=="directinversion+pix2pix_zero":
            return self.edit_image_directinversion_pix2pix_zero(image_path, prompt_src, prompt_tar, guidance_scale, image_size)
        elif edit_method=="null-text-inversion+pix2pix_zero":
            return self.edit_image_null_text_inversion_pix2pix_zero(image_path, prompt_src, prompt_tar, guidance_scale, image_size)
        elif edit_method=="ours+pix2pix_zero":
            return self.edit_image_null_text_inversion_pix2pix_zero(image_path, prompt_src, prompt_tar, guidance_scale, image_size, ours=True)
        elif edit_method=="negative-prompt-inversion+pix2pix_zero":
            return self.edit_image_negative_prompt_pix2pix_zero(image_path, prompt_src, prompt_tar, guidance_scale, image_size)
        else:
            raise ValueError(f"edit method {edit_method} not supported")

    ## convert sentences to sentence embeddings
    def load_sentence_embeddings(self, l_sentences, tokenizer, text_encoder):
        with torch.no_grad():
            l_embeddings = []
            for sent in l_sentences:
                text_inputs = tokenizer(
                        sent,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                text_input_ids = text_inputs.input_ids
                prompt_embeds = text_encoder(text_input_ids.to(self.device), attention_mask=None)[0]
                l_embeddings.append(prompt_embeds)
        return torch.concat(l_embeddings, dim=0).mean(dim=0).unsqueeze(0)


    def edit_image_ddim_pix2pix_zero(self, image_path,
                    prompt_src,
                    prompt_tar,
                    guidance_scale=7.5,
                    image_size=[512,512]):
        image_gt = Image.open(image_path).resize(image_size, Image.Resampling.LANCZOS)
        if image_gt.mode == 'RGBA':
            image_gt = image_gt.convert('RGB')
        # generate the caption
        prompt_str = self.model_blip.generate({"image": self.vis_processors["eval"](image_gt).unsqueeze(0).to(self.device)})[0]
        latent_list, x_inv_image, x_dec_img = self.pipe(
                prompt_str, 
                guidance_scale=1,
                num_inversion_steps=self.num_ddim_steps,
                img=image_gt
            )
        
        inversion_latent=latent_list[-1].detach()
        
        mean_emb_src = self.load_sentence_embeddings([prompt_str], self.edit_pipe.tokenizer, self.edit_pipe.text_encoder)
        mean_emb_tar = self.load_sentence_embeddings([prompt_tar], self.edit_pipe.tokenizer, self.edit_pipe.text_encoder)
        
        rec_pil, edit_pil = self.edit_pipe(prompt_str,
                    num_inference_steps=self.num_ddim_steps,
                    x_in=inversion_latent,
                    edit_dir=(mean_emb_tar.mean(0)-mean_emb_src.mean(0)).unsqueeze(0),
                    guidance_amount=self.xa_guidance,
                    guidance_scale=guidance_scale,
                    negative_prompt=prompt_str # use the unedited prompt for the negative prompt
            )
        
        image_instruct = txt_draw(f"source prompt: {prompt_str}\ntarget prompt: {prompt_tar}")
        
        out_image=np.concatenate((np.array(image_instruct),np.array(image_gt),np.array(rec_pil[0]),np.array(edit_pil[0])),1)
        
        return Image.fromarray(out_image)
        

    def edit_image_directinversion_pix2pix_zero(self, image_path,
                    prompt_src,
                    prompt_tar,
                    guidance_scale=7.5,
                    image_size=[512,512]):
        image_gt = Image.open(image_path).resize(image_size, Image.Resampling.LANCZOS)
        if image_gt.mode == 'RGBA':
            image_gt = image_gt.convert('RGB')

        # generate the caption
        prompt_str = self.model_blip.generate({"image": self.vis_processors["eval"](image_gt).unsqueeze(0).to(self.device)})[0]
        latent_list, x_inv_image, x_dec_img = self.pipe(
                prompt_str, 
                guidance_scale=1,
                num_inversion_steps=self.num_ddim_steps,
                img=image_gt
            )
        
        inversion_latent=latent_list[-1].detach()
        
        mean_emb_src = self.load_sentence_embeddings([prompt_str], self.edit_pipe.tokenizer, self.edit_pipe.text_encoder)
        mean_emb_tar = self.load_sentence_embeddings([prompt_tar], self.edit_pipe.tokenizer, self.edit_pipe.text_encoder)
        
        rec_pil, edit_pil = self.edit_pipe(prompt_str,
                    num_inference_steps=self.num_ddim_steps,
                    x_in=inversion_latent,
                    edit_dir=(mean_emb_tar.mean(0)-mean_emb_src.mean(0)).unsqueeze(0),
                    guidance_amount=self.xa_guidance,
                    guidance_scale=guidance_scale,
                    negative_prompt=prompt_str, # use the unedited prompt for the negative prompt
                    latent_list=latent_list
            )
        
        image_instruct = txt_draw(f"source prompt: {prompt_str}\ntarget prompt: {prompt_tar}")
        
        out_image=np.concatenate((np.array(image_instruct),np.array(image_gt),np.array(rec_pil[0]),np.array(edit_pil[0])),1)
        
        return Image.fromarray(out_image)
    
    #To-Do: Implement the following methods
    def edit_image_null_text_inversion_pix2pix_zero(self, image_path,
                    prompt_src,
                    prompt_tar,
                    guidance_scale=7.5,
                    image_size=[512,512], 
                    ours=False):
        image_gt = Image.open(image_path).resize(image_size, Image.Resampling.LANCZOS)
        if image_gt.mode == 'RGBA':
            image_gt = image_gt.convert('RGB')
        image_gt_ = np.array(image_gt)
        if ours:
            register_attention_control(self.ldm_stable, None, ours)
        # generate the caption
        prompt_str = self.model_blip.generate({"image": self.vis_processors["eval"](image_gt).unsqueeze(0).to(self.device)})[0]
        null_inversion = NullInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        
        _, _, x_stars, uncond_embeddings = null_inversion.invert(
            image_gt=image_gt_, prompt=prompt_str,guidance_scale=guidance_scale, ours=ours)
        x_t = x_stars[-1]

        mean_emb_src = self.load_sentence_embeddings([prompt_str], self.edit_pipe.tokenizer, self.edit_pipe.text_encoder)
        mean_emb_tar = self.load_sentence_embeddings([prompt_tar], self.edit_pipe.tokenizer, self.edit_pipe.text_encoder)
        
        rec_pil, edit_pil = self.edit_pipe(prompt_str,
                    num_inference_steps=self.num_ddim_steps,
                    x_in=x_t,
                    edit_dir=(mean_emb_tar.mean(0)-mean_emb_src.mean(0)).unsqueeze(0),
                    guidance_amount=self.xa_guidance,
                    guidance_scale=guidance_scale,
                    negative_prompt_embeds=uncond_embeddings,
            )
        
        image_instruct = txt_draw(f"source prompt: {prompt_str}\ntarget prompt: {prompt_tar}")
        
        out_image=np.concatenate((np.array(image_instruct),np.array(image_gt),np.array(rec_pil[0]),np.array(edit_pil[0])),1)
        
        return Image.fromarray(out_image)

    def edit_image_negative_prompt_pix2pix_zero(self, image_path,
                    prompt_src,
                    prompt_tar,
                    guidance_scale=7.5,
                    image_size=[512,512]):
        image_gt = Image.open(image_path).resize(image_size, Image.Resampling.LANCZOS)
        if image_gt.mode == 'RGBA':
            image_gt = image_gt.convert('RGB')
        image_gt_ = np.array(image_gt)
        # generate the caption
        prompt_str = self.model_blip.generate({"image": self.vis_processors["eval"](image_gt).unsqueeze(0).to(self.device)})[0]
        negative_inversion = NegativePromptInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        
        _, image_enc_latent, x_stars, uncond_embeddings = negative_inversion.invert(
            image_gt=image_gt_, prompt=prompt_str, npi_interp=0)
        x_t = x_stars[-1]

        mean_emb_src = self.load_sentence_embeddings([prompt_str], self.edit_pipe.tokenizer, self.edit_pipe.text_encoder)
        mean_emb_tar = self.load_sentence_embeddings([prompt_tar], self.edit_pipe.tokenizer, self.edit_pipe.text_encoder)
        
        rec_pil, edit_pil = self.edit_pipe(prompt_str,
                    num_inference_steps=self.num_ddim_steps,
                    x_in=x_t,
                    edit_dir=(mean_emb_tar.mean(0)-mean_emb_src.mean(0)).unsqueeze(0),
                    guidance_amount=self.xa_guidance,
                    guidance_scale=guidance_scale,
                    negative_prompt=prompt_str, # use the unedited prompt for the negative prompt
            )
        
        image_instruct = txt_draw(f"source prompt: {prompt_str}\ntarget prompt: {prompt_tar}")
        
        out_image=np.concatenate((np.array(image_instruct),np.array(image_gt),np.array(rec_pil[0]),np.array(edit_pil[0])),1)
        
        return Image.fromarray(out_image)
    def edit_image_inversion_free_editing_pix2pix_zero(self, image_path,
                    prompt_src,
                    prompt_tar,
                    guidance_scale=7.5,
                    image_size=[512,512]):
        pass