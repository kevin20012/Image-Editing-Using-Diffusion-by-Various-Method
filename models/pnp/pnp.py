import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline
import numpy as np
from PIL import Image
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from models.p2p.inversion import DirectInversion, NullInversion, NegativePromptInversion
import torchvision.transforms as T
from .register import register_attention_control

from utils.utils import txt_draw,load_512,latent2image

OURS = True


def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start

class Preprocess(nn.Module):
    def __init__(self, device,model_key,num_ddim_steps):
        super().__init__()

        self.device = device
        self.use_depth = False

        print(f'[INFO] loading stable diffusion...')
        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", 
                                                 ).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder",
                                                          ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet",
                                                         ).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(num_ddim_steps)
        print(f'[INFO] loaded stable diffusion!')


    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def load_img(self, image_path):
        image_pil = T.Resize([512,512])(Image.open(image_path).convert("RGB"))
        image = T.ToTensor()(image_pil).unsqueeze(0).to(self.device)
        return image

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def ddim_inversion(self, cond, latent):
        latent_list=[latent]
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(timesteps):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                latent_list.append(latent)
        return latent_list

    @torch.no_grad()
    def ddim_sample(self, x, cond):
        timesteps = self.scheduler.timesteps
        latent_list=[]
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(timesteps):
                    cond_batch = cond.repeat(x.shape[0], 1, 1)
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = (
                        self.scheduler.alphas_cumprod[timesteps[i + 1]]
                        if i < len(timesteps) - 1
                        else self.scheduler.final_alpha_cumprod
                    )
                    mu = alpha_prod_t ** 0.5
                    sigma = (1 - alpha_prod_t) ** 0.5
                    mu_prev = alpha_prod_t_prev ** 0.5
                    sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                    eps = self.unet(x, t, encoder_hidden_states=cond_batch).sample

                    pred_x0 = (x - sigma * eps) / mu
                    x = mu_prev * pred_x0 + sigma_prev * eps
                    latent_list.append(x)
        return latent_list
    
    def prev_step(self, model_output, timestep: int, sample):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        difference_scale_pred_original_sample= - beta_prod_t ** 0.5  / alpha_prod_t ** 0.5
        difference_scale_pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 
        difference_scale = alpha_prod_t_prev ** 0.5 * difference_scale_pred_original_sample + difference_scale_pred_sample_direction
        
        return prev_sample,difference_scale
    
    def get_noise_pred_single(self, latents, t, context):
        # print("latent shape",latents.shape)
        # print("context shape",context.shape)
        noise_pred = self.unet(latents, t, encoder_hidden_states=context).sample
        return noise_pred
    
    

    @torch.no_grad()
    def extract_latents(self, num_steps, data_path,
                        inversion_prompt='', guidance_scale=7.5):
        self.scheduler.set_timesteps(num_steps)

        cond_ = self.get_text_embeds(inversion_prompt, "")
        cond = self.get_text_embeds(inversion_prompt, "")[1].unsqueeze(0)
        image = self.load_img(data_path)
        latent = self.encode_imgs(image)

        inverted_x = self.ddim_inversion(cond, latent) #X0, X1, ..., XT
        latent_reconstruction = self.ddim_sample(inverted_x[-1], cond) #XT', XT-1', ..., X0'
        rgb_reconstruction = self.decode_latents(latent_reconstruction[-1])
        latent_reconstruction.reverse() #X0', X1', ..., XT'
        return inverted_x, rgb_reconstruction, latent_reconstruction


def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)


def register_attention_control_efficient(model, injection_schedule, alpha=None):
    def sa_forward(self, place_in_unet, alpha=None):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads

            is_cross = encoder_hidden_states is not None
            residual = x
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            if not is_cross and self.injection_schedule is not None and (
                    self.t in self.injection_schedule or self.t == 1000):
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)

                source_batch_size = int(q.shape[0] // 3)
                # inject unconditional
                q[source_batch_size:2 * source_batch_size] = q[:source_batch_size]
                k[source_batch_size:2 * source_batch_size] = k[:source_batch_size]
                # inject conditional
                q[2 * source_batch_size:] = q[:source_batch_size]
                k[2 * source_batch_size:] = k[:source_batch_size]

                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
            else:
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)

            v = self.to_v(encoder_hidden_states)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            if alpha is not None:
                mid_scale, down_scale = alpha, alpha
            
                if self.to_k.in_features != self.to_q.in_features:
                    pass

                #------------------------------------cross attention------------------------------------------------------
                else:
                #-------------------------------------self attention ----------------------------------------------------        
                    #-----------------------------------------------------------------------------------------------
                    if place_in_unet == "up":
                        out = (1-down_scale)*out + residual*(down_scale)

            return to_out(out)

        return forward
    
    def ca_forward(self, place_in_unet, alpha=None):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            residual = x
            h = self.heads
            q = self.to_q(x)
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)
            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)
            if alpha is not None:
                mid_scale, down_scale = alpha, alpha
            
                if self.to_k.in_features != self.to_q.in_features:
                    pass

                #------------------------------------cross attention------------------------------------------------------
                else:
                #-------------------------------------self attention ----------------------------------------------------        
                    #-----------------------------------------------------------------------------------------------
                    if place_in_unet == "up":
                        out = (1-down_scale)*out + residual*(down_scale)
            return to_out(out)

        return forward
    
    if alpha is not None:
        def register_recr(net_, count, place_in_unet):
            # print(net_.__class__.__name__)
            if net_.__class__.__name__ == 'Attention':
                net_.forward = ca_forward(net_, place_in_unet, alpha=alpha)
                return count + 1
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    count = register_recr(net__, count, place_in_unet)
            return count

        cross_att_count = 0
        sub_nets = model.unet.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                cross_att_count += register_recr(net[1], 0, "down")
            elif "up" in net[0]:
                cross_att_count += register_recr(net[1], 0, "up")
            elif "mid" in net[0]:
                cross_att_count += register_recr(net[1], 0, "mid")
        # print(f"total cross attention layers: {cross_att_count}")

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module, "up", alpha=alpha)
            setattr(module, 'injection_schedule', injection_schedule)


def register_conv_control_efficient(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 3)
                # inject unconditional
                hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                # inject conditional
                hidden_states[2 * source_batch_size:] = hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)

class PNP(nn.Module):
    def __init__(self,num_ddim_steps=50,device="cuda"):
        super().__init__()
        self.device = device
        model_key = "runwayml/stable-diffusion-v1-5"

        # Create SD models
        print('Loading SD model')

        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(num_ddim_steps, device=self.device)
        self.num_ddim_steps=num_ddim_steps

        
        self.toy_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.toy_scheduler.set_timesteps(self.num_ddim_steps)
        timesteps_to_save, num_inference_steps = get_timesteps(self.toy_scheduler, num_inference_steps=self.num_ddim_steps,
                                                                strength=1.0,
                                                                device=self.device)
        self.timesteps_to_save = timesteps_to_save
        self.num_inference_steps = num_inference_steps
        self.model = Preprocess(self.device, model_key=model_key, num_ddim_steps=num_ddim_steps)
        
        print('SD model loaded')

    def __call__(self, edit_method, image_path, prompt_src, prompt_tar, guidance_scale=7.5, image_size=[512,512], alpha=None):
        if edit_method=="ddim+pnp":
            return self.edit_image_ddim_PnP(image_path, prompt_src, prompt_tar, guidance_scale, image_size, alpha=alpha)
        elif edit_method=="directinversion+pnp":
            return self.edit_image_directinversion_PnP(image_path, prompt_src, prompt_tar, guidance_scale, image_size, alpha=alpha)
        elif edit_method=="null-text-inversion+pnp":
            return self.edit_image_null_text_inversion_pnp(image_path, prompt_src, prompt_tar, guidance_scale, image_size, alpha=alpha)
        elif edit_method=="negative-prompt-inversion+pnp":
            return self.edit_image_negative_prompt_inversion_pnp(image_path, prompt_src, prompt_tar, guidance_scale, image_size, alpha=alpha)
        
        else:
            raise ValueError(f"edit method {edit_method} not supported")
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def get_data(self,image_path):
        # load image
        image = Image.open(image_path).convert('RGB') 
        image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
        image = T.ToTensor()(image).to(self.device)
        return image

    @torch.no_grad()
    def denoise_step(self, x, i, t,guidance_scale,noisy_latent, null_or_negative=False):
        # register the time step and features in pnp injection modules
        latent_model_input = torch.cat(([noisy_latent]+[x] * 2))

        register_time(self, t.item())

        # compute text embeddings
        if null_or_negative:
            text_embed_input = torch.cat([self.pnp_guidance_embeds[i], self.text_embeds], dim=0)
        else:
            text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        _,noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        # if noise_loss is not None:
        #     denoised_latent = torch.concat((denoised_latent[:1]+noise_loss[:1],denoised_latent[1:]))
        return denoised_latent

    def init_pnp(self, conv_injection_t, qk_injection_t, alpha=None):
        self.qk_injectionum_ddim_steps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injectionum_ddim_steps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient(self, self.qk_injectionum_ddim_steps, alpha=alpha)
        register_conv_control_efficient(self, self.conv_injectionum_ddim_steps)

    def run_pnp(self,image_path,noisy_latent,target_prompt,guidance_scale=7.5, uncond_embeddings=None, pnp_f_t=0.8,pnp_attn_t=0.5, alpha=None):
        
        # load image
        self.image = self.get_data(image_path)
        self.eps = noisy_latent[-1]

        self.text_embeds = self.get_text_embeds(target_prompt, "ugly, blurry, black, low res, unrealistic")
        if uncond_embeddings is None:
            self.pnp_guidance_embeds = self.get_text_embeds("", "").chunk(2)[0]
        else:
            self.pnp_guidance_embeds = uncond_embeddings
        
        pnp_f_t = int(self.num_ddim_steps * pnp_f_t)
        pnp_attn_t = int(self.num_ddim_steps * pnp_attn_t)
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t, alpha=alpha)
        if uncond_embeddings is None:
            edited_img = self.sample_loop(self.eps,guidance_scale,noisy_latent)
        else:
            edited_img = self.sample_loop(self.eps,guidance_scale,noisy_latent, True)
        return edited_img

    def sample_loop(self, x,guidance_scale,noisy_latent, null_or_negative=False):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(self.scheduler.timesteps):
                if null_or_negative:
                    x = self.denoise_step(x, i, t,guidance_scale,noisy_latent[-1-i], null_or_negative)
                else:
                    x = self.denoise_step(x, i, t,guidance_scale,noisy_latent[-1-i])
            decoded_latent = self.decode_latent(x)
                
        return decoded_latent



    def edit_image_ddim_PnP(self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        image_shape=[512,512],
        alpha=None
    ):
        torch.cuda.empty_cache()
        image_gt = load_512(image_path)
        _, rgb_reconstruction, latent_reconstruction = self.model.extract_latents(data_path=image_path,
                                            num_steps=self.num_ddim_steps,
                                            inversion_prompt=prompt_src, guidance_scale=guidance_scale)
        
        edited_image=self.run_pnp(image_path,latent_reconstruction,prompt_tar,guidance_scale, alpha=alpha)
        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

        return Image.fromarray(np.concatenate((
            image_instruct,
            image_gt,
            np.uint8(255*np.array(rgb_reconstruction[0].permute(1,2,0).cpu().detach())),
            np.uint8(255*np.array(edited_image[0].permute(1,2,0).cpu().detach())),
            ),1))



    def edit_image_directinversion_PnP(self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        image_shape=[512,512],
        alpha=None
    ):
        torch.cuda.empty_cache()
        image_gt = load_512(image_path)
        inverted_x, rgb_reconstruction, _ = self.model.extract_latents(data_path=image_path,
                                            num_steps=self.num_ddim_steps,
                                            inversion_prompt=prompt_src)

        edited_image=self.run_pnp(image_path,inverted_x,prompt_tar,guidance_scale, alpha=alpha)
        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

        return Image.fromarray(np.concatenate((
            image_instruct,
            image_gt,
            np.uint8(np.array(latent2image(model=self.vae, latents=inverted_x[1].to(self.vae.dtype))[0])),
            np.uint8(255*np.array(edited_image[0].permute(1,2,0).cpu().detach())),
            ),1))

    def edit_image_null_text_inversion_pnp(self,image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        image_shape=[512,512],
        alpha=None
    ):
        torch.cuda.empty_cache()
        image_gt = load_512(image_path)

        null_inversion = NullInversion(model=self.model,
                                    num_ddim_steps=self.num_ddim_steps)

        _, _, inverted_x, uncond_embeddings = null_inversion.invert(
            image_gt=image_gt, prompt=prompt_src,guidance_scale=guidance_scale, alpha=alpha)
        

        edited_image=self.run_pnp(image_path,inverted_x,prompt_tar,guidance_scale, uncond_embeddings, alpha=alpha)
        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

        return Image.fromarray(np.concatenate((
            image_instruct,
            image_gt,
            np.uint8(np.array(latent2image(model=self.vae, latents=inverted_x[1].to(self.vae.dtype))[0])),
            np.uint8(255*np.array(edited_image[0].permute(1,2,0).cpu().detach())),
            ),1))
    
    def edit_image_negative_prompt_inversion_pnp(self,image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        image_shape=[512,512],
        alpha=None
    ):
        torch.cuda.empty_cache()
        image_gt = load_512(image_path)

        negative_inversion = NegativePromptInversion(model=self.model,
                                    num_ddim_steps=self.num_ddim_steps)

        _, _, inverted_x, uncond_embeddings = negative_inversion.invert(
            image_gt=image_gt, prompt=prompt_src, alpha=alpha)
        

        edited_image=self.run_pnp(image_path,inverted_x,prompt_tar,guidance_scale, uncond_embeddings, alpha=alpha)
        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

        return Image.fromarray(np.concatenate((
            image_instruct,
            image_gt,
            np.uint8(np.array(latent2image(model=self.vae, latents=inverted_x[1].to(self.vae.dtype))[0])),
            np.uint8(255*np.array(edited_image[0].permute(1,2,0).cpu().detach())),
            ),1))



