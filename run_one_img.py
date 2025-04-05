import os 
import numpy as np
import argparse
import json
from PIL import Image
import torch
import random

def mask_decode(encoded_mask,image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))
    
    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1
            
    mask_array=mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1
            
    return mask_array


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--data_path', type=str, default="data") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="output") # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs = '+', type=str, default=["ddim+p2p"]) # the editing methods that needed to run
    # available editing methods combination: 
    # [ddim, null-text-inversion, negative-prompt-inversion, directinversion, inversion-free-editing] + 
    # [p2p, masactrl, pix2pix_zero, pnp]
    args = parser.parse_args()
    
    rerun_exist_images=args.rerun_exist_images
    data_path=args.data_path
    output_path=args.output_path
    edit_category_list=args.edit_category_list
    edit_method_list=args.edit_method_list
    
    
    
    # with open(f"{data_path}/mapping_file.json", "r") as f:
    #     editing_instruction = json.load(f)
    
    # for key, item in editing_instruction.items():
        
    #     if item["editing_type_id"] not in edit_category_list:
    #         continue
        
    original_prompt = "A white horse running in the field"#item["original_prompt"].replace("[", "").replace("]", "")
    editing_prompt = "Water color of a white horse running in the field"#item["editing_prompt"].replace("[", "").replace("]", "")
    image_path = "./img/horse.png"#os.path.join(f"{data_path}/annotation_images", item["image_path"])
    editing_instruction = ""#item["editing_instruction"]
    blended_word = []#item["blended_word"].split(" ") if item["blended_word"] != "" else []
    # mask = Image.fromarray(np.uint8(mask_decode(item["mask"])[:,:,np.newaxis].repeat(3,2))).convert("L")
    mask = Image.fromarray(np.uint8(mask_decode([1 for i in range(512*512)])[:,:,np.newaxis].repeat(3,2))).convert("L")

    for edit_method in edit_method_list:
        present_image_save_path=image_path.replace(data_path, os.path.join(output_path,edit_method))
        print(present_image_save_path)
        print(os.path.exists(present_image_save_path), rerun_exist_images)
        if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
            print(f"editing image [{image_path}] with [{edit_method}]")
            setup_seed()
            torch.cuda.empty_cache()
            print(edit_method)
            if edit_method.split('+')[-1] == "p2p":
                from models.p2p.p2p_editor import P2PEditor
                p2p_editor=P2PEditor(edit_method_list, torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),num_ddim_steps=50)
                edited_image = p2p_editor(edit_method,
                                        image_path=image_path,
                                    prompt_src=original_prompt,
                                    prompt_tar=editing_prompt,
                                    guidance_scale=7.5,
                                    cross_replace_steps=0.4,
                                    self_replace_steps=0.6,
                                    blend_word=(((blended_word[0], ),
                                                (blended_word[1], ))) if len(blended_word) else None,
                                    eq_params={
                                        "words": (blended_word[1], ),
                                        "values": (2, )
                                    } if len(blended_word) else None,
                                    proximal="l0",
                                    quantile=0.75,
                                    use_inversion_guidance=True,
                                    recon_lr=1,
                                    recon_t=400,
                                    )
            elif edit_method.split('+')[-1] == "masactrl":
                from models.masactrl.masactrl import MasaCtrlEditor
                masactrl_editor=MasaCtrlEditor(edit_method_list, torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') )
                edited_image = masactrl_editor(edit_method,
                                        image_path=image_path,
                                    prompt_src=original_prompt,
                                    prompt_tar=editing_prompt,
                                    guidance_scale=7.5,
                                    step=4,
                                    layper=10
                                    )
            elif edit_method.split('+')[-1] == "pix2pix_zero":
                from models.pix2pix_zero.pix2pix_zero import Pix2PixZeroEditor
                pix2pix_zero = Pix2PixZeroEditor(edit_method_list, torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
                edited_image = pix2pix_zero(edit_method, image_path=image_path, prompt_src=original_prompt, prompt_tar=editing_prompt, guidance_scale=7.5)
            elif edit_method.split('+')[-1] == "pnp":
                from models.pnp.pnp import PNP as PNPEditor
                pnp_editor = PNPEditor(50, torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
                edited_image = pnp_editor(edit_method, image_path=image_path, prompt_src=original_prompt, prompt_tar=editing_prompt, guidance_scale=7.5)
            
            if not os.path.exists(os.path.dirname(present_image_save_path)):
                os.makedirs(os.path.dirname(present_image_save_path))
            edited_image.save(present_image_save_path)
            
            print(f"finish")
            
        else:
            print(f"skip image [{image_path}] with [{edit_method}]")