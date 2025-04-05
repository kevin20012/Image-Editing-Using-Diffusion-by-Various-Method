# available editing methods combination: 
    # [ddim, null-text-inversion, negative-prompt-inversion, directinversion] + 
    # [p2p, masactrl, pix2pix_zero, pnp]

# p2p only!
conda run -n p2p --no-capture-output python -u run_one_img.py --data_path img \
                --output_path output \
                --edit_method_list directinversion+p2p

# masactrl only!
conda run -n masactrl --no-capture-output python -u run_one_img.py --data_path img \
                --output_path output \
                --edit_method_list null-text-inversion+masactrl
# # pix2pix_zero only!
conda run -n pix2pix_zero --no-capture-output python -u run_one_img.py --data_path img \
                --output_path output \
                --edit_method_list directinversion+pix2pix_zero

# # pnp only!
conda run -n pnp --no-capture-output python -u run_one_img.py --data_path img \
                --output_path output \
                --edit_method_list negative-prompt-inversion+pnp