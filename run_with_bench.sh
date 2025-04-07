# available editing methods combination: 
    # [ddim, null-text-inversion, negative-prompt-inversion, directinversion] + 
    # [p2p, masactrl, pix2pix_zero, pnp]

# p2p only!
conda run -n sdxl_null --no-capture-output python -u run_with_bench.py \
                --output_path ./calc_score/img \
                --edit_category_list 0 \
                --edit_method_list ddim+p2p negative-prompt-inversion+p2p directinversion+p2p null-text-inversion+p2p \

# masactrl only!
# conda run -n masactrl --no-capture-output python -u run_with_bench.py \
#                 --output_path ./calc_score/img \
#                 --edit_category_list 0 \
#                 --edit_method_list ddim+masactrl negative-prompt-inversion+masactrl directinversion+masactrl null-text-inversion+masactrl \
# # pix2pix_zero only!
# conda run -n pix2pix_zero --no-capture-output python -u run_with_bench.py \
#                 --output_path ./calc_score/img \
#                 --edit_category_list 0 \
#                 --edit_method_list null-text-inversion+pix2pix_zero ours+pix2pix_zero ddim+pix2pix_zero ours+pix2pix_zero null-text-inversion+pix2pix_zero negative-prompt-inversion+pix2pix_zero directinversion+pix2pix_zero \
#                 --rerun_exist_images
# # pnp only!
# conda run -n pnp --no-capture-output python -u run_with_bench.py \
#                 --output_path ./calc_score/img \
#                 --edit_category_list 0 \
#                 --edit_method_list ddim+pnp negative-prompt-inversion+pnp directinversion+pnp null-text-inversion+pnp \
#                 --rerun_exist_images