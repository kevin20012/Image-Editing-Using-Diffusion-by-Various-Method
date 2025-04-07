import torch
import torch.nn.functional as F
from torchvision import transforms as T
from skimage.io import imread, imsave
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.transform import resize
from skimage import color
import lpips
import os
from Splice.models.extractor import VitExtractor
from PIL import Image
import json
from tqdm import tqdm
import argparse
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAYER_NUM = 11

# define the extractor
dino_preprocess = T.Compose(
    [
        T.Resize(224, antialias=True),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
vit_extractor = VitExtractor("dino_vitb8", device)


def calculate_structure_distance(x1, x2):
    # load the image
    input_img = Image.open(x1).convert("RGB")
    input_img = (
        T.Compose([T.Resize(224, antialias=True), T.ToTensor()])(input_img)
        .unsqueeze(0)
        .to(device)
    )

    edited_img = Image.open(x2).convert("RGB")
    edited_img = (
        T.Compose([T.Resize(224, antialias=True), T.ToTensor()])(edited_img)
        .unsqueeze(0)
        .to(device)
    )

    dist = 0

    for _ in range(LAYER_NUM):
        input_feature = vit_extractor.get_feature_from_input(
            dino_preprocess(input_img)
        )[LAYER_NUM][:, 0, :]
        edited_feature = vit_extractor.get_feature_from_input(
            dino_preprocess(edited_img)
        )[LAYER_NUM][:, 0, :]

        dist += F.cosine_similarity(input_feature, edited_feature)

    return dist.item() / LAYER_NUM


def get_masked_image(x, mask):
    img = imread(x)
    mask_img = imread(mask)
    
    if mask_img.shape[-1] == 4:
        alpha = mask_img[..., 3]  # 알파 채널
    else:
        raise ValueError("마스크 이미지에 알파 채널이 없습니다.")
    
    # 검정색 영역을 식별 (알파 값이 255인 부분)
    black_mask = alpha == 255
    
    # 결과 이미지 생성
    masked = img.copy()
    
    masked[black_mask, ...] = 0 # 검정색 적용
    
    return masked

def calculate_background_preservation(x1, x2, mask):
    im1 = get_masked_image(x1, mask)
    im1 = resize(im1, (512, 512))
    # im1 = color.rgba2rgb(im1)

    im2 = get_masked_image(x2, mask)
    im2 = resize(im2, (512, 512))
    # im2 = color.rgba2rgb(im2)
    
    psnr_val = psnr(im1, im2)
    ssim_val = ssim(im1, im2, multichannel=True, channel_axis=2, data_range=1.0)

    lpips_alex = lpips.LPIPS(net="alex", verbose=False)
    lpips_val = lpips_alex(
        torch.from_numpy(im1).permute(2, 0, 1).to(torch.float),
        torch.from_numpy(im2).permute(2, 0, 1).to(torch.float),
    )

    return psnr_val, ssim_val, lpips_val

def print_score(image_path, edit_tech, image_name, image_info, file_st, ext="png"):
    print = file_st.write
    image_path = os.path.join(image_path, edit_tech)
    EXT = ext #"png"  # 이미지 확장자

    image_original = image_path +"/"+ image_name + "_original." + EXT
    image_ddim_wo_ours = image_path +"/"+ image_name + "_ddim_wo_ours."+ EXT
    image_ddim_w_ours = image_path +"/"+ image_name + "_ddim_w_ours."+ EXT
    image_direct_wo_ours = image_path +"/"+ image_name + "_directinversion_wo_ours."+ EXT
    image_direct_w_ours = image_path +"/"+ image_name + "_directinversion_w_ours."+ EXT
    image_neg_wo_ours = image_path +"/"+ image_name + "_negative-prompt-inversion_wo_ours."+ EXT
    image_neg_w_ours = image_path +"/"+ image_name + "_negative-prompt-inversion_w_ours."+ EXT
    image_null_wo_ours = image_path +"/"+ image_name + "_null-text-inversion_wo_ours."+ EXT
    image_null_w_ours = image_path +"/"+ image_name + "_null-text-inversion_w_ours."+ EXT
    image_mask = image_path +"/"+ image_name + "_mask."+ EXT
    
    ddim_wo_ours_score = (
        calculate_structure_distance(image_original, image_ddim_wo_ours),
        *calculate_background_preservation(image_original, image_ddim_wo_ours, image_mask),
    )
    ddim_w_ours_score = (
        calculate_structure_distance(image_original, image_ddim_w_ours),
        *calculate_background_preservation(image_original, image_ddim_w_ours, image_mask),
    )
    direct_wo_ours_score = (
        calculate_structure_distance(image_original, image_direct_wo_ours),
        *calculate_background_preservation(image_original, image_direct_wo_ours, image_mask),
    )
    direct_w_ours_score = (
        calculate_structure_distance(image_original, image_direct_w_ours),
        *calculate_background_preservation(image_original, image_direct_w_ours, image_mask),
    )
    neg_wo_ours_score = (
        calculate_structure_distance(image_original, image_neg_wo_ours),
        *calculate_background_preservation(image_original, image_neg_wo_ours, image_mask),
    )
    neg_w_ours_score = (
        calculate_structure_distance(image_original, image_neg_w_ours),
        *calculate_background_preservation(image_original, image_neg_w_ours, image_mask),
    )
    null_wo_ours_score = (
        calculate_structure_distance(image_original, image_null_wo_ours),
        *calculate_background_preservation(image_original, image_null_wo_ours, image_mask),
    )
    null_w_ours_score = (
        calculate_structure_distance(image_original, image_null_w_ours),
        *calculate_background_preservation(image_original, image_null_w_ours, image_mask),
    )

    def get_bold(num, high_best=True):
        score_list = [ddim_wo_ours_score[num], ddim_w_ours_score[num], direct_wo_ours_score[num], direct_w_ours_score[num], neg_wo_ours_score[num], neg_w_ours_score[num], null_wo_ours_score[num], null_w_ours_score[num]]
        
        result = []
        for i in range(len(score_list)//2):
            temp_list = score_list[i*2:i*2+2]
            target = sorted(temp_list)[-1 if high_best else 0]

            if temp_list[0] == target:
                temp = "<strong>%.4f</strong>" % temp_list[0]
                result.append(temp)
                temp = "%.4f" % temp_list[1]
                result.append(temp)
            else:
                temp = "%.4f" % temp_list[0]
                result.append(temp)
                temp = "<strong>%.4f</strong>" % temp_list[1]
                result.append(temp)
                
            
        return result

    # print(f"**"+image_info["original_prompt"]+"**  \n")
    # print(f"**→ " + image_info["editing_prompt"]+"**\n")
    # # print("────────────────────────────────────────────────────────────\n")
    # # print("METHOD : structure_distance↑ / background_preservation (PSNR↑ / SSIM↑ / LPIPS↓)\n")
    # # print("────────────────────────────────────────────────────────────\n")
    # # print("ours   : %.4f / (%.4f / %.4f / %.4f)\n" % ours_score)
    # # print("ddim_w  : %.4f / (%.4f / %.4f / %.4f)\n" % ddim_w_score)
    # # print("direct_w : %.4f / (%.4f / %.4f / %.4f)\n" % direct_w_score)
    # # print("negative_w : %.4f / (%.4f / %.4f / %.4f)\n" % neg_w_score)
    # # print("null_text_w : %.4f / (%.4f / %.4f / %.4f)\n" % null_w_score)
    # # print("────────────────────────────────────────────────────────────\n")
    # print(f"<img src='assets/{image_name}.png'>\n")
    # print("| 지표 ↓ / 모델 → | Ours | w/ DDIM | w/ Direct Inversion | w/ Negative Inversion | w/ Null-text Inversion\n")
    # print("| :-- | :--: | :--: | :--: | :--: | :--: |\n")
    # print("| Structure Distance ↑ | %s | %s | %s | %s | %s |\n" % get_bold(0, True))
    # print("| Background Preservation (PSNR ↑) | %s | %s | %s | %s | %s |\n" % get_bold(1, True))
    # print("| Background Preservation (SSIM ↑) | %s | %s | %s | %s | %s |\n" % get_bold(2, True))
    # print("| Background Preservation (LPIPS ↓) | %s | %s | %s | %s | %s |\n\n" % get_bold(3, False))

    # get_bold 함수 결과를 미리 계산하여 변수에 저장합니다.
    bold0 = get_bold(0, True)
    bold1 = get_bold(1, True)
    bold2 = get_bold(2, True)
    bold3 = get_bold(3, False)

    print("<p><strong>" + image_info["original_prompt"] + "</strong></p>")
    print("<p><strong>→ " + image_info["editing_prompt"] + "</strong></p>")
    print("<p><img src='assets/" + image_name + ".png' alt='Image'></p>")
    print("<table border='1' cellspacing='0' cellpadding='5'>")
    print("  <thead>")
    print("    <tr>")
    print("      <th>지표 ↓ / 모델 →</th>")
    print("      <th>ddim</th>")
    print("      <th>ddim w/ ours</th>")
    print("      <th>direct_inv</th>")
    print("      <th>direct_inv w/ ours</th>")
    print("      <th>negative_inv</th>")
    print("      <th>negative_inv w/ ours</th>")
    print("      <th>null text inv</th>")
    print("      <th>null text inv w/ ours</th>")
    print("    </tr>")
    print("  </thead>")
    print("  <tbody>")
    print("    <tr>")
    print("      <td>Structure Distance ↑</td>")
    for i in range(8):
        print("      <td>" + bold0[i] + "</td>")
    print("    </tr>")
    print("    <tr>")
    print("      <td>Background Preservation (PSNR ↑)</td>")
    for i in range(8):
        print("      <td>" + bold1[i] + "</td>")
    print("    </tr>")
    print("    <tr>")
    print("      <td>Background Preservation (SSIM ↑)</td>")
    for i in range(8):
        print("      <td>" + bold2[i] + "</td>")
    print("    </tr>")
    print("    <tr>")
    print("      <td>Background Preservation (LPIPS ↓)</td>")
    for i in range(8):
        print("      <td>" + bold3[i] + "</td>")
    print("    </tr>")
    print("  </tbody>")
    print("</table>")



parser = argparse.ArgumentParser()
parser.add_argument("-e", "--edit_tech", type=str, default="pix2pix_zero")
parser.add_argument("-i", "--img_path", type=str, default="./img")
parser.add_argument("-o", "--output", type=str, default="./")

args = parser.parse_args()

if __name__ == "__main__":
    img_path = args.img_path

    with open(f"../PIE-Bench/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)
    # 이미지 개수 찾기
    img_path = os.path.join(args.img_path, args.edit_tech)
    files = glob.glob(f"{img_path}/*.png")
    # 그룹: key = 이미지 번호, value = {컬럼명: 파일 경로}
    groups = {}

    for file in files:
        if "_" not in file:
            continue  # '_'가 없는 파일은 건너뜁니다.
        # 예시: "000000000020_original.png" -> 번호: "000000000020", 나머지: "original.png"
        file = file.split(img_path+"/")[1]
        prefix, rest = file.split("_", 1)
        prefix = int(prefix)
        groups[prefix] = 1

    # 그룹을 정렬 (이미지 번호 오름차순)
    sorted_keys = sorted(groups.keys())

    datas = sorted(editing_instruction.items(), key=lambda x: x[0])[sorted_keys[0]:sorted_keys[-1]+1]
    if os.path.exists(args.output) == False:
        os.mkdir(args.output)
    output = os.path.join(args.output, args.edit_tech + "_calc.html")
    f = open(output, "w")
    for key, item in tqdm(datas, desc=f"{args.edit_tech}..."):
        print_score(args.img_path, args.edit_tech, key, item, f)
    f.close()

