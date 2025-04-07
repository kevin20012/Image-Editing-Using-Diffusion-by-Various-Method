import os
import glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse

# --- 1. 컬럼 순서 및 파일명 매핑 ---
columns = ["original", "mask", "ddim", "ddim w/ ours", "direct_inv", "direct_inv w/ ours", "negative_inv", "negative_inv w/ ours", "null text inv", "null text inv w/ ours"]

# 파일명에 사용할 이름 매핑 (파일명에서는 공백 대신 '-' 사용)
col_mapping = {
    "original": "original",
    "mask": "mask",
    "ddim": "ddim_wo_ours",
    "ddim w/ ours": "ddim_w_ours",
    "direct_inv": "directinversion_wo_ours",
    "direct_inv w/ ours": "directinversion_w_ours",
    "negative_inv": "negative-prompt-inversion_wo_ours",
    "negative_inv w/ ours": "negative-prompt-inversion_w_ours",
    "null text inv": "null-text-inversion_wo_ours",
    "null text inv w/ ours": "null-text-inversion_w_ours"
}

# --- 2. 파일 목록 읽어와서 그룹화 ---
# 현재 디렉토리에서 png 파일 모두 가져오기
def draw_result(edit_tech, img_dir, output_dir):
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)
    target_dir = edit_tech
    files = glob.glob(f"{img_dir}/{target_dir}/*.png")
    # 그룹: key = 이미지 번호, value = {컬럼명: 파일 경로}
    groups = {}

    for file in files:
        if "_" not in file:
            continue  # '_'가 없는 파일은 건너뜁니다.
        # 예시: "000000000020_original.png" -> 번호: "000000000020", 나머지: "original.png"
        file = file.split(target_dir+"/")[1]
        prefix, rest = file.split("_", 1)
        # 확장자 제거 후 type 부분 추출
        file_type = os.path.splitext(rest)[0]
        # 그룹에 추가 (이미지 번호별로 딕셔너리 생성)
        if prefix not in groups:
            groups[prefix] = {}
        groups[prefix][file_type] = file

    # 그룹을 정렬 (이미지 번호 오름차순)
    sorted_keys = sorted(groups.keys())

    # --- 3. 기본 이미지 크기 및 셀, 여백 설정 ---
    # 모든 이미지가 동일한 크기라고 가정하고, 첫 그룹의 첫번째 이미지(예: original)를 사용
    sample_image_path = None
    print(sorted_keys)
    for key in sorted_keys:
        print(key)
        if col_mapping["original"] in groups[key]:
            sample_image_path = os.path.join(img_dir, target_dir, groups[key][col_mapping["original"]])
            break

    if sample_image_path is None:
        raise ValueError("original 이미지 파일을 찾을 수 없습니다.")

    sample_image = Image.open(sample_image_path)
    img_width, img_height = sample_image.size
    img_width = img_width//2
    img_height = img_height//2
    sample_image.resize((img_width, img_height))

    # 셀 여백과 텍스트 영역 크기 설정
    padding = 10  # 각 셀 사이의 여백
    header_height = 50  # 컬럼명 헤더 영역 높이
    number_col_width = 200  # 이미지 번호를 출력할 왼쪽 열의 폭

    # --- 4. 전체 캔버스 크기 계산 ---
    # 총 열: 왼쪽 번호 열 + 이미지 7개
    total_cols = 1 + len(columns)
    total_rows = 1 + len(sorted_keys)  # 헤더 행 + 각 그룹

    canvas_width = number_col_width + len(columns) * (img_width + padding) + padding
    canvas_height = len(sorted_keys) * (header_height + padding + img_height + padding)

    # --- 5. 캔버스 생성 및 그리기 준비 ---
    merged_img = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(merged_img)
    font = ImageFont.truetype("Times New Roman.ttf", 30)  # 기본 폰트 사용

    # --- 6. 헤더 행 (컬럼명) 그리기 ---
    # 왼쪽 번호 셀은 빈칸 혹은 "번호" 텍스트 처리
    header_x = padding
    header_y = padding
    # draw.text((header_x + (number_col_width - draw.textsize("Image Number", font=font)[0]) / 2,
    #            header_y + (header_height - draw.textsize("Image Number", font=font)[1]) / 2),
    #           "Image Number", fill="black", font=font)

    # 각 컬럼명 그리기
    def draw_column_title(row_index):
        for i, col in enumerate(columns):
            # 각 셀의 시작 x 좌표 계산 (왼쪽 번호 열 이후)
            cell_x = number_col_width + padding + i * (img_width + padding)
            # 중앙에 텍스트 위치 계산
            text_w, text_h = draw.textsize(col, font=font)
            text_x = cell_x + (img_width - text_w) / 2
            text_y = header_y + (header_height - text_h) / 2 + row_index * (header_height + padding + img_height + padding)
            draw.text((text_x, text_y), col, fill="black", font=font)

    # --- 7. 각 그룹별로 행 추가 (왼쪽 번호 텍스트와 이미지 붙여넣기) ---
    for row_index, key in enumerate(sorted_keys):
        draw_column_title(row_index)
        # 현재 행의 y 좌표 (헤더 후, 패딩 포함)
        if row_index == 0:
            cell_y = header_height + padding
        else:
            cell_y = header_height + padding + row_index * (header_height + padding + img_height + padding)
        
        # 왼쪽 번호 셀에 이미지 번호 출력 (중앙 정렬)
        num_text = key
        text_w, text_h = draw.textsize(num_text, font=font)
        text_x = padding + (number_col_width - text_w) / 2
        text_y = cell_y + (img_height - text_h) / 2
        draw.text((text_x, text_y), num_text, fill="black", font=font)
        
        # 각 컬럼에 대해 이미지 붙여넣기
        for col_index, col in enumerate(columns):
            # 파일 이름에 사용할 타입명 (매핑)
            file_type = col_mapping[col]
            # 해당 그룹에 해당 이미지가 있으면 열기
            if file_type in groups[key]:
                try:
                    if file_type == "mask":
                        img = Image.open(os.path.join(img_dir, target_dir, groups[key][file_type])).convert('RGB')
                        img = np.array(img)
                        img[img==1] = 255
                        img = Image.fromarray(img).resize((img_width, img_height))
                    else:
                        img = Image.open(os.path.join(img_dir, target_dir, groups[key][file_type])).resize((img_width, img_height))
                except Exception as e:
                    print(f"이미지 열기 실패: {groups[key][file_type]}, 에러: {e}")
                    continue
            else:
                # 해당 이미지가 없으면 건너뜁니다.
                continue
            # 이미지 붙여넣을 x 좌표 계산
            cell_x = number_col_width + padding + col_index * (img_width + padding)
            merged_img.paste(img, (int(cell_x), int(cell_y)))

    # --- 8. 최종 이미지 저장 ---

    for idx, img_num in enumerate(sorted_keys):
        crop_img = merged_img.crop((0, canvas_height//len(sorted_keys)*idx, canvas_width, canvas_height//len(sorted_keys)*(idx+1)))
        output_path = os.path.join(output_dir, img_num+".png")
        crop_img.save(output_path)

def draw_result_for_all(edit_tech, img_dir, output_dir):
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)
    target_dir = edit_tech
    files = glob.glob(f"{img_dir}/{target_dir}/*.png")
    # 그룹: key = 이미지 번호, value = {컬럼명: 파일 경로}
    groups = {}

    for file in files:
        if "_" not in file:
            continue  # '_'가 없는 파일은 건너뜁니다.
        # 예시: "000000000020_original.png" -> 번호: "000000000020", 나머지: "original.png"
        file = file.split(target_dir+"/")[1]
        prefix, rest = file.split("_", 1)
        # 확장자 제거 후 type 부분 추출
        file_type = os.path.splitext(rest)[0]
        # 그룹에 추가 (이미지 번호별로 딕셔너리 생성)
        if prefix not in groups:
            groups[prefix] = {}
        groups[prefix][file_type] = file

    # 그룹을 정렬 (이미지 번호 오름차순)
    sorted_keys = sorted(groups.keys())

    # --- 3. 기본 이미지 크기 및 셀, 여백 설정 ---
    # 모든 이미지가 동일한 크기라고 가정하고, 첫 그룹의 첫번째 이미지(예: original)를 사용
    sample_image_path = None
    print(sorted_keys)
    for key in sorted_keys:
        print(key)
        if col_mapping["original"] in groups[key]:
            sample_image_path = os.path.join(img_dir, target_dir, groups[key][col_mapping["original"]])
            break

    if sample_image_path is None:
        raise ValueError("original 이미지 파일을 찾을 수 없습니다.")

    sample_image = Image.open(sample_image_path)
    img_width, img_height = sample_image.size
    img_width = img_width//2
    img_height = img_height//2
    sample_image.resize((img_width, img_height))

    # 셀 여백과 텍스트 영역 크기 설정
    padding = 10  # 각 셀 사이의 여백
    header_height = 50  # 컬럼명 헤더 영역 높이
    number_col_width = 200  # 이미지 번호를 출력할 왼쪽 열의 폭

    # --- 4. 전체 캔버스 크기 계산 ---
    # 총 열: 왼쪽 번호 열 + 이미지 7개
    total_cols = 1 + len(columns)
    total_rows = 1 + len(sorted_keys)  # 헤더 행 + 각 그룹

    canvas_width = number_col_width + len(columns) * (img_width + padding) + padding
    canvas_height = header_height + len(sorted_keys) * (img_height + padding) + padding

    # --- 5. 캔버스 생성 및 그리기 준비 ---
    merged_img = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(merged_img)
    font = ImageFont.truetype("Times New Roman.ttf", 30)  # 기본 폰트 사용

    # --- 6. 헤더 행 (컬럼명) 그리기 ---
    # 왼쪽 번호 셀은 빈칸 혹은 "번호" 텍스트 처리
    header_x = padding
    header_y = padding
    # draw.text((header_x + (number_col_width - draw.textsize("Image Number", font=font)[0]) / 2,
    #            header_y + (header_height - draw.textsize("Image Number", font=font)[1]) / 2),
    #           "Image Number", fill="black", font=font)

    # 각 컬럼명 그리기
    def draw_column_title(row_index):
        for i, col in enumerate(columns):
            # 각 셀의 시작 x 좌표 계산 (왼쪽 번호 열 이후)
            cell_x = number_col_width + padding + i * (img_width + padding)
            # 중앙에 텍스트 위치 계산
            text_w, text_h = draw.textsize(col, font=font)
            text_x = cell_x + (img_width - text_w) / 2
            text_y = header_y + (header_height - text_h) / 2
            draw.text((text_x, text_y), col, fill="black", font=font)

    # --- 7. 각 그룹별로 행 추가 (왼쪽 번호 텍스트와 이미지 붙여넣기) ---
    for row_index, key in enumerate(sorted_keys):
        if row_index == 0:
            draw_column_title(row_index)
        # 현재 행의 y 좌표 (헤더 후, 패딩 포함)
        cell_y = header_height + padding + row_index * (img_height + padding)
        
        # 왼쪽 번호 셀에 이미지 번호 출력 (중앙 정렬)
        num_text = key
        text_w, text_h = draw.textsize(num_text, font=font)
        text_x = padding + (number_col_width - text_w) / 2
        text_y = cell_y + (img_height - text_h) / 2
        draw.text((text_x, text_y), num_text, fill="black", font=font)
        
        # 각 컬럼에 대해 이미지 붙여넣기
        for col_index, col in enumerate(columns):
            # 파일 이름에 사용할 타입명 (매핑)
            file_type = col_mapping[col]
            # 해당 그룹에 해당 이미지가 있으면 열기
            if file_type in groups[key]:
                try:
                    if file_type == "mask":
                        img = Image.open(os.path.join(img_dir, target_dir, groups[key][file_type])).convert('RGB')
                        img = np.array(img)
                        img[img==1] = 255
                        img = Image.fromarray(img).resize((img_width, img_height))
                    else:
                        img = Image.open(os.path.join(img_dir, target_dir, groups[key][file_type])).resize((img_width, img_height))
                except Exception as e:
                    print(f"이미지 열기 실패: {groups[key][file_type]}, 에러: {e}")
                    continue
            else:
                # 해당 이미지가 없으면 건너뜁니다.
                continue
            # 이미지 붙여넣을 x 좌표 계산
            cell_x = number_col_width + padding + col_index * (img_width + padding)
            merged_img.paste(img, (int(cell_x), int(cell_y)))

    # --- 8. 최종 이미지 저장 ---
    output_path = os.path.join(output_dir, target_dir+".png")
    merged_img.save(output_path)



parser = argparse.ArgumentParser()
parser.add_argument("-e", "--edit_tech", type=str, default="p2p")
parser.add_argument("-i", "--img_path", type=str, default="./img_temp")
parser.add_argument("-o", "--output", type=str, default="./")

args = parser.parse_args()

if __name__ == "__main__":
    draw_result(args.edit_tech, args.img_path, args.output)
    draw_result_for_all(args.edit_tech, args.img_path, args.output)