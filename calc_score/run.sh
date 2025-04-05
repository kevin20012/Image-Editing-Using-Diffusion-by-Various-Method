# make_result -> 이미지 표 형태로 만들어서 자르기
# calculate_score -> 마스크를 가지고 점수 계산
# 만들어진 zip 파일을 노션에 임포트하면 자동으로 표로 들어감.

python make_result.py -e "pix2pix_zero" -i "./img" -o "./result/pix2pix_zero/assets"
python  calculate_score.py -e "pix2pix_zero" -i "./img" -o "./result/pix2pix_zero"

zip -r to_notion_pix2pix_zero.zip result/pix2pix_zero/

python make_result.py -e "pnp" -i "./img" -o "./result/pnp/assets"
python  calculate_score.py -e "pnp" -i "./img" -o "./result/pnp"

zip -r to_notion_pnp.zip result/pnp/