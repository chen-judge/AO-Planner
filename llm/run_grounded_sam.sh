split="val_unseen"
data_path="data/logs/llm/release_test"

python llm/grounded_sam_Gemini.py \
  --config /disk2/jqchen/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint /disk2/jqchen/Grounded-Segment-Anything/groundingdino_swint_ogc.pth \
  --sam_checkpoint /disk2/jqchen/Grounded-Segment-Anything/sam_vit_h_4b8939.pth \
  --input_path ${data_path} \
  --box_threshold 0.4 \
  --text_threshold 0.4 \
  --text_prompt "ground" \
  --device "cuda"