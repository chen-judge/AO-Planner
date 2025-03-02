import argparse
import os
import sys
import glob
import json
import torch
import re
import time
from PIL import Image

sys.path.append("/disk2/jqchen/Grounded-Segment-Anything/")  # TODO: set your path to Grounded-SAM

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from prompting.Gemini import gemini_infer


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


def find_start_point(mask, height, width):
    start_point = (width//2, height-5)  # point radius 5
    if mask[start_point[1], start_point[0]]:
        return start_point
    else:
        offest = 0
        for _ in range(width//2):
            offest += 1
            point = (width//2-offest, height-5)
            if mask[point[1], point[0]]:
                print(f'Start point founded: {point}')
                return point

            point = (width//2+offest, height-5)
            if mask[point[1], point[0]]:
                print(f'Start point founded: {point}')
                return point

        raise ValueError('Start Point Not Found!!!')


def sample_points(masks, distance=50, multi_start=False):
    """
    sample points with only one starting point-0
    """
    _, _, height, width = masks.shape
    mask = masks.sum(dim=(0, 1))

    width_num = width // distance
    height_num = height // distance

    if multi_start:
        points_list = []
    else:
        start_point = (width // 2, height - 5)
        points_list = [start_point]
    for i in range(1, height_num):
        for j in range(1, width_num):
            if mask[height - 1 - i * distance, j * distance]:
                points_list.append((j * distance, height - 1 - i * distance))

    print(points_list)
    return points_list


def vis_candidates(image, points, alpha=0.6, multi_start=False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 0.6
    font_scale = 1.0
    font_thickness = 2
    font_color = (255, 0, 0)

    for i, point in enumerate(points):
        if multi_start:
            label = str(i+1)
        else:
            label = str(i)

        overlay = image.copy()
        cv2.circle(overlay, point, 5, (0, 0, 255), -1)
        label_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_origin = (point[0] - label_size[0] // 2, point[1] - label_size[1] // 2)
        cv2.putText(overlay, label, text_origin, font, font_scale, font_color, font_thickness)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image


def vis_paths(image, cand_points_list, waypoints, paths, alpha=0.6, multi_start=False, offset=0):
    visualized_segment = []

    for waypoint, path in zip(waypoints, paths):
        if path[-1] != waypoint:
            path.append(waypoint)

        points = [cand_points_list[point_id] for point_id in path]
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

        if multi_start:
            # add a line to the bottom of the image, otherwise, some bottom waypoints will not have any visualized paths
            overlay = image.copy()
            start_point = [points[0][0], 511]
            end_point = list(points[0])

            # avoid overlap, usually maximum_path=3
            if offset != 0:
                if (start_point, end_point) in visualized_segment:
                    start_point[0] -= offset
                    start_point[1] -= offset
                    end_point[0] -= offset
                    end_point[1] -= offset
                    if (start_point, end_point) in visualized_segment:
                        start_point[0] += 2*offset
                        start_point[1] += 2*offset
                        end_point[0] += 2*offset
                        end_point[1] += 2*offset
                visualized_segment.append((start_point, end_point))

            start_point, end_point = tuple(start_point), tuple(end_point)
            overlay = cv2.line(overlay, start_point, end_point, color, 2)
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


        for i in range(len(points) - 1):
            overlay = image.copy()
            start_point = list(points[i])
            end_point = list(points[i + 1])

            # avoid overlap, usually maximum_path=3
            if offset != 0:
                if (start_point, end_point) in visualized_segment:
                    start_point[0] -= 2
                    start_point[1] -= 2
                    end_point[0] -= 2
                    end_point[1] -= 2
                    if (start_point, end_point) in visualized_segment:
                        start_point[0] += 4
                        start_point[1] += 4
                        end_point[0] += 4
                        end_point[1] += 4
                visualized_segment.append((start_point, end_point))

            start_point, end_point = tuple(start_point), tuple(end_point)
            overlay = cv2.line(overlay, start_point, end_point, color, 2)  # 在图像上绘制线段
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image


def query_llm(image_name, instruction, temperature=0.0):

    # find waypoints and corresponding path
    background = """You are a robot and need to identify potential 'Waypoints' and corresponding 'Paths' in the environment from the current observed image."""
    waypoint_def = f"""'Waypoints' refer to locations that can be reached and meet the following conditions. 1. They are on the ground and maintain a reasonable distance from obstacles to avoid collisions. 2. Ideally, they occupy crucial positions at the center of different regions and can be connected to various regions. 3. Select the most representative waypoints (up to 3), preferably not too close to each other."""
    waypoint_output = f"""Some position candidates on the ground are annotated with IDs in the image. You need to select some of them and provide the IDs as your selected 'Waypoints'."""
    path_output = """For these 'Waypoints', you also need to select some positions that need to be passed through to reach each selected waypoint. For each path, you can start from any of the points in the bottom row of the image. You need to ensure that connecting the selected positions in order can form some shortest 'Paths' that lead to the 'Waypoints' while navigating around obstacles to avoid collisions."""

    output_requirement = """You should return a JSON object that has the fields 'Waypoints' (a list recording waypoints) and 'Paths' (a list recording paths to each waypoint)."""

    if instruction:
        instr_des = f"""'Instruction': '{instruction}', is a step-by-step detailed guidance for navigation, but you might have already executed some of the commands. If key information from the 'Instruction', such as scene descriptions, landmarks, and objects, appears in the observed image, select the corresponding waypoint and path."""
        system = f"""{background} {waypoint_def}\n\n{waypoint_output} {path_output}\n\n{instr_des}\n\n{output_requirement}"""
    else:
        system = f"""{background} {waypoint_def}\n\n{waypoint_output} {path_output}\n\n{output_requirement}"""

    image_list = [image_name]
    answer, tokens = gemini_infer(system=system, text=None, image_list=image_list, temperature=temperature)
    print(answer)

    data = parse_results(answer)
    return data


def parse_results(answer):
    pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(pattern, answer, re.MULTILINE)
    data = {}
    if match:
        json_str = match.group(1)
        try:
            data = json.loads(json_str)
        except:
            data['log'] = f"JSON format error\noriginal anwer: {answer}"
    else:
        data['log'] = f"JSON not found\noriginal anwer: {answer}"

    return data


def process_image(new_files):
    for file_id, image_path in enumerate(new_files):
        print(f'######### {image_path} ##########')
        # load image
        image_pil, image = load_image(image_path)

        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )

        # empty bbox will result in errors when running SAM
        data = {}
        if not pred_phrases:
            data['log'] = 'No bbox founded!'

        else:
            # run SAM
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(device),
                multimask_output=False,
            )

            cand_points_list = sample_points(masks, distance=50, multi_start=True)
            image_points = vis_candidates(image, cand_points_list, alpha=0.8, multi_start=True)
            cand_points_list.insert(0, (256, 512-5))  # add point-0 here

            image_points = cv2.cvtColor(image_points, cv2.COLOR_BGR2RGB)
            image_points_name = image_path.replace('.png', "_sample_points.jpg")
            cv2.imwrite(image_points_name, image_points)

            if len(cand_points_list) > 40:
                data['log'] = f"#############\n An excessive number of points: {len(cand_points_list)}. For example, the agent faces the wall. \n#############\n"
            elif len(cand_points_list) == 1:
                data['log'] = f"#############\n There is very limited ground area and only one starting point. \n#############\n"
            else:

                with open(image_path.replace('.png', '_instr.json'), 'r') as file:
                    instr_data = json.load(file)
                instruction = instr_data["instruction"]

                # predict paths
                data = query_llm(image_points_name, instruction, temperature=0)

                data['cand_points_list'] = cand_points_list
                with open(image_path.replace('.png', ".json"), 'w') as file:
                    json.dump(data, file)

                sleep_time = 1
                print(f'sleeping {sleep_time} seconds for saving {image_path.replace(".png", ".json")}')
                time.sleep(sleep_time)

                try:
                    # visualize paths
                    image_path_pred = vis_paths(image_points, cand_points_list, data['Waypoints'], data['Paths'], alpha=0.8, multi_start=True, offset=0)
                    image_res_name = image_path.replace('.png', "_path_results.jpg")
                    cv2.imwrite(image_res_name, image_path_pred)
                except:
                    data['log'] = '###############\n Some errors in visulization \n#######################'

        print(f'data\n{data}')
        with open(image_path.replace('.png', ".json"), 'w') as file:
            json.dump(data, file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, default=None, help="path to image file")
    parser.add_argument("--input_path", type=str, default=None, help="path to image files")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    image_path = args.input_image
    input_path = args.input_path
    text_prompt = args.text_prompt
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    # load grounding dino model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

    files = set(glob.glob(os.path.join(input_path, '*png')))  # raw image is png while other visualized results are in jpg format
    while True:
        current_files = glob.glob(os.path.join(input_path, '*png'))
        print(f'detecting files: {len(current_files)}')
        time.sleep(1)  # waiting for saving image before processing it
        for img_file in current_files:
            json_file = img_file.replace('.png', '.json')
            if not os.path.exists(json_file):
                process_image([img_file])
                time.sleep(3)  # avoid frequent requests






