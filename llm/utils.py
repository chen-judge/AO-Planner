import os
import time
import json
import numpy as np
from PIL import Image
import cv2


def llm_waypoint_predictor_single_view(observation, img_file_name, rgb_name, position, heading):
    image = observation[rgb_name]
    Image.fromarray(image).save(img_file_name)
    json_file_name = img_file_name.replace('.png', '.json')

    while not os.path.exists(json_file_name):
        time.sleep(1)

    # print(f'waiting for saving {json_file_name}...')
    time.sleep(1)

    with open(json_file_name, 'r') as f:
        llm_result = json.load(f)  # cand_points_list, waypoints and paths

    if 'log' not in llm_result.keys():
        if llm_result['Paths']:
            depth_name = rgb_name.replace('rgb', 'depth')
            ob_depth = observation[depth_name] * 10.0  # denormalization

            ob_angle = parse_ob_angle(rgb_name)
            waypoints = []
            paths = []

            if len(llm_result["Waypoints"]) == len(llm_result["Paths"]):
                for waypoint_id in llm_result["Waypoints"]:
                    u_image, v_image = llm_result["cand_points_list"][waypoint_id]
                    angle, xz_dist, world_position = pixel_to_world(u_image, v_image, ob_depth, ob_angle, position, heading)
                    waypoints.append(
                        {
                            "angle": angle,  # angle in the camera coordinate system, counterclockwise
                            "distance": xz_dist,
                            "position": world_position
                        }
                    )
            else:
                print('different number between waypoints and paths, use path[-1] as waypoint instead')
                for path_ids in llm_result["Paths"]:
                    waypoint_id = path_ids[-1]
                    u_image, v_image = llm_result["cand_points_list"][waypoint_id]
                    angle, xz_dist, world_position = pixel_to_world(u_image, v_image, ob_depth, ob_angle, position,
                                                                    heading)
                    waypoints.append(
                        {
                            "angle": angle,  # angle in the camera coordinate system, counterclockwise
                            "distance": xz_dist,
                            "position": world_position
                        }
                    )

            for path_ids in llm_result["Paths"]:
                if path_ids:
                    path = []
                    for point_id in path_ids:
                        if point_id == 0:  # neglect the starting point 0 in the image
                            continue
                        u_image, v_image = llm_result["cand_points_list"][point_id]
                        _, _, world_position = pixel_to_world(u_image, v_image, ob_depth, ob_angle, position, heading)
                        path.append(world_position.tolist())
                    paths.append(path)

            llm_waypoint_path_cands = {"waypoints": waypoints, "paths": paths}
            return llm_result, llm_waypoint_path_cands

    print("No waypoints in this observation")
    return None, None


def parse_ob_angle(rgb_name):
    parts = rgb_name.split('_')
    if len(parts) == 1:
        ob_angle = 0.0
    else:
        ob_angle = float(parts[-1])
        ob_angle = np.deg2rad(ob_angle)

    return ob_angle


def pixel_to_world(u_image, v_image, ob_depth, ob_angle, position, heading):
    camera_relative_position = pixel_to_camera(u_image, v_image, ob_depth, hfov=90, image_width=512, image_height=512)
    theta, xz_dist = angle_distance_in_camera(camera_relative_position)

    base_heading = heading - np.pi + ob_angle  # -np.pi because agent_angle=0 is towards the z axis instead of -z axis
    world_position = camera_to_world(camera_relative_position, position, theta, xz_dist, base_heading=base_heading)

    angle = theta - np.pi / 2  # set the camera's forward direction to 0 degrees and the angle range is [-pi/4, pi/4]
    waypoint_angle = angle + ob_angle

    return waypoint_angle, xz_dist, world_position


def pixel_to_camera(u_image, v_image, ob_depth, hfov, image_width=512, image_height=512):
    """
    https://aihabitat.org/docs/habitat-api/view-transform-warp.html
    """
    hfov = float(hfov) * np.pi / 180.  # important

    intrinsic_k = np.array([
        [1 / np.tan(hfov / 2.), 0., 0., 0.],
        [0., 1 / np.tan(hfov / 2.), 0., 0.],
        [0., 0., 1, 0],
        [0., 0., 0, 1]])

    u = u_image/image_width * 2 - 1  # u [-1, 1]
    v = 1 - v_image/image_height * 2  # v [1, -1]

    depth = ob_depth[v_image, u_image, 0]  # make sure that RGB and Depth are same in size.

    uv_c0 = np.array([u*depth, v*depth, -depth, 1.0])
    camera_relative_position = np.matmul(np.linalg.inv(intrinsic_k), uv_c0)

    return camera_relative_position


def angle_distance_in_camera(camera_relative_position):
    rel_x = camera_relative_position[0]
    rel_z = -camera_relative_position[2]
    xz_dist = max(np.sqrt(rel_x ** 2 + rel_z ** 2), 1e-8)
    theta = np.arccos(rel_x / xz_dist)  # [0, pi] in radian
    assert np.pi / 4 <= theta <= 3 * np.pi / 4  # HFOV=90

    return theta, xz_dist


def camera_to_world(camera_relative_position, camera_base_position, theta, xz_dist, base_heading=0):
    theta += base_heading
    dx = xz_dist * np.cos(theta)
    dz = xz_dist * np.sin(theta)
    dy = camera_relative_position[1]

    rel_coordinates = [dx, dy, -dz]
    world_position = rel_coordinates + camera_base_position

    return world_position


def vis_ghost_nodes(img_file_name, llm_result, current_ghost_cnt, img_to_ghost_node_dict, ghost_node_to_img_dict, alpha=0.8, multi_start=True, offset=0):
    """
    modified original implementation in grounded_sam_Gemini.py
    """
    image = cv2.imread(img_file_name)

    cand_points_list, waypoints, paths = llm_result["cand_points_list"], llm_result["Waypoints"], llm_result["Paths"]

    visualized_segment = []

    for i, (waypoint, path) in enumerate(zip(waypoints, paths)):
        # if 0 not in path:
        #     path.insert(0, 0)
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
            overlay = cv2.line(overlay, start_point, end_point, color, 2)  # 在图像上绘制线段
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

        ghost_id = str(current_ghost_cnt)
        current_ghost_cnt += 1
        ghost_node_image_name = img_file_name.replace('.png', "_ghost_results.jpg")

        if ghost_node_image_name not in img_to_ghost_node_dict.keys():
            img_to_ghost_node_dict[ghost_node_image_name] = [ghost_id]
        else:
            img_to_ghost_node_dict[ghost_node_image_name].append(ghost_id)

        ghost_node_to_img_dict[ghost_id] = ghost_node_image_name
        image = vis_points(image, cand_points_list[waypoint], ghost_id, alpha=0.8)

    return image, current_ghost_cnt, img_to_ghost_node_dict, ghost_node_to_img_dict


def vis_points(image, point, label, alpha=0.6, multi_start=False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 0.6
    font_scale = 1.0
    font_thickness = 2
    font_color = (255, 0, 0)

    overlay = image.copy()
    cv2.circle(overlay, point, 5, (0, 0, 255), -1)

    label_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    text_origin = (point[0] - label_size[0], point[1] - label_size[1] // 2)
    cv2.putText(overlay, label, text_origin, font, font_scale, font_color, font_thickness)

    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image