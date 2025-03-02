import re
import cv2
from llm.prompting.GPT_api import make_image_content
import json


class PromptManager(object):
    def __init__(self, args):
        self.args = args
        self.history = []
        self.planning = []

    def make_graph_history(self, action, t, direction=None, selected_img=None):

        if direction:
            if direction == 'front':
                direction_phrase = 'move forward'
            elif direction == 'left':
                direction_phrase = 'turn left'
            elif direction == 'backward':
                direction_phrase = 'turn around'
            else:
                assert direction == 'right'
                direction_phrase = f'turn right'

            self.history.append(
                {f'Step {t}, {direction_phrase} towards the scene in the image below and proceed along path to location {action}': selected_img})

        else:
            direction_phrase = "move"
            self.history.append(f'Step {t}, {direction_phrase} towards Location {action}.')


    def make_graph_baseline_prompts(self, instruction, img_to_ghost_node_dict, ghost_pos, t):

        background = """You are an embodied robot that navigates in the real world."""
        background_supp = """You need to explore between some locations marked with IDs and ultimately find the destination to stop.""" \
        + """ At each step, a series of images corresponding to the locations you have explored and have observed will be provided to you."""
        instr_des = """'Instruction' is a global, step-by-step detailed guidance, but you might have already executed some of the commands. You need to carefully discern the commands that have not been executed yet."""
        history = """'History' represents the places you have explored in previous steps along with their corresponding images. It may include the correct landmarks mentioned in the 'Instruction' as well as some past erroneous explorations."""
        pre_planning = """'Previous Planning' records previous long-term multi-step planning info that you can refer to now."""
        option = f"""'Options' are some navigable location IDs with some observed images from front, backward, left, and right views. You need to select one location from the set as your next move. These IDs are also marked in the provided images."""
        requirement = """For each provided image of the environments, you should combine the 'Instruction' and carefully examine the relevant information, such as scene descriptions, landmarks, and objects. You need to align 'Instruction' with 'History' to estimate your instruction execution progress. """
        dist_require = """If you can already see the destination, estimate the distance between you and it. If the distance is far, continue moving and try to stop within 1 meter of the destination."""
        thought = """Your answer should be JSON format and must include three fields: 'Thought', 'New Planning' and 'Action'. You need to combine 'Instruction', your past 'History', 'Options', and the provided images to think about what to do next and why, and complete your thinking into 'Thought'. """

        new_planning = """Based on your 'Previous Planning' and current 'Thought', you also need to update your new multi-step planning to 'New Planning'."""
        action = """Place only the ID of the chosen location in 'Action'. If you think you have arrived at the destination, place 'Stop' into 'Action'."""

        task_description = f"""{background} {background_supp}\n{instr_des}\n{history}\n{pre_planning}\n{option}\n{requirement}\n{dist_require}\n{thought}\n{new_planning}\n{action}"""

        rgb_images = []
        action_node_id = []
        for i, (img_name, node_id) in enumerate(img_to_ghost_node_dict.items()):
            direction = self.convert_img_name_to_direction(img_name)
            action_node_id.extend(node_id)
            node_id_text = ", ".join(node_id)
            img_text = f"""({direction}) Locations {{{node_id_text}}} in Image {i}"""
            rgb_images.append({img_text: img_name})
        action_space_text = ", ".join(action_node_id)

        user_content = []
        if t == 0:
            init_history = 'The navigation has just begun, with no history.'
            prompt = f"""Instruction: {instruction}\nHistory: {init_history}\nPrevious Planning: {self.planning[-1]}\nOptions (step {str(t)}): Locations {{{action_space_text}}}\n"""
            user_content.append({"type": "text", "text": prompt})

        else:
            prompt = f"""Instruction: {instruction}\nPrevious Planning: {self.planning[-1]}\nHistory:\n"""
            user_content.append({"type": "text", "text": prompt})

            history_content = make_image_content(self.history)
            user_content.extend(history_content)

            option_prompt = f"""Options (step {str(t)}): Locations {{{action_space_text}}}\n"""
            user_content.append({"type": "text", "text": option_prompt})


        option_content = make_image_content(rgb_images)
        user_content.extend(option_content)

        messages = [
                    {"role": "system",
                     "content": task_description
                     },
                    {"role": "user",
                     "content": user_content
                     }
                ]

        return messages

    def convert_img_name_to_direction(self, img_name):
        if 'rgb__' in img_name:
            return 'front'
        elif 'rgb_90__' in img_name:
            return 'left'
        elif 'rgb_180__' in img_name:
            return 'backward'
        else:
            assert 'rgb_270__' in img_name
            return 'right'

    def parse_json(self, answer):
        """
        Needed in Gemini, not for GPT
        """
        pattern = r'```json\s*([\s\S]*?)\s*```'
        match = re.search(pattern, answer, re.MULTILINE)
        if match:
            json_str = match.group(1)
            data = json.loads(json_str)
            print('\n', data)
            return data
        else:
            print("JSON not found")
            return None

    def parse_num(self, action):
        if isinstance(action, int):
            selected_id = action
        else:
            numbers = re.findall(r'\d+', action)
            selected_id = int(numbers[0])
        return selected_id

