import os

from .utils import init_submodules, save_json, load_json
import importlib
from itertools import chain
from pathlib import Path

from .distributed import get_rank, print0


class KING(object):
    def __init__(self, device, output_path):
        self.device = device                        # cuda or cpu
        self.output_path = output_path              # output directory to save KING results
        os.makedirs(self.output_path, exist_ok=True)

    def build_full_metrics_list(self, ):
        #return ["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality", "object_class", "multiple_objects", "color", "spatial_relationship", "scene", "temporal_style", 'overall_consistency', "human_action", "temporal_flickering", "motion_smoothness", "dynamic_degree", "appearance_style"]        
        return ['subject_consistency']
        
    def build_full_info_json(self, videos_path, name, dimension_list, prompt_list=[], special_str='', verbose=False, **kwargs):
        cur_full_info_list=[] # to save the prompt and video path info for the current dimensions

        if os.path.isfile(videos_path):
            cur_full_info_list = [{"prompt_en": get_prompt_from_filename(videos_path), "dimension": dimension_list, "video_list": [videos_path]}]
            if len(prompt_list) == 1:
                cur_full_info_list[0]["prompt_en"] = prompt_list[0]
        else:
            video_names = os.listdir(videos_path)

            cur_full_info_list = []

            for filename in video_names:
                postfix = Path(os.path.join(videos_path, filename)).suffix
                if postfix.lower() not in ['.mp4', '.gif', '.jpg', '.png']:
                    continue
                cur_full_info_list.append({
                    "prompt_en": get_prompt_from_filename(filename), 
                    "dimension": dimension_list, 
                    "video_list": [os.path.join(videos_path, filename)]
                })

            if len(prompt_list) > 0:
                prompt_list = {os.path.join(videos_path, path): prompt_list[path] for path in prompt_list}
                assert len(prompt_list) >= len(cur_full_info_list), """
                    Number of prompts should match with number of videos.\n
                    Got {len(prompt_list)=}, {len(cur_full_info_list)=}\n
                    To read the prompt from filename, delete --prompt_file and --prompt_list
                    """

                all_video_path = [os.path.abspath(file) for file in list(chain.from_iterable(vid["video_list"] for vid in cur_full_info_list))]
                backslash = "\n"
                assert len(set(all_video_path) - set([os.path.abspath(path_key) for path_key in prompt_list])) == 0, f"""
                The prompts for the following videos are not found in the prompt file: \n
                {backslash.join(set(all_video_path) - set([os.path.abspath(path_key) for path_key in prompt_list]))}
                """

                video_map = {}
                for prompt_key in prompt_list:
                    video_map[os.path.abspath(prompt_key)] = prompt_list[prompt_key]

                for video_info in cur_full_info_list:
                    video_info["prompt_en"] = video_map[os.path.abspath(video_info["video_list"][0])]
        
        cur_full_info_path = os.path.join(self.output_path, name+'_full_info.json')
        save_json(cur_full_info_list, cur_full_info_path)
        print0(f'Evaluation meta data saved to {cur_full_info_path}')
        return cur_full_info_path


    def evaluate(self, videos_path, name, dimension_list=None, **kwargs):
        results_dict = {}
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()
        submodules_dict = init_submodules(dimension_list, local=local, read_frame=read_frame)

        cur_full_info_path = self.build_full_info_json(videos_path, name, dimension_list, prompt_list, mode=mode, **kwargs)
        
        for dimension in dimension_list:
            try:
                dimension_module = importlib.import_module(f'king.{dimension}')
                evaluate_func = getattr(dimension_module, f'eval_{dimension}')
            except Exception as e:
                raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}')
            submodules_list = submodules_dict[dimension]
            print0(f'cur_full_info_path: {cur_full_info_path}') # TODO: to delete
            results = evaluate_func(cur_full_info_path, self.device, submodules_list, **kwargs)
            results_dict[dimension] = results
        output_name = os.path.join(self.output_path, name+'_eval_results.json')
        if get_rank() == 0:
            save_json(results_dict, output_name)
            print0(f'Evaluation results saved to {output_name}')
