import os

from .utils import init_submodules, save_json, load_json
import importlib
from itertools import chain
from pathlib import Path

from .distributed import get_rank, print0
from .misc import DEFAULT_DIMENSIONS as default_dimensions

import sys
from types import ModuleType

class YOLOX(ModuleType):
    def __getattr__(self, name):
        return getattr(importlib.import_module('king.dimensions.subject_consistency.yolox'), name)

sys.modules['yolox'] = YOLOX('yolox')


class KING(object):
    def __init__(self, device, output_path):
        self.device = device                        # cuda or cpu
        self.output_path = output_path              # output directory to save KING results
        os.makedirs(self.output_path, exist_ok=True)

    def build_full_dimension_list(self, ):
        return default_dimensions
        
    def build_full_info_json(self, videos_path, name, dimension_list, special_str='', verbose=False, **kwargs):
        cur_full_info_list=[] # to save the prompt and video path info for the current dimensions

        if os.path.isfile(videos_path):
            cur_full_info_list = [{ "dimension": dimension_list, "video_list": [videos_path]}]
        else:
            video_names = os.listdir(videos_path)

            cur_full_info_list = []

            for filename in video_names:
                postfix = Path(os.path.join(videos_path, filename)).suffix
                if postfix.lower() not in ['.mp4', '.gif', '.jpg', '.png']:
                    continue
                cur_full_info_list.append({
                    "dimension": dimension_list, 
                    "video_list": [os.path.join(videos_path, filename)]
                })
        
        cur_full_info_path = os.path.join(self.output_path, name+'_full_info.json')
        save_json(cur_full_info_list, cur_full_info_path)
        print0(f'Evaluation meta data saved to {cur_full_info_path}')
        return cur_full_info_path


    def evaluate(self, videos_path, name, dimension_list=None, **kwargs):
        results_dict = {}
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()
        submodules_dict = init_submodules(dimension_list)

        cur_full_info_path = self.build_full_info_json(videos_path, name, dimension_list, **kwargs)
        
        for dimension in dimension_list:
            try:
                dimension_module = importlib.import_module(f'king.dimensions.{dimension}')
                evaluate_func = getattr(dimension_module, f'eval_{dimension}')
            except Exception as e:
                raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}')
            submodules_list = submodules_dict[dimension]
            print0(f'cur_full_info_path: {cur_full_info_path}') # TODO: to delete
            results = evaluate_func(cur_full_info_path, self.device, submodules_list, **kwargs)
            # remove cache results
            results_dict[dimension] = results[:2]
        output_name = os.path.join(self.output_path, name+'_eval_results.json')
        if get_rank() == 0:
            save_json(results_dict, output_name)
            print0(f'Evaluation results saved to {output_name}')
