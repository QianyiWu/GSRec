#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.feat_dim = 32
        self.n_offsets = 10
        self.voxel_size =  0.001 # if voxel_size<=0, using 1nn dist
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4
        
        self.implicit_sdf_divide_factor = 1.0
        self.sdf_inside_out = False
        
        # params for implicit sdf network

        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self.start_model = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        # self.eval = True
        self.eval = False
        self.lod = 0
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        
        self.offset_lr_init = 0.01
        # self.offset_lr_final = 0.0005
        self.offset_lr_final = 0.001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.normal_lr_init = 0.01
        self.normal_lr_final = 0.001
        self.normal_lr_delay_mult = 0.01
        self.normal_lr_max_steps = 30_000
        
        self.feature_lr = 0.0075
        # self.feature_lr = 0.0003
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        # self.scaling_lr = 0.002
        self.rotation_lr = 0.002
        self.ratio_grid_mlp = 10
        
        
        
        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        # self.mlp_cov_lr_init = 0.002
        # self.mlp_cov_lr_final = 0.0004
        self.mlp_cov_lr_final = 0.004
        # self.mlp_cov_lr_final = 0.00004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000

        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000
        
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000
        
        self.mlp_sdf_lr_init = 0.0005
        self.mlp_sdf_lr_final = 0.00005
        self.mlp_sdf_lr_delay_mult = 0.5
        self.mlp_sdf_lr_max_steps = 15_000
        self.sdf_start_iter = 15_000



        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_depth = 0.1
        self.lambda_normal_l1 = 0.1
        self.lambda_normal_cos = 0.1
        self.lambda_norm_reg = 0.1
        self.lambda_normal_consistency = 0.2
        self.lambda_eki = 0.1
        self.lambda_render_norm_reg = 0.2
        self.lambda_depth_sdf = 10
        
        self.use_mask_for_normal = False
        self.use_mask_for_rgb = False
        
        self.fmls_sdf_offset = 0.01
        self.fmls_use_normal = False
        self.fmls_normal_weight = 0.05
        
        # for anchor densification
        self.start_stat = 500
        self.update_from = 1500
        self.update_interval = 100
        self.update_until = 15_000
        self.mesh_iter = 10_000
        
        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
