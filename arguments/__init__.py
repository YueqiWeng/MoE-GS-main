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

        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.lod = 0

        self.appearance_dim = 32
        self.lowpoly = False
        self.ds = 1
        self.ratio = 1 # sampling the input point cloud
        self.undistorted = False 
        
        # In the Bungeenerf dataset, we propose to set the following three parameters to True,
        # Because there are enough dist variations.
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False

        # MoE-GS (Scene Experts). num_experts=1 keeps original Scaffold-GS.
        # train_method.sh --method moe defaults to 8 experts + Top-2.
        self.num_experts = 1
        self.moe_top_k = 2
        # Number of Linear layers in each attribute MLP / MoE expert (Scaffold default=2).
        self.mlp_depth = 2

        # MoE architecture:
        #   flat          — flat AttributeMoE
        #   dynamic_tree  — growing TreeMoE (+GNN via --use_gnn)
        self.moe_arch = "flat"
        self.moe_branch_factor = 8
        self.moe_tree_depth = 3
        self.moe_max_leaves = 14
        # grow_factor: children spawned per split (0 => same as branch_factor).
        self.moe_grow_factor = 2
        self.moe_max_total_grows = 4

        # MoE+GNN dual-router: AnchorGNN spatial context → GNN router logits
        # fused as z = z_base + γ * z_gnn (experts still use raw fv).
        self.use_gnn = False
        self.gnn_hidden_dim = 32
        self.gnn_layers = 2
        self.gnn_k = 16
        self.gnn_gamma_init = 0.5

        # Optional warm-start: path to baseline / flat-MoE checkpoints.pth
        self.start_baseline_mlp = ""
        # Optional explicit geometry ply (default unused; Dynamic densifies from SfM)
        self.start_gaussian_ply = ""

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
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002
        
        
        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000
        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = 30_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        
        # for anchor densification
        self.start_stat = 500
        self.update_from = 1500
        self.update_interval = 100
        self.update_until = 15_000
        
        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002

        # MoE-GS losses / curriculum (paper defaults)
        self.moe_warmup = 1500
        self.lambda_opacity = 0.01
        self.lambda_load = 0.01
        # Mild concentration after sparse warmup (0 = off). Keep small to avoid expert collapse.
        self.lambda_router_entropy = 0.002
        self.opacity_reg_tau = 0.0

        self.gnn_lr_init = 0.0075
        self.gnn_lr_final = 0.0001
        self.gnn_lr_delay_mult = 0.01
        self.gnn_lr_max_steps = 30_000

        # Dynamic v1/v2 growth (fullwin): start 4, selective grow, freeze @20k.
        # hetero_easy=hard=1.0 disables scene budget (always allow up to max grows).
        self.tree_grow_start = 6000
        self.tree_grow_interval = 4000
        self.tree_grow_stop = 20000
        self.tree_grow_max_per_event = 1
        self.tree_grow_max_heads_per_event = 2
        self.tree_grow_min_score_ratio = 1.08
        self.tree_grow_min_neff = 1.1
        self.tree_grow_min_mass_frac = 0.12
        self.tree_grow_min_loss_ratio = 0.95
        self.tree_grow_hetero_easy = 1.0
        self.tree_grow_hetero_hard = 1.0
        self.tree_grow_require_feat_var_median = False
        self.tree_grow_force = False
        self.tree_grow_dense_iters = 1500

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
