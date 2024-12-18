# from monai.networks.nets import SwinUNETR
# from css.swin_unetr_css_merging import SwinUNETR
from css.swin_unetr_css_merging_skip import SwinUNETR as SwinUNETR
from css.swin_unetr_css_merging_skipV2 import SwinUNETR as SwinUNETR_css_merging
from css.swin_unetr_css_merging_skipV3 import SwinUNETR as SwinUNETR_css_MultiScaleMerging
from css.swin_unetr_css_merging_skipV3_AddEncForX3 import SwinUNETR as SwinUNETR_css_merging_skipV3_AddEncForX3
from css.swin_unetr_css_merging_skipV3_maxpoolskip import SwinUNETR as SwinUNETR_css_merging_skipV3_maxpoolskip

import torch
import os
import json
def css_model(args):
    if args.model == 'swin_unetr':
        # ******************************************Create Swin UNETR model*******************************************
        print("Using swin_unetr!")
        model = SwinUNETR(
            img_size=args.ref_window,
            in_channels=1,
            out_channels=6,
            feature_size=48,
            use_checkpoint=True,
            # use_v2=True,
            use_ln = args.use_ln,
            merging_type=args.merging_type,
            css_skip=args.css_skip,
            use_1x1_conv_for_skip=args.use_1x1_conv_for_skip,
            use_css_skip_m4 = args.use_css_skip_m4,
            use_css_skip_m1V2 = args.use_css_skip_m1V2
        ).to(args.device)
        if args.test:
            assert os.path.isfile(args.ref_weight), "weight path is not a file"
            print(f"weight_path: {args.ref_weight}")
            model.load_state_dict(torch.load(args.ref_weight))
            return model
        weight = torch.load("css/model_swinvit.pt")
        model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights !")
        return model
    
    elif args.model == 'swin_unetr_css_merging':
        print("Using swin_unetr_css_merging!")
        # ******************************************Create Swin UNETR model*******************************************
        model = SwinUNETR_css_merging(
            img_size=args.ref_window,
            in_channels=1,
            out_channels=6,
            feature_size=48,
            use_checkpoint=True,
            use_ln = args.use_ln,
            # use_v2=True,
            merging_type=args.merging_type,
            css_skip=args.css_skip,
            use_1x1_conv_for_skip=args.use_1x1_conv_for_skip,
            use_css_skip_m4 = args.use_css_skip_m4,
            use_css_skip_m1V2 = args.use_css_skip_m1V2
        ).to(args.device)
        if args.test:
            assert os.path.isfile(args.ref_weight), "weight path is not a file"
            print(f"weight_path: {args.ref_weight}")
            model.load_state_dict(torch.load(args.ref_weight))
            return model
        weight = torch.load("css/model_swinvit.pt")
        model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights !")
        return model
    
    elif args.model == 'swin_unetr_css_MultiScaleMerging':
        print("Using swin_unetr_css_MultiScaleMerging!")
        # ******************************************Create Swin UNETR model*******************************************
        model = SwinUNETR_css_MultiScaleMerging(
            img_size=args.ref_window,
            in_channels=1,
            out_channels=6,
            feature_size=48,
            use_checkpoint=True,
            use_ln = args.use_ln,
            # use_v2=True,
            merging_type=args.merging_type,
            css_skip=args.css_skip,
            use_1x1_conv_for_skip=args.use_1x1_conv_for_skip,
            use_css_skip_m4 = args.use_css_skip_m4,
            use_css_skip_m1V2 = args.use_css_skip_m1V2
        ).to(args.device)
        if args.test:
            assert os.path.isfile(args.ref_weight), "weight path is not a file"
            print(f"weight_path: {args.ref_weight}")
            model.load_state_dict(torch.load(args.ref_weight))
            return model
        weight = torch.load("css/model_swinvit.pt")
        model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights !")
        return model
    
    elif args.model == 'SwinUNETR_css_merging_skipV3_AddEncForX3':
        print("Using SwinUNETR_css_merging_skipV3_AddEncForX3!")
        # ******************************************Create Swin UNETR model*******************************************
        model = SwinUNETR_css_merging_skipV3_AddEncForX3(
            img_size=args.ref_window,
            in_channels=1,
            out_channels=6,
            feature_size=48,
            use_checkpoint=True,
            use_ln = args.use_ln,
            # use_v2=True,
            merging_type=args.merging_type,
            css_skip=args.css_skip,
            use_1x1_conv_for_skip=args.use_1x1_conv_for_skip,
            use_css_skip_m4 = args.use_css_skip_m4,
            use_css_skip_m1V2 = args.use_css_skip_m1V2
        ).to(args.device)
        if args.test:
            assert os.path.isfile(args.ref_weight), "weight path is not a file"
            print(f"weight_path: {args.ref_weight}")
            model.load_state_dict(torch.load(args.ref_weight))
            return model
        weight = torch.load("css/model_swinvit.pt")
        model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights !")
        return model
    
    elif args.model == 'SwinUNETR_css_merging_skipV3_maxpoolskip':
        print("Using SwinUNETR_css_merging_skipV3_maxpoolskip!")
        # ******************************************Create Swin UNETR model*******************************************
        model = SwinUNETR_css_merging_skipV3_maxpoolskip(
            img_size=args.ref_window,
            in_channels=1,
            out_channels=6,
            feature_size=48,
            use_checkpoint=True,
            use_ln = args.use_ln,
            # use_v2=True,
            merging_type=args.merging_type,
            css_skip=args.css_skip,
            use_1x1_conv_for_skip=args.use_1x1_conv_for_skip,
            use_dec_change_C_in_css_skip=args.use_dec_change_C_in_css_skip,
            use_css_skip_m4 = args.use_css_skip_m4,
            use_css_skip_m1V2 = args.use_css_skip_m1V2
        ).to(args.device)
        if args.test:
            assert os.path.isfile(args.ref_weight), "weight path is not a file"
            print(f"weight_path: {args.ref_weight}")
            model.load_state_dict(torch.load(args.ref_weight))
            return model
        weight = torch.load("css/model_swinvit.pt")
        model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights !")
        return model
    
    else:
        raise ValueError(f'暂不支持{args.model}~')