# from monai.networks.nets import SwinUNETR
# from css.swin_unetr_css_merging import SwinUNETR
from css.swin_unetr_css_merging_skip import SwinUNETR as SwinUNETR
from css.swin_unetr_css_merging_skipV2 import SwinUNETR as SwinUNETR_css_merging

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
    else:
        raise ValueError(f'暂不支持{args.model}~')