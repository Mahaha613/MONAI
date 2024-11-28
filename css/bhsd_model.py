# from monai.networks.nets import SwinUNETR
from css.swin_unetr_css_merging import SwinUNETR
import torch
import os
import json
def css_model(args):
    if args.model == 'swin_unetr':
        # ******************************************Create Swin UNETR model*******************************************
        model = SwinUNETR(
            img_size=args.ref_window,
            in_channels=1,
            out_channels=6,
            feature_size=48,
            use_checkpoint=True,
            # use_v2=True,
            merging_type=args.merging_type
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
        # ******************************************Create Swin UNETR model*******************************************
    else:
        raise ValueError(f'暂不支持{args.model}~')