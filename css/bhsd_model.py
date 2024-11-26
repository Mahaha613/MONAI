# from monai.networks.nets import SwinUNETR
from css.swin_unetr_css_merging import SwinUNETR
import torch

def css_model(model_name, args, device):
    if model_name == 'swin_unetr':
        # ******************************************Create Swin UNETR model*******************************************
        model = SwinUNETR(
            img_size=args.ref_window,
            in_channels=1,
            out_channels=6,
            feature_size=48,
            use_checkpoint=True,
            # use_v2=True,
            merging_type=args.merging_type
        ).to(device)

        weight = torch.load("css/model_swinvit.pt")
        model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights !")
        return model
        # ******************************************Create Swin UNETR model*******************************************
    else:
        raise ValueError(f'暂不支持{model_name}~')