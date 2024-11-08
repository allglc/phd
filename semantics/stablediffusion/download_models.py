import open_clip

arch="ViT-H-14"
version="laion2b_s32b_b79k"
open_clip.create_model_and_transforms(arch, pretrained=version)