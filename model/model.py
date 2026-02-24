import os
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import timm

from .losses import DirectedODRLoss


def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" not in checkpoint_model:
        return

    pos_embed_checkpoint = checkpoint_model["pos_embed"]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches

    new_size = int(model.patch_embed.num_patches**0.5)
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)

    if orig_size == new_size:
        return

    print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")

    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    patch_tokens = pos_embed_checkpoint[:, num_extra_tokens:]

    patch_tokens = patch_tokens.reshape(
        -1, orig_size, orig_size, embedding_size
    ).permute(0, 3, 1, 2)
    patch_tokens = torch.nn.functional.interpolate(
        patch_tokens,
        size=(new_size, new_size),
        mode="bicubic",
        align_corners=False,
    )
    patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(
        -1, new_size * new_size, embedding_size
    )

    checkpoint_model["pos_embed"] = torch.cat((extra_tokens, patch_tokens), dim=1)


def convert_google_vit_weights(state_dict):
    converted_dict = {}
    key_mapping = {
        "vit.embeddings.cls_token": "embeddings.cls_token",
        "vit.embeddings.position_embeddings": "embeddings.position_embeddings",
        "vit.embeddings.patch_embeddings.projection.weight": "embeddings.patch_embeddings.projection.weight",
        "vit.embeddings.patch_embeddings.projection.bias": "embeddings.patch_embeddings.projection.bias",
        "vit.layernorm.weight": "layernorm.weight",
        "vit.layernorm.bias": "layernorm.bias",
        "classifier.weight": None,
        "classifier.bias": None,
    }

    for i in range(12):
        key_mapping.update(
            {
                f"vit.encoder.layer.{i}.attention.attention.query.weight": f"encoder.layer.{i}.attention.attention.query.weight",
                f"vit.encoder.layer.{i}.attention.attention.query.bias": f"encoder.layer.{i}.attention.attention.query.bias",
                f"vit.encoder.layer.{i}.attention.attention.key.weight": f"encoder.layer.{i}.attention.attention.key.weight",
                f"vit.encoder.layer.{i}.attention.attention.key.bias": f"encoder.layer.{i}.attention.attention.key.bias",
                f"vit.encoder.layer.{i}.attention.attention.value.weight": f"encoder.layer.{i}.attention.attention.value.weight",
                f"vit.encoder.layer.{i}.attention.attention.value.bias": f"encoder.layer.{i}.attention.attention.value.bias",
                f"vit.encoder.layer.{i}.attention.output.dense.weight": f"encoder.layer.{i}.attention.output.dense.weight",
                f"vit.encoder.layer.{i}.attention.output.dense.bias": f"encoder.layer.{i}.attention.output.dense.bias",
                f"vit.encoder.layer.{i}.intermediate.dense.weight": f"encoder.layer.{i}.intermediate.dense.weight",
                f"vit.encoder.layer.{i}.intermediate.dense.bias": f"encoder.layer.{i}.intermediate.dense.bias",
                f"vit.encoder.layer.{i}.output.dense.weight": f"encoder.layer.{i}.output.dense.weight",
                f"vit.encoder.layer.{i}.output.dense.bias": f"encoder.layer.{i}.output.dense.bias",
                f"vit.encoder.layer.{i}.layernorm_before.weight": f"encoder.layer.{i}.layernorm_before.weight",
                f"vit.encoder.layer.{i}.layernorm_before.bias": f"encoder.layer.{i}.layernorm_before.bias",
                f"vit.encoder.layer.{i}.layernorm_after.weight": f"encoder.layer.{i}.layernorm_after.weight",
                f"vit.encoder.layer.{i}.layernorm_after.bias": f"encoder.layer.{i}.layernorm_after.bias",
            }
        )

    for old_key, value in state_dict.items():
        if old_key in key_mapping:
            new_key = key_mapping[old_key]
            if new_key is not None:
                converted_dict[new_key] = value
        else:
            converted_dict[old_key] = value

    return converted_dict


class BackboneRegressor(nn.Module):
    def __init__(
        self,
        backbone="vit_base_patch16_224",
        use_pretrained=True,
        img_size=224,
    ):
        super().__init__()

        self.backbone_name = backbone
        self.using_pretrained = False
        self.use_pretrained = use_pretrained

        self.pretrained_paths = {
            "vit_base_patch16_224": "google/vit-base-patch16-224/pytorch_model.bin",
            "retfound_dinov2_meh": "YukunZhou/RETFound_dinov2_meh/RETFound_dinov2_meh.pth",
        }

        if backbone == "vit_base_patch16_224":
            self.backbone = self._create_vit_backbone(img_size)
            self.feature_dim = 768
        elif backbone == "retfound_dinov2_meh":
            self.backbone = self._create_retfound_backbone(img_size)
            self.feature_dim = 1024
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.head = nn.Sequential(nn.Linear(self.feature_dim, 1))

        if backbone == "retfound_dinov2_meh":
            for param in self.backbone.parameters():
                param.requires_grad = False
            num_layers_to_unfreeze = 2
            layers = getattr(self.backbone, "blocks", None)
            if isinstance(layers, nn.ModuleList):
                for i in range(len(layers) - num_layers_to_unfreeze, len(layers)):
                    for param in layers[i].parameters():
                        param.requires_grad = True
                print(f"Unfrozen the last {num_layers_to_unfreeze} layers of RETFound.")

            norm_layer = getattr(self.backbone, "norm", None)
            if isinstance(norm_layer, nn.Module):
                for param in norm_layer.parameters():
                    param.requires_grad = True

            for param in self.head.parameters():
                param.requires_grad = True
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"Model initialized: {backbone} | Pretrained: {self.using_pretrained}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

    def _create_vit_backbone(self, img_size):
        config = ViTConfig(
            image_size=img_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            patch_size=16,
        )
        backbone = ViTModel(config, add_pooling_layer=False)

        if self.use_pretrained:
            vit_path = self.pretrained_paths.get("vit_base_patch16_224")
            if vit_path is None:
                raise RuntimeError("ViT pretrained path not defined")

            if not os.path.exists(vit_path):
                raise FileNotFoundError(vit_path)

            print(f"Loading pretrained ViT weights from {vit_path}")
            checkpoint = torch.load(vit_path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)
            converted_state_dict = convert_google_vit_weights(state_dict)

            msg = backbone.load_state_dict(converted_state_dict, strict=False)
            self.using_pretrained = True
            print(f"Loaded pretrained ViT weights: {msg}")
        else:
            print("use_pretrained=False, ViT uses random initialization")

        return backbone

    def _create_retfound_backbone(self, img_size):
        model = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=0,
            img_size=img_size,
            dynamic_img_size=True,
        )

        if self.use_pretrained:
            ret_path = self.pretrained_paths.get("retfound_dinov2_meh")
            if ret_path is None:
                raise RuntimeError("RETFound pretrained path not defined")

            if not os.path.exists(ret_path):
                raise FileNotFoundError(ret_path)

            print(f"Loading RETFound weights from {ret_path}")
            checkpoint = torch.load(ret_path, map_location="cpu")
            checkpoint_model = checkpoint.get("teacher", checkpoint)

            checkpoint_model = {
                k.replace("backbone.", ""): v for k, v in checkpoint_model.items()
            }
            checkpoint_model = {
                k.replace("mlp.w12.", "mlp.fc1."): v
                for k, v in checkpoint_model.items()
            }
            checkpoint_model = {
                k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()
            }

            state_dict = model.state_dict()
            for k in ["head.weight", "head.bias"]:
                if (
                    k in checkpoint_model
                    and checkpoint_model[k].shape != state_dict[k].shape
                ):
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            interpolate_pos_embed(model, checkpoint_model)

            msg = model.load_state_dict(checkpoint_model, strict=False)
            if hasattr(model, "head") and isinstance(model.head, torch.nn.Linear):
                torch.nn.init.trunc_normal_(model.head.weight, std=1e-3)
                torch.nn.init.constant_(model.head.bias, 1.5)
            self.using_pretrained = True
            print(f"Loaded RETFound pretrained weights: {msg}")
        else:
            print("use_pretrained=False, RETFound uses random initialization")

        return model

    def forward(self, x):
        if self.backbone_name == "vit_base_patch16_224":
            outputs = self.backbone(x)
            features = outputs.last_hidden_state[:, 0, :]
        elif self.backbone_name == "retfound_dinov2_meh":
            features = self.backbone.forward_features(x)[:, 0, :]
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        score = self.head(features)
        return {"score": score, "features": features}


class DRGradingModelWrapper(nn.Module):
    def __init__(
        self,
        backbone="vit_base_patch16_224",
        use_pretrained=True,
        img_size=224,
        lambda_odr=0.5,
        kernel_k=10,
    ):
        super().__init__()

        self.model = BackboneRegressor(
            backbone=backbone,
            use_pretrained=use_pretrained,
            img_size=img_size,
        )

        self.mse_loss = nn.MSELoss()
        self.odr_loss_fn = DirectedODRLoss(
            scales=(1, 2, 3), kernel_k=kernel_k
        )
        self.lambda_odr = lambda_odr

    def forward(self, pixel_values, labels=None):
        outputs = self.model(pixel_values)
        score = outputs["score"]
        features = outputs["features"]

        loss = None
        l_ord = None
        l_odr = None

        if labels is not None:
            labels_float = labels.float().view(-1, 1)
            l_ord = self.mse_loss(score, labels_float)
            l_odr = self.odr_loss_fn(features, score, labels)
            loss = l_ord + self.lambda_odr * l_odr

        return {"loss": loss, "loss_mse": l_ord, "loss_odr": l_odr, "logits": score}
