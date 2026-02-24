import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

from model.model import DRGradingModelWrapper
from data.dataset import DRDataset, get_train_transform, get_val_transform
from utils.utils import set_seed, BACKBONE_DISPLAY_MAP
from utils.logger import get_logger


class DRTrainer(Trainer):
    def __init__(self, num_classes=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.compute_metrics = self._compute_metrics_wrapper

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(**inputs)
        loss = outputs["loss"]

        if model.training:
            logs = {}
            if outputs["loss_mse"] is not None:
                logs["loss_mse"] = outputs["loss_mse"].detach().item()
            if outputs["loss_odr"] is not None:
                logs["loss_odr"] = outputs["loss_odr"].detach().item()
            self.log(logs)

        return (loss, outputs) if return_outputs else loss

    def _compute_metrics_wrapper(self, eval_pred):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            found_logits = False
            for pred in predictions:
                if len(pred) == len(labels):
                    predictions = pred
                    found_logits = True
                    break
            if not found_logits:
                predictions = predictions[-1]

        pred_scores = np.asarray(predictions).flatten()

        preds_class = np.round(pred_scores)
        preds_class = np.clip(preds_class, 0, self.num_classes - 1).astype(int)

        kappa = cohen_kappa_score(labels, preds_class, weights="quadratic")
        kappa_norm = (kappa + 1.0) / 2.0
        f1_macro = f1_score(labels, preds_class, average="macro")

        return {
            "qwk": kappa,
            "f1_macro": f1_macro,
            "qwk_f1_sum": kappa_norm + f1_macro,
        }


class StripMetricsCallback(TrainerCallback):
    def __init__(self, keys_to_strip=None):
        self.keys_to_strip = set(keys_to_strip or [])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        for key in list(logs.keys()):
            if key in self.keys_to_strip:
                del logs[key]


def main(args):
    backbone_display = BACKBONE_DISPLAY_MAP.get(args.backbone, args.backbone)
    run_name = f"D-ODR({backbone_display})_{args.dataset}_lambda{args.lambda_odr}_bs{args.batch_size}_k{args.kernel_k}_seed{args.seed}"
    output_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    logger = get_logger(
        name=run_name,
        log_dir=args.log_dir,
        console_output=True,
    )

    set_seed(args.seed)

    train_file = os.path.join(args.splits_root, f"{args.dataset}_train.txt")
    val_file = os.path.join(args.splits_root, f"{args.dataset}_crossval.txt")

    train_dataset = DRDataset(
        images_root=args.dataset_root,
        data_file=train_file,
        transform=get_train_transform(args.img_size),
    )

    val_dataset = DRDataset(
        images_root=args.dataset_root,
        data_file=val_file,
        transform=get_val_transform(args.img_size),
    )

    logger.info(
        f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}"
    )

    model = DRGradingModelWrapper(
        backbone=args.backbone,
        use_pretrained=args.use_pretrained,
        img_size=args.img_size,
        lambda_odr=args.lambda_odr,
        kernel_k=args.kernel_k,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.log_dir, args.dataset),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="qwk_f1_sum",
        greater_is_better=True,
        dataloader_num_workers=args.num_workers,
        fp16=(args.dtype == "fp16"),
        bf16=(args.dtype == "bf16"),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        remove_unused_columns=False,
        report_to=["swanlab"],
        run_name=run_name,
    )

    trainer = DRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        num_classes=args.num_classes,
        callbacks=[StripMetricsCallback(keys_to_strip=[])],
    )

    logger.info("Starting Training...")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    best_model_path = os.path.join(output_dir, "best_model")
    trainer.save_model(best_model_path)
    logger.info(f"Best model saved to {best_model_path}")

    eval_results = trainer.evaluate()
    logger.info(f"Final Evaluation Results: {eval_results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DR Grading Model")

    parser.add_argument(
        "--dataset",
        type=str,
        default="APTOS",
        choices=["APTOS", "MESSIDOR", "DEEPDR", "DDR"],
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="GDRBench/images",
    )
    parser.add_argument(
        "--splits_root",
        type=str,
        default="GDRBench/splits",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="vit_base_patch16_224",
        choices=[
            "vit_base_patch16_224",
            "retfound_dinov2_meh",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lambda_odr", type=float, default=1.0)
    parser.add_argument("--kernel_k", type=int, default=25)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--use_pretrained", type=bool, default=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"]
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")

    args = parser.parse_args()
    main(args)
