import argparse
import os
import json
import torch
from .utils import (
    LanguageDataset,
    Collator,
    compute_metrics,
    WhisperLanguageConfig,
    WhisperLanguageDetector   
)
from torch.distributed import init_process_group, destroy_process_group
from transformers import (
    WhisperProcessor,
    WhisperConfig,
    Trainer,
    TrainingArguments
)

parser = argparse.ArgumentParser()

parser.add_argument("--train_data", type=str, default="dataset/train.jsonl")
parser.add_argument("--test_data", type=str, default="dataset/test.jsonl")
parser.add_argument("--base_model", type=str, default="openai/whisper-large-v3")
parser.add_argument("--output_dir", type=str, default="whisper-language-detector")
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--eval_steps", type=int, default=100)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--fp16", type=bool, default=True)
parser.add_argument("--run_name", type=str, default=None, help="WandB run name, if None, will use default naming")
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--eval_batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--ddp", action="store_true", help="Use Distributed Data Parallel (DDP) for training")


args = parser.parse_args()

with open(args.train_data, "r", encoding="utf-8") as f:
    train_data = [json.loads(line) for line in f]
    languages = list(set([item["language"].lower() for item in train_data]))
    
LANG2ID = {lang: idx for idx, lang in enumerate(languages)}
ID2LANG = {idx: lang for lang, idx in LANG2ID.items()}


# =========================
# Train
# =========================
def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

def main():
    if args.ddp:
        ddp_setup()
        
    with open(args.train_data, "r", encoding="utf-8") as f:
            train_data = [json.loads(line) for line in f]
            languages = list(set([item["language"].lower() for item in train_data]))
            
    LANG2ID = {lang: idx for idx, lang in enumerate(languages)}
    ID2LANG = {idx: lang for lang, idx in LANG2ID.items()}
        
    processor = WhisperProcessor.from_pretrained(args.base_model)
    train_dataset = LanguageDataset(args.train_data, processor)
    test_dataset = LanguageDataset(args.test_data, processor)
    
    base_config = WhisperConfig.from_pretrained(args.base_model)
    config_dict = base_config.to_dict()
    config_dict.pop("model_type", None)

    config = WhisperLanguageConfig(
        **config_dict
    )
    config.architectures = ["WhisperLanguageDetector"]
    model = WhisperLanguageDetector(config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        dataloader_num_workers=args.num_workers,
        learning_rate=args.lr,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=args.eval_steps,
        optim="adamw_torch",
        fp16=args.fp16,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=("wandb" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Batch" else "none"),
        run_name=(args.run_name if args.run_name is not None else None)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=Collator(),
        compute_metrics=compute_metrics
    )


    trainer.train()
    
    if not args.ddp or (training_args.local_rank == 0 or training_args.local_rank == -1):
        model.save_pretrained(os.path.join(args.output_dir, "checkpoint-final"))
    
    if args.ddp:   
        destroy_process_group()


if __name__ == "__main__":

    main()