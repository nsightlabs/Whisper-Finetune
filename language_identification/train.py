import argparse
import os
import json
import torch
import librosa
import soundfile as sf
import torch.nn as nn
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset
from torch.distributed import init_process_group, destroy_process_group
from transformers import (
    WhisperProcessor,
    WhisperModel,
    WhisperConfig,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PretrainedConfig
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
# Whisper LID Dataset
# =========================
class LanguageDataset(Dataset):
    def __init__(
        self,
        data_file,
        processor,
        sampling_rate=16000
    ):
        self.data_file = data_file
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.data_list = []
        self._load_data_list()
        
    def _load_data_list(self):
        with open(self.data_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        
        loaded_languages = []  
        for item in tqdm(data, desc=f"Loading data from {os.path.basename(self.data_file)}"):
            if os.path.exists(item['audio']):
                sample, sampling_rate = sf.read(item['audio'], dtype='float32')
                if self.sampling_rate != sampling_rate:
                    sample = self.resample(sample, orig_sr=sampling_rate, target_sr=self.sampling_rate)
                features = self.processor(
                    sample,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt"
                ).input_features[0]
                label = LANG2ID.get(item["language"].lower())
                loaded_languages.append(item["language"].lower())
                self.data_list.append({
                    "input_features":features,
                    "labels":torch.tensor(label, dtype=torch.long)
                })
        print("Language Counts:", Counter(loaded_languages))
    
    
    # def _load_data_list(self):
    #     from concurrent.futures import ProcessPoolExecutor
    #     with open(self.data_file, "r", encoding="utf-8") as f:
    #         data = [json.loads(line) for line in f]
        
    #     tasks = [
    #         (item, self.processor, self.sampling_rate)
    #         for item in data
    #     ]

    #     with ProcessPoolExecutor(max_workers=4) as executor:
    #         results = list(tqdm(executor.map(self.process_item, tasks), total=len(tasks), desc="Processing audio"))
    #         self.data_list = [x for x in results if x is not None]
                
    def __getitem__(self, idx):
        return self.data_list[idx]        
        
    def __len__(self):
        return len(self.data_list)
    
    @staticmethod
    def process_item(args):
        item, processor, sampling_rate = args
        if not os.path.exists(item["audio"]):
            return None

        sample, sr = sf.read(item["audio"], dtype="float32")
        if sr != sampling_rate:
            sample = librosa.resample(sample, orig_sr=sr, target_sr=sampling_rate)

        features = processor(
            sample,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_features[0]
        label = LANG2ID[item["language"].lower()]

        return {
            "input_features": features,
            "labels": torch.tensor(label, dtype=torch.long)
        }
    
    @staticmethod
    def resample(sample, orig_sr, target_sr):
        sample = librosa.resample(sample, orig_sr=orig_sr, target_sr=target_sr)
        return sample


# =========================
# Whisper LID Model
# =========================
class WhisperLanguageConfig(PretrainedConfig):
    model_type = "whisper-language-detector"
    def __init__(
        self,
        whisper_model,
        num_languages,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.whisper_model = whisper_model
        self.num_languages = num_languages

class WhisperLanguageDetector(PreTrainedModel):
    config_class = WhisperLanguageConfig
    def __init__(
        self,
        config
    ):
        super().__init__(config)
        whisper = WhisperModel.from_pretrained(config.whisper_model)
        self.encoder = whisper.encoder
        hidden = config.d_model
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, config.num_languages)
        )

    def forward(
        self,
        input_features,
        labels=None
    ):
        outputs = self.encoder(input_features)
        hidden_states = (
            outputs.last_hidden_state
        )
        embedding = hidden_states.mean(dim=1)
        logits = self.classifier(
            embedding
        )

        loss=None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                logits,
                labels
            )

        return {
            "loss":loss,
            "logits":logits
        }


# =========================
# Data Collator
# =========================
class Collator:
    def __call__(self, batch):
        input_features = torch.stack([x["input_features"] for x in batch])
        labels = torch.stack([x["labels"] for x in batch])

        return {
            "input_features":input_features,
            "labels":labels
        }


# =========================
# Metrics
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(
        axis=-1
    )

    accuracy = (
        preds == labels
    ).mean()

    return {
        "accuracy":accuracy
    }


# =========================
# Train
# =========================
def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

def main():
    if args.ddp:
        ddp_setup()
    processor = WhisperProcessor.from_pretrained(args.base_model)
    train_dataset = LanguageDataset(args.train_data, processor)
    test_dataset = LanguageDataset(args.test_data, processor)
    
    base_config = WhisperConfig.from_pretrained(args.base_model)
    config_dict = base_config.to_dict()
    config_dict.pop("model_type", None)

    config = WhisperLanguageConfig(
        whisper_model=args.base_model,
        num_languages=len(LANG2ID),
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