import os
import json
import torch
import librosa
import soundfile as sf
import torch.nn as nn
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset
from transformers import (
    WhisperModel,
    PreTrainedModel,
    PretrainedConfig
)


# =========================
# Whisper LID Model
# =========================
class WhisperLanguageConfig(PretrainedConfig):
    model_type = "whisper-language-detector"
    def __init__(
        self,
        **kwargs
    ):
        print(kwargs)
        super().__init__(**kwargs)
        self.whisper_model = "openai/whisper-medium"
        self.num_languages = 3

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
# Whisper LID Dataset
# =========================
class LanguageDataset(Dataset):
    def __init__(
        self,
        data_file,
        processor,
        LANG2ID,
        sampling_rate=16000
    ):        
        self.LANG2ID = LANG2ID
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
                label = self.LANG2ID.get(item["language"].lower())
                loaded_languages.append(item["language"].lower())
                self.data_list.append({
                    "input_features":features,
                    "labels":torch.tensor(label, dtype=torch.long)
                })
        print("Language Counts:", Counter(loaded_languages))
                
    def __getitem__(self, idx):
        return self.data_list[idx]        
        
    def __len__(self):
        return len(self.data_list)
    
    @staticmethod
    def resample(sample, orig_sr, target_sr):
        sample = librosa.resample(sample, orig_sr=orig_sr, target_sr=target_sr)
        return sample
    

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

