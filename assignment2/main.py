
import sys
import os
import shutil
import warnings
import json

# Check if CUDA is available
import torch
if not torch.cuda.is_available():
  warnings.warn('CUDA is not available.')

import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from sklearn.metrics import f1_score   
import time

from typing import List, Dict, Union, Optional, Tuple
import torch
import torch.nn as nn


from dataclasses import dataclass
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm


from datasets import Dataset
import transformers
from transformers import AutoTokenizer, AutoModel

from datasets import load_dataset
from tokenizers import Tokenizer

from transformer import Transformer, MultiHeadedAttention
from lstm import EncoderDecoder


dataset_train = load_dataset("yelp_polarity", split="train", cache_dir="assignment/data")
dataset_test = load_dataset("yelp_polarity", split="test[:1000]", cache_dir="assignment/data")

"""### 🔍 Quick look at the data
Lets have quick look at a few samples in our test set.
"""

args = sys.argv[1:]
args_experimental_setting = int(args[0])


n_samples_to_see = 3
# for i in range(n_samples_to_see):
#   print("-"*30)
#   print("title:", dataset_test[i]["text"])
#   print("label:", dataset_test[i]["label"])



tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# # Testing
# input_sample = "Welcome to IFT6135. We now teach you 🤗(HUGGING FACE) Library :DDD."
# tokenizer.tokenize(input_sample)



class Collate:
    def __init__(self, tokenizer: str, max_len: int) -> None:
        self.tokenizer_name = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.max_len = max_len

    def __call__(self, batch: List[Dict[str, Union[str, int]]]) -> Dict[str, torch.Tensor]:
        texts = list(map(lambda batch_instance: batch_instance["text"], batch))
        tokenized_inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        labels = list(map(lambda batch_instance: int(batch_instance["label"]), batch))
        labels = torch.LongTensor(labels)
        return dict(tokenized_inputs, **{"labels": labels})

"""#### 🧑‍🍳 Setting up the collate function"""

tokenizer_name = "bert-base-uncased"
sample_max_length = 256
collate = Collate(tokenizer=tokenizer_name, max_len=sample_max_length)

"""### 4. Models"""
class ReviewClassifier(nn.Module):
    def __init__(self, backbone: str, backbone_hidden_size: int, nb_classes: int):
        super(ReviewClassifier, self).__init__()
        self.backbone = backbone
        self.backbone_hidden_size = backbone_hidden_size
        self.nb_classes = nb_classes
        self.back_bone = AutoModel.from_pretrained(
            self.backbone,
            output_attentions=False,
            output_hidden_states=False,
        )
        self.classifier = torch.nn.Linear(self.backbone_hidden_size, self.nb_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        back_bone_output = self.back_bone(input_ids, attention_mask=attention_mask)
        hidden_states = back_bone_output[0]
        pooled_output = hidden_states[:, 0]  # getting the [CLS] token
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(
                logits.view(-1, self.nb_classes),
                labels.view(-1),
            )
            return loss, logits
        return logits

class ReviewClassifierLSTM(nn.Module):
    def __init__(self, nb_classes: int, encoder_only: bool = False,
        with_attn: bool = True, dropout: int = 0.5, hidden_size: int = 256):
        super(ReviewClassifierLSTM, self).__init__()
        self.nb_classes = nb_classes
        self.encoder_only = encoder_only

        if with_attn:
            attn = MultiHeadedAttention(head_size = 2*hidden_size, num_heads=1)
        else:
            attn = None

        self.back_bone = EncoderDecoder(dropout=dropout, encoder_only=encoder_only,
                                        attn=attn, hidden_size=hidden_size)

        if self.encoder_only:
            self.classifier = torch.nn.Linear(hidden_size*2, self.nb_classes)
        else:
            self.classifier = torch.nn.Linear(hidden_size, self.nb_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pooled_output, _ = self.back_bone(input_ids, attention_mask)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(
                logits.view(-1, self.nb_classes),
                labels.view(-1),
            )
            return loss, logits
        return logits


class ReviewClassifierTransformer(nn.Module):
    def __init__(self, nb_classes: int, num_heads: int = 4, num_layers: int = 4, block: str="prenorm", dropout: float = 0.3):
        super(ReviewClassifierTransformer, self).__init__()
        self.nb_classes = nb_classes
        self.back_bone = Transformer(num_heads=num_heads, num_layers=num_layers, block=block, dropout=dropout)
        self.classifier = torch.nn.Linear(256, self.nb_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attention_mask = torch.cat([torch.ones(attention_mask.shape[0]).unsqueeze(1).to(device),
                                    attention_mask], dim=1)
        back_bone_output = self.back_bone(input_ids, attention_mask)
        hidden_states = back_bone_output
        pooled_output = hidden_states
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(
                logits.view(-1, self.nb_classes),
                labels.view(-1),
            )
            return loss, logits
        return logits

"""### 5. Trainer"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"--> Device selected: {device}")
def train_one_epoch(
    model: torch.nn.Module, training_data_loader: DataLoader, optimizer: torch.optim.Optimizer, logging_frequency: int, testing_data_loader: DataLoader, logger: dict):
    model.train()
    optimizer.zero_grad()
    epoch_loss = 0
    logging_loss = 0
    start_time = time.time()
    mini_start_time = time.time()
    for step, batch in enumerate(training_data_loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs[0]
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        logging_loss += loss.item()

        if (step + 1) % logging_frequency == 0:
            freq_time = time.time()-mini_start_time
            logger['train_time'].append(freq_time+logger['train_time'][-1])
            logger['train_losses'].append(logging_loss/logging_frequency)
            print(f"Training loss @ step {step+1}: {logging_loss/logging_frequency}")
            eval_acc, eval_f1, eval_loss, eval_time = evaluate(model, testing_data_loader)
            logger['eval_accs'].append(eval_acc)
            logger['eval_f1s'].append(eval_f1)
            logger['eval_losses'].append(eval_loss)
            logger['eval_time'].append(eval_time+logger['eval_time'][-1])

            logging_loss = 0
            mini_start_time = time.time()

    return epoch_loss / len(training_data_loader), time.time()-start_time


def evaluate(model: torch.nn.Module, test_data_loader: DataLoader):
    model.eval()
    model.to(device)
    eval_loss = 0
    correct_predictions = {i: 0 for i in range(2)}
    total_predictions = {i: 0 for i in range(2)}
    preds = []
    targets = []
    start_time = time.time()
    with torch.no_grad():
        for step, batch in enumerate(test_data_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            eval_loss += loss.item()

            predictions = np.argmax(outputs[1].detach().cpu().numpy(), axis=1)
            preds.extend(predictions.tolist())
            targets.extend(batch["labels"].cpu().numpy().tolist())

            for target, prediction in zip(batch["labels"].cpu().numpy(), predictions):
                if target == prediction:
                    correct_predictions[target] += 1
                total_predictions[target] += 1
    accuracy = (100.0 * sum(correct_predictions.values())) / sum(total_predictions.values())
    f1 = f1_score(targets, preds)
    model.train()
    return accuracy, round(f1, 4), eval_loss / len(test_data_loader), time.time() - start_time


def save_logs(dictionary, log_dir, exp_id):
  log_dir = os.path.join(log_dir, exp_id)
  os.makedirs(log_dir, exist_ok=True)
  # Log arguments
  with open(os.path.join(log_dir, "args.json"), "w") as f:
    json.dump(dictionary, f, indent=2)

def save_model(model, log_dir, exp_id):
  log_dir = os.path.join(log_dir, exp_id)
  os.makedirs(log_dir, exist_ok=True)
  # Save model
  torch.save(model.state_dict(), f"{log_dir}/model_{exp_id}.pt")


batch_size = 512 # previously 512
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate)



"""### 6. Problem 3
Feel free to modify this code however it is convenient for you to produce a report except for the model parameters.
"""

logging_frequency = 100 # previously 100
learning_rate = 1e-5
nb_epoch=4

#for i in range(8, 9):
experimental_setting = args_experimental_setting

if experimental_setting == 1:
    print("Setting 1: LSTM, no dropout, encoder only")
    model = ReviewClassifierLSTM(nb_classes=2, dropout=0, encoder_only=True)
if experimental_setting == 2:
    print("Setting 2: LSTM, dropout, encoder only")
    model = ReviewClassifierLSTM(nb_classes=2, dropout=0.3, encoder_only=True)
if experimental_setting == 3:
    print("Setting 3: LSTM, dropout, encoder-decoder, no attention")
    model = ReviewClassifierLSTM(nb_classes=2, dropout=0.3, encoder_only=False, with_attn=False)
if experimental_setting == 4:
    print("Setting 4: LSTM, dropout, encoder-decoder, with attention")
    model = ReviewClassifierLSTM(nb_classes=2, dropout=0.3, encoder_only=False, with_attn=True)
if experimental_setting == 5:
    print("Setting 5: Transformer, 2 layers, pre-normalization")
    model = ReviewClassifierTransformer(nb_classes=2, num_heads=4, num_layers=2, block='prenorm', dropout=0.3)
if experimental_setting == 6:
    print("Setting 6: Transformer, 4 layers, pre-normalization")
    model = ReviewClassifierTransformer(nb_classes=2, num_heads=4, num_layers=4, block='prenorm', dropout=0.3)
if experimental_setting == 7:
    print("Setting 7: Transformer, 2 layers, post-normalization")
    model = ReviewClassifierTransformer(nb_classes=2, num_heads=4, num_layers=2, block='postnorm', dropout=0.3)
if experimental_setting == 8:
    #nb_epoch = 2
    print("Setting 8: Fine-tuning BERT")
    model = ReviewClassifier(backbone="bert-base-uncased", backbone_hidden_size=768, nb_classes=2)
for parameter in model.back_bone.parameters():
    parameter.requires_grad= False


# setting up the optimizer
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, eps=1e-8)
model.to(device)
logger = dict()
logger['train_time'] = [0]
logger['eval_time'] = [0]
logger['train_losses'] = []
logger['eval_accs'] = []
logger['eval_f1s'] = []
logger['eval_losses'] = []
logger["epoch_train_loss"] = []
logger["epoch_train_time"] = []
logger["epoch_eval_loss"] = []
logger["epoch_eval_time"] = []
logger["epoch_eval_acc"] = []
logger["epoch_eval_f1"] = []

logger['parameters'] = sum([p.numel() for p in model.back_bone.parameters() if p.requires_grad])

for epoch in range(nb_epoch):
    print(f"Epoch {epoch+1}")
    if experimental_setting == 8 and epoch>1: #unfreezing layer 10 for fine-tuning
        for name, param in model.back_bone.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
    train_loss, train_time = train_one_epoch(model, train_loader, optimizer, logging_frequency, test_loader, logger)
    eval_acc, eval_f1, eval_loss, eval_time  = evaluate(model, test_loader)
    logger["epoch_train_loss"].append(train_loss)
    logger["epoch_train_time"].append(train_time)
    logger["epoch_eval_loss"].append(eval_loss)
    logger["epoch_eval_time"].append(eval_time)
    logger["epoch_eval_acc"].append(eval_acc)
    logger["epoch_eval_f1"].append(eval_f1)
    print(f"    Epoch: {epoch+1} Loss/Test: {eval_loss}, Loss/Train: {train_loss}, Acc/Test: {eval_acc}, F1/Test: {eval_f1}, Train Time: {train_time}, Eval Time: {eval_time}")

logger['train_time'] = logger['train_time'][1:]
logger['eval_time'] = logger['eval_time'][1:]
save_logs(logger, "assignment/log", str(experimental_setting))
save_model(model, "assignment/models", str(experimental_setting))


# """### 7. Augment the original reviews"""

# from textattack.augmentation import Augmenter
# from textattack.transformations import WordSwapQWERTY
# from textattack.transformations import WordSwapExtend
# from textattack.transformations import WordSwapContract
# from textattack.transformations import WordSwapHomoglyphSwap
# from textattack.transformations import CompositeTransformation
# from textattack.transformations import WordSwapRandomCharacterDeletion
# from textattack.transformations import WordSwapNeighboringCharacterSwap
# from textattack.transformations import WordSwapRandomCharacterInsertion
# from textattack.transformations import WordSwapRandomCharacterSubstitution

# # Word-level Augmentations
# word_swap_contract = True
# word_swap_extend = False
# word_swap_homoglyph_swap = False


# # Character-level Augmentations
# word_swap_neighboring_character_swap = True
# word_swap_qwerty = False
# word_swap_random_character_deletion = False
# word_swap_random_character_insertion = False
# word_swap_random_character_substitution = False

# # Check all the augmentations that you wish to apply!

# # NOTE: Try applying each augmentation individually, and observe the changes.

# # Apply augmentations
# augmentations = []
# if word_swap_contract:
#   augmentations.append(WordSwapContract())
# if word_swap_extend:
#   augmentations.append(WordSwapExtend())
# if word_swap_homoglyph_swap:
#   augmentations.append(WordSwapHomoglyphSwap())
# if word_swap_neighboring_character_swap:
#   augmentations.append(WordSwapNeighboringCharacterSwap())
# if word_swap_qwerty:
#   augmentations.append(WordSwapQWERTY())
# if word_swap_random_character_deletion:
#   augmentations.append(WordSwapRandomCharacterDeletion())
# if word_swap_random_character_insertion:
#   augmentations.append(WordSwapRandomCharacterInsertion())
# if word_swap_random_character_substitution:
#   augmentations.append(WordSwapRandomCharacterSubstitution())

# transformation = CompositeTransformation(augmentations)


# augmenter = Augmenter(transformation=transformation,
# pct_words_to_swap=0.5,
# transformations_per_example=1)

# review = "I loved the food and the service was great!"
# augmented_review = augmenter.augment(review)[0]
# print("Augmented review:\n")
# print(augmented_review)



# def getPrediction(text):
#     """
#     Outputs model prediction based on the input text.

#     Args:
#     text: String
#     Input text

#     Returns:
#     item of pred: Iterable
#     Prediction on the input text
#     """
#     inputs = tokenizer(text, padding="max_length", max_length=256,
#     truncation=True, return_tensors="pt",
#     return_token_type_ids=False)
#     for key, value in inputs.items():
#         inputs[key] = value.to(device)


#     outputs = model(**inputs)
#     pred = torch.argmax(outputs, dim=1)
#     return pred.item()