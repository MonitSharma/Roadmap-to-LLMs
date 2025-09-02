import torch
import torch.nn as nn   
from torch.utils.data import DataLoader, Dataset, random_split
from config import *
from torch.utils.tensorboard import SummaryWriter 
import tqdm
from dataset import BilingualDataset, causal_mask
from model import *
# libraries of hugging face
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


from pathlib import Path

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):

    bos_idx = tokenizer_tgt.token_to_id("[BOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.full((1, 1), bos_idx, dtype=torch.long, device=device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        decoder_output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(decoder_output[:, -1])

        _, next_word = torch.max(prob, dim=1)

        next_word_tensor = torch.full((1, 1), next_word.item(), dtype=torch.long, device=device)
        decoder_input = torch.cat([decoder_input, next_word_tensor], dim=1)
        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)






def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device,print_msg, global_state,writer, num_examples=2):
    model.eval()

    count = 0
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count +=1 
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Validation batch size should be 1"

            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            print_msg("-"*console_width)
            print_msg(f"Source: {source_text}")
            print_msg(f"Expected: {target_text}")
            print_msg(f"Predicted: {model_out_text}")


            if count == num_examples:
                break



def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"])
        tokenizer.train_from_iterator((item["translation"][config[lang]] for item in ds["train"]), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}')

    # build tokenizer using the full dataset dict
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, "lang_src")
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, "lang_tgt")

    # Use HuggingFace splits for train/validation
    # Check for validation split, otherwise split train
    if "validation" in ds_raw:
        train_split = ds_raw["train"]
        val_split = ds_raw["validation"]
    else:
        # Create validation split from train
        train_val = ds_raw["train"].train_test_split(test_size=0.1, seed=42)
        train_split = train_val["train"]
        val_split = train_val["test"]

    train_ds = BilingualDataset(train_split, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_split, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])

    # Calculate max length in training split only
    max_len_src = 0
    max_len_tgt = 0
    for item in train_split:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length in source language ({config['lang_src']}): {max_len_src}")
    print(f"Max length in target language ({config['lang_tgt']}): {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config['d_model'],
        config['num_heads'],
        config['num_encoder_layers'],
        config['num_decoder_layers'],
        config['d_ff'],
        config['dropout'] if 'dropout' in config else 0.1,
        config['seq_len']
    )
    return model

def train_model(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, len(tokenizer_src.get_vocab()), len(tokenizer_tgt.get_vocab())).to(device)

    writer = SummaryWriter(log_dir=f"runs/{config['experiment_name']}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config["preload"] is not None:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename, map_location=device)
        initial_epoch = state["epoch"] + 1 
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state["global_step"]


    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1)


    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", unit="batch")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)

            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix(loss=loss.item())
            writer.add_scalar("Training Loss", loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # Run validation once per epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config["seq_len"], device, print, global_step, writer)

        # save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            "epoch": epoch,
            "model_state_dict":  model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        }, model_filename)

    writer.close()

if __name__ == "__main__":
    config = get_config()
    train_model(config)