'''
Train and test bidirectional language models.
'''

import os
import time
import json
import re
import glob
import csv
import logging
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

from .data import (
    Vocabulary,
    UnicodeCharsVocabulary,
    InvalidNumberOfCharacters,
    BidirectionalLMDataset,
)
from .model import BidirectionalLanguageModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DTYPE = torch.float32
DTYPE_INT = torch.int64


def _get_activation_fn(name):
    name = (name or "relu").lower()
    if name == "relu":
        return F.relu
    if name == "tanh":
        return torch.tanh
    if name == "gelu":
        return F.gelu
    logger.warning("Unknown char_cnn.activation=%s, fallback to relu", name)
    return F.relu


def print_variable_summary(model):
    '''
    Print model parameter summary
    '''
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        logger.info(f"{name}: {list(param.shape)} ({param_count} parameters)")
    logger.info(f"Total parameters: {total_params}")


class LanguageModelDataset(Dataset):
    '''
    PyTorch Dataset for language model training
    '''
    def __init__(self, data_generator, max_samples=None):
        self.data_generator = data_generator
        self.max_samples = max_samples
        self.samples = []
        
        # Pre-load samples
        count = 0
        for sample in data_generator:
            self.samples.append(sample)
            count += 1
            if max_samples and count >= max_samples:
                break
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class LanguageModelTrainer:
    '''
    A class to train bidirectional language models in PyTorch
    
    All hyperparameters and model configuration is specified in a dictionary
    of 'options'.
    '''
    
    def __init__(
        self,
        options,
        train_prefix,
        vocab_file,
        test_prefix=None,
        valid_prefix=None,
    ):
        '''
        Initialize the language model trainer
        
        Args:
            options: dictionary containing model hyperparameters
            train_prefix: prefix for training data files
            vocab_file: path to vocabulary file
            test_prefix: prefix for test data files (optional)
            valid_prefix: glob for validation shards (optional; for val loss / perplexity)
        '''
        self.options = options
        self.train_prefix = train_prefix
        self.vocab_file = vocab_file
        self.test_prefix = test_prefix
        self.valid_prefix = valid_prefix
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize vocabulary
        self._init_vocabulary()
        
        # Build model
        self._build_model()
        
        # Initialize optimizer
        self._init_optimizer()
        
        # Initialize data loaders
        self._init_data_loaders()
    
    def _init_vocabulary(self):
        '''
        Initialize vocabulary
        '''
        if self.options.get('char_cnn'):
            self.vocab = UnicodeCharsVocabulary(
                self.vocab_file, 
                self.options['char_cnn']['max_characters_per_token']
            )
        else:
            self.vocab = Vocabulary(self.vocab_file, validate_file=True)
    
    def _build_model(self):
        '''
        Build the language model
        '''
        # Create a temporary options file for model initialization
        options_file = 'temp_options.json'
        with open(options_file, 'w') as f:
            json.dump(self.options, f)
        
        # For training, we don't load pre-trained weights
        # Create empty weight file
        import h5py
        weight_file = 'temp_weights.hdf5'
        with h5py.File(weight_file, 'w') as f:
            pass  # Empty file
        
        try:
            self.model = BidirectionalLanguageModel(
                options_file=options_file,
                weight_file=weight_file,
                use_character_inputs=self.options.get('char_cnn') is not None
            )
        except Exception as exc:
            # If loading fails, create model from scratch
            logger.warning(
                "Falling back to SimpleLanguageModel due to model init failure: %s",
                exc,
            )
            self.model = self._create_model_from_scratch()
        
        # Clean up temporary files
        if os.path.exists(options_file):
            os.remove(options_file)
        if os.path.exists(weight_file):
            os.remove(weight_file)
        
        self.model.to(self.device)
        
        # Print model summary
        print_variable_summary(self.model)
    
    def _create_model_from_scratch(self):
        '''
        Create model from scratch when pre-trained weights are not available
        '''
        # This is a simplified version for training from scratch
        class SimpleLanguageModel(nn.Module):
            def __init__(self, options, vocab_size):
                super().__init__()
                self.options = options
                
                dropout = float(options.get("dropout", 0.1))
                lstm_cfg = options.get("lstm", {})
                self.use_skip_connections = bool(
                    lstm_cfg.get("use_skip_connections", True)
                )
                self.proj_clip = lstm_cfg.get("proj_clip")
                if self.proj_clip is not None:
                    self.proj_clip = float(self.proj_clip)

                if options.get('char_cnn'):
                    # Character-based model
                    char_cfg = options["char_cnn"]
                    char_vocab_size = char_cfg["n_characters"]
                    char_embed_dim = char_cfg["embedding"]["dim"]
                    self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim)
                    self.char_activation = _get_activation_fn(
                        char_cfg.get("activation", "relu")
                    )

                    filters = char_cfg.get("filters") or [[3, 256]]
                    self.char_cnn_layers = nn.ModuleList(
                        nn.Conv1d(char_embed_dim, int(num), int(width), padding=int(width) // 2)
                        for width, num in filters
                    )
                    total_filters = sum(int(num) for _, num in filters)

                    n_highway = int(char_cfg.get("n_highway", 0))
                    self.highway_layers = nn.ModuleList(
                        nn.Linear(total_filters, total_filters * 2)
                        for _ in range(n_highway)
                    )

                    proj_dim = int(lstm_cfg["projection_dim"])
                    self.projection = nn.Linear(total_filters, proj_dim)
                    input_dim = proj_dim
                else:
                    # Token-based model
                    embed_dim = options.get('embedding_dim', 512)
                    self.token_embedding = nn.Embedding(vocab_size, embed_dim)
                    input_dim = embed_dim
                
                # LSTM layers
                hidden_dim = int(lstm_cfg["dim"])
                self.hidden_dim = hidden_dim
                n_layers = int(lstm_cfg["n_layers"])
                lstm_dropout = dropout if n_layers > 1 else 0.0
                
                self.forward_lstm = nn.LSTM(
                    input_dim, hidden_dim, n_layers, 
                    batch_first=True, dropout=lstm_dropout
                )
                self.backward_lstm = nn.LSTM(
                    input_dim, hidden_dim, n_layers, 
                    batch_first=True, dropout=lstm_dropout
                )
                self.input_dropout = nn.Dropout(dropout)
                self.skip_proj = (
                    nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None
                )
                
                # Output projection
                self.output_projection = nn.Linear(hidden_dim, vocab_size)
            
            def forward(self, inputs):
                if self.options.get('char_cnn'):
                    return self._forward_char(inputs)
                else:
                    return self._forward_token(inputs)
            
            def _forward_char(self, char_ids):
                batch_size, seq_len, max_chars = char_ids.shape
                mask = (char_ids.sum(dim=-1) > 0).float()
                
                # Character embedding and CNN
                char_embeds = self.char_embedding(char_ids)
                char_embeds = char_embeds.transpose(2, 3)
                
                # CNN processing from config char_cnn.filters
                x = char_embeds.view(-1, char_embeds.size(2), char_embeds.size(3))
                conv_outputs = []
                for conv in self.char_cnn_layers:
                    y = conv(x)
                    y = self.char_activation(y)
                    y = F.max_pool1d(y, kernel_size=y.size(2)).squeeze(2)
                    conv_outputs.append(y)
                x = torch.cat(conv_outputs, dim=1).view(batch_size, seq_len, -1)

                for highway in self.highway_layers:
                    gate_and_transform = highway(x)
                    transform, gate = torch.chunk(gate_and_transform, 2, dim=-1)
                    transform = F.relu(transform)
                    gate = torch.sigmoid(gate)
                    x = gate * transform + (1.0 - gate) * x
                
                token_embeddings = self.projection(x)
                
                return self._run_lstm(token_embeddings, mask)
            
            def _forward_token(self, token_ids):
                mask = (token_ids > 0).float()
                token_embeddings = self.token_embedding(token_ids)
                return self._run_lstm(token_embeddings, mask)
            
            def _run_lstm(self, embeddings, mask):
                embeddings = self.input_dropout(embeddings)
                # Forward LSTM
                forward_out, _ = self.forward_lstm(embeddings)
                
                # Backward LSTM
                backward_in = torch.flip(embeddings, dims=[1])
                backward_out, _ = self.backward_lstm(backward_in)
                backward_out = torch.flip(backward_out, dims=[1])

                if self.use_skip_connections:
                    skip = embeddings if self.skip_proj is None else self.skip_proj(embeddings)
                    forward_out = forward_out + skip
                    backward_out = backward_out + skip
                if self.proj_clip is not None:
                    forward_out = torch.clamp(
                        forward_out, min=-self.proj_clip, max=self.proj_clip
                    )
                    backward_out = torch.clamp(
                        backward_out, min=-self.proj_clip, max=self.proj_clip
                    )
                
                # Combine representations
                bi = torch.cat([forward_out, backward_out], dim=-1)
                skip_base = embeddings if self.skip_proj is None else self.skip_proj(embeddings)
                emb2 = torch.cat([skip_base, skip_base], dim=-1)
                lm_embeddings = torch.stack([emb2, bi], dim=1)
                
                return {
                    'lm_embeddings': lm_embeddings,
                    'mask': mask,
                    'forward_output': forward_out,
                    'backward_output': backward_out
                }
        
        vocab_size = len(self.vocab)
        return SimpleLanguageModel(self.options, vocab_size)
    
    def _init_optimizer(self):
        '''
        Initialize optimizer
        '''
        lr = self.options.get('learning_rate', 0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.options.get('lr_decay_steps', 10), 
            gamma=self.options.get('lr_decay_rate', 0.9)
        )
    
    def _init_data_loaders(self):
        '''
        Build BiLM data iterators from pretokenized shard files (glob pattern).
        '''
        self._train_lm = BidirectionalLMDataset(
            self.train_prefix,
            self.vocab,
            test=False,
            shuffle_on_load=True,
        )
        self._test_lm = None
        if self.test_prefix and glob.glob(self.test_prefix):
            self._test_lm = BidirectionalLMDataset(
                self.test_prefix,
                self.vocab,
                test=True,
                shuffle_on_load=False,
            )
        self._valid_prefix_resolved = (
            self.valid_prefix if self.valid_prefix and glob.glob(self.valid_prefix) else None
        )
        if self._valid_prefix_resolved:
            logger.info(
                "Validation shards for metrics: %d (%s)",
                len(glob.glob(self.valid_prefix)),
                self.valid_prefix,
            )
        n_train = len(glob.glob(self.train_prefix))
        logger.info(
            "Loaded BidirectionalLMDataset for train_prefix=%s (%d shards)",
            self.train_prefix,
            n_train,
        )
        if self._test_lm is not None:
            logger.info(
                "Test data: %d shards",
                len(glob.glob(self.test_prefix)),
            )
    
    def train_step(self, batch):
        '''
        Perform one training step on a batch from BidirectionalLMDataset.
        '''
        self.model.train()
        self.optimizer.zero_grad()

        loss = self._compute_lm_loss(batch)

        loss.backward()
        
        # Gradient clipping
        if self.options.get('clip_grad_norm'):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.options['clip_grad_norm']
            )
        
        self.optimizer.step()
        
        return loss.item()

    def _compute_lm_loss_components(self, batch):
        '''
        Forward + backward next-token losses (BiLM). Returns (total, nll_fwd, nll_bwd)
        each mean over positions; total = nll_fwd + nll_bwd (same scale as training loss).
        '''
        use_chars = self.options.get("char_cnn") is not None
        if use_chars:
            fwd = torch.from_numpy(batch["tokens_characters"]).long().to(self.device)
            rev = torch.from_numpy(batch["tokens_characters_reverse"]).long().to(
                self.device
            )
        else:
            fwd = torch.from_numpy(batch["token_ids"]).long().to(self.device)
            rev = torch.from_numpy(batch["token_ids_reverse"]).long().to(self.device)

        tgt_f = torch.from_numpy(batch["next_token_id"]).long().to(self.device)
        tgt_r = torch.from_numpy(batch["next_token_id_reverse"]).long().to(self.device)

        out_f = self.model(fwd)
        out_r = self.model(rev)

        logits_f = self.model.output_projection(out_f["forward_output"])
        logits_r = self.model.output_projection(out_r["forward_output"])

        vocab = logits_f.size(-1)
        loss_f = F.cross_entropy(
            logits_f.view(-1, vocab),
            tgt_f.view(-1),
        )
        loss_r = F.cross_entropy(
            logits_r.view(-1, vocab),
            tgt_r.view(-1),
        )
        total = loss_f + loss_r
        return total, loss_f, loss_r

    def _compute_lm_loss(self, batch):
        total, _, _ = self._compute_lm_loss_components(batch)
        return total

    def _run_validation(self):
        '''
        Mean BiLM loss and geometric-mean perplexity over a fixed number of
        validation batches. Rebuilds a one-pass dataset each call (test=True
        exhausts shards).
        Returns (val_loss, val_perplexity) or (None, None) if disabled / no data.
        '''
        if not self._valid_prefix_resolved:
            return None, None
        max_batches = int(self.options.get("eval_max_batches", 50))
        if max_batches <= 0:
            return None, None

        lm = BidirectionalLMDataset(
            self._valid_prefix_resolved,
            self.vocab,
            test=True,
            shuffle_on_load=False,
        )
        bs = self.options.get("batch_size", 32)
        unroll = self.options.get("unroll_steps", 20)
        it = lm.iter_batches(bs, unroll)

        sum_total = 0.0
        sum_f = 0.0
        sum_b = 0.0
        n = 0
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_batches):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                total, lf, lb = self._compute_lm_loss_components(batch)
                sum_total += float(total.item())
                sum_f += float(lf.item())
                sum_b += float(lb.item())
                n += 1
        self.model.train()

        if n == 0:
            return None, None
        avg_total = sum_total / n
        avg_f = sum_f / n
        avg_b = sum_b / n
        # Geometric mean of directional perplexities: sqrt(exp(f)*exp(b)) = exp((f+b)/2)
        val_ppl = math.exp((avg_f + avg_b) / 2.0)
        return avg_total, val_ppl

    def _compute_loss(self, outputs, inputs):
        '''
        Compute language modeling loss (single forward pass; legacy / eval).
        '''
        if 'forward_output' in outputs and 'backward_output' in outputs:
            forward_logits = self.model.output_projection(outputs['forward_output'])
            backward_logits = self.model.output_projection(outputs['backward_output'])

            if isinstance(inputs, dict) and 'targets' in inputs:
                targets = inputs['targets']
            else:
                batch_size, seq_len = forward_logits.shape[:2]
                targets = torch.randint(
                    0,
                    forward_logits.size(-1),
                    (batch_size, seq_len),
                    device=self.device,
                )

            forward_loss = F.cross_entropy(
                forward_logits.view(-1, forward_logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )
            backward_loss = F.cross_entropy(
                backward_logits.view(-1, backward_logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )

            return forward_loss + backward_loss
        return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def evaluate(self, data_loader=None):
        '''
        Evaluate on an explicit data_loader, or on test_prefix when configured.
        '''
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            if data_loader is not None:
                for batch in data_loader:
                    if isinstance(batch, dict):
                        inputs = {k: v.to(self.device) for k, v in batch.items()}
                    else:
                        inputs = batch.to(self.device)

                    outputs = self.model(inputs)
                    loss = self._compute_loss(outputs, inputs)

                    total_loss += loss.item()
                    num_batches += 1
            elif self._test_lm is not None:
                batch_iter = self._test_lm.iter_batches(
                    self.options.get('batch_size', 32),
                    self.options.get('unroll_steps', 20),
                )
                for batch in batch_iter:
                    total_loss += self._compute_lm_loss(batch).item()
                    num_batches += 1
            else:
                return None

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _log_metrics_csv(
        self,
        save_dir,
        event,
        loss=None,
        learning_rate=None,
        global_step=None,
        epoch=None,
        val_loss=None,
        val_perplexity=None,
    ):
        if not save_dir:
            return
        path = os.path.join(save_dir, "metrics.csv")
        os.makedirs(save_dir, exist_ok=True)
        new_file = not os.path.exists(path) or os.path.getsize(path) == 0
        row = [
            int(time.time()),
            global_step if global_step is not None else "",
            epoch if epoch is not None else "",
            "" if loss is None else float(loss),
            "" if learning_rate is None else float(learning_rate),
            "" if val_loss is None else float(val_loss),
            "" if val_perplexity is None else float(val_perplexity),
            event,
        ]
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(
                    [
                        "unix_time",
                        "global_step",
                        "epoch",
                        "loss",
                        "learning_rate",
                        "val_loss",
                        "val_perplexity",
                        "event",
                    ]
                )
            w.writerow(row)
    
    def train(self, num_epochs, save_dir=None):
        '''
        Train the model on pretokenized shards from BidirectionalLMDataset.
        '''
        logger.info(f"Starting training for {num_epochs} epochs")

        batch_size = self.options.get('batch_size', 32)
        num_steps = self.options.get('unroll_steps', 20)
        steps_per_epoch = self.options.get('steps_per_epoch', 1000)

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            total_loss = 0.0
            num_batches = 0

            batch_iter = self._train_lm.iter_batches(batch_size, num_steps)

            pbar = tqdm(
                range(steps_per_epoch),
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                leave=True,
            )
            for _ in pbar:
                batch = next(batch_iter)
                loss = self.train_step(batch)
                total_loss += loss
                num_batches += 1
                pbar.set_postfix(loss=f"{loss:.4f}")

            # Update learning rate
            self.scheduler.step()
            
            # Log epoch results
            avg_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start_time
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s, "
                f"Average Loss: {avg_loss:.4f}, "
                f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}"
            )
            val_loss, val_ppl = (
                self._run_validation() if self._valid_prefix_resolved else (None, None)
            )
            if val_loss is not None:
                logger.info(
                    "Epoch %d validation loss=%.4f perplexity=%.2f",
                    epoch + 1,
                    val_loss,
                    val_ppl,
                )
            self._log_metrics_csv(
                save_dir,
                "epoch",
                loss=avg_loss,
                learning_rate=self.scheduler.get_last_lr()[0],
                epoch=epoch + 1,
                val_loss=val_loss,
                val_perplexity=val_ppl,
            )
            
            # Save checkpoint
            if save_dir and (epoch + 1) % 5 == 0:
                self.save_checkpoint(save_dir, epoch=epoch + 1)
                self._log_metrics_csv(
                    save_dir,
                    "checkpoint",
                    loss=avg_loss,
                    learning_rate=self.scheduler.get_last_lr()[0],
                    epoch=epoch + 1,
                    val_loss=val_loss,
                    val_perplexity=val_ppl,
                )

        if save_dir:
            vloss, vppl = (
                self._run_validation() if self._valid_prefix_resolved else (None, None)
            )
            self._log_metrics_csv(
                save_dir,
                "training_finished",
                learning_rate=self.scheduler.get_last_lr()[0],
                epoch=num_epochs,
                val_loss=vloss,
                val_perplexity=vppl,
            )
        
        logger.info("Training completed!")

    def train_max_steps(self, max_steps, save_dir=None):
        '''
        Train for a fixed number of optimizer steps (infinite shard stream).
        '''
        logger.info("Starting training for %d steps", max_steps)

        batch_size = self.options.get('batch_size', 32)
        num_unroll = self.options.get('unroll_steps', 20)
        save_every = int(self.options.get('save_every_steps', 10000))
        log_every = int(self.options.get('log_every_steps', 100))
        raw_val_every = self.options.get("validation_every_steps")
        if raw_val_every is None:
            val_every = save_every
        else:
            val_every = int(raw_val_every)
        sched_interval = self.options.get("lr_scheduler_step_interval")
        if sched_interval is not None:
            sched_interval = int(sched_interval)

        batch_iter = self._train_lm.iter_batches(batch_size, num_unroll)
        sched_calls = 0

        pbar = tqdm(range(max_steps), desc="train", leave=True)
        last_loss = None
        last_val_loss = last_val_ppl = None
        for step in pbar:
            batch = next(batch_iter)
            loss = self.train_step(batch)
            last_loss = loss
            pbar.set_postfix(loss=f"{loss:.4f}")
            lr = self.scheduler.get_last_lr()[0]

            val_loss = val_ppl = None
            if self._valid_prefix_resolved and val_every > 0:
                at_val_interval = (step + 1) % val_every == 0
                at_ckpt = save_every > 0 and (step + 1) % save_every == 0
                if at_val_interval or at_ckpt:
                    val_loss, val_ppl = self._run_validation()
                    if val_loss is not None:
                        last_val_loss, last_val_ppl = val_loss, val_ppl
                        logger.info(
                            "step %d validation loss=%.4f perplexity=%.2f",
                            step + 1,
                            val_loss,
                            val_ppl,
                        )

            if log_every > 0 and (step + 1) % log_every == 0:
                logger.info("step %d loss=%.4f", step + 1, loss)
                self._log_metrics_csv(
                    save_dir,
                    "train",
                    loss=loss,
                    learning_rate=lr,
                    global_step=step + 1,
                    val_loss=val_loss,
                    val_perplexity=val_ppl,
                )

            if save_dir and save_every > 0 and (step + 1) % save_every == 0:
                self.save_checkpoint(save_dir, global_step=step + 1)
                self._log_metrics_csv(
                    save_dir,
                    "checkpoint",
                    loss=loss,
                    learning_rate=self.scheduler.get_last_lr()[0],
                    global_step=step + 1,
                    val_loss=val_loss,
                    val_perplexity=val_ppl,
                )

            if sched_interval and (step + 1) % sched_interval == 0:
                self.scheduler.step()
                sched_calls += 1
                lr_after = self.scheduler.get_last_lr()[0]
                logger.info(
                    "LR scheduler step (%d), lr=%.6f",
                    sched_calls,
                    lr_after,
                )
                self._log_metrics_csv(
                    save_dir,
                    "lr_schedule",
                    loss=loss,
                    learning_rate=lr_after,
                    global_step=step + 1,
                    val_loss=val_loss,
                    val_perplexity=val_ppl,
                )

        vfin_loss, vfin_ppl = last_val_loss, last_val_ppl
        if (
            save_dir
            and self._valid_prefix_resolved
            and val_every > 0
            and vfin_loss is None
        ):
            vfin_loss, vfin_ppl = self._run_validation()
            if vfin_loss is not None:
                logger.info(
                    "final validation loss=%.4f perplexity=%.2f",
                    vfin_loss,
                    vfin_ppl,
                )

        if save_dir and last_loss is not None:
            self._log_metrics_csv(
                save_dir,
                "training_finished",
                loss=last_loss,
                learning_rate=self.scheduler.get_last_lr()[0],
                global_step=max_steps,
                val_loss=vfin_loss,
                val_perplexity=vfin_ppl,
            )

        logger.info("Training completed (%d steps).", max_steps)
    
    def save_checkpoint(self, save_dir, epoch=None, global_step=None):
        '''
        Save model checkpoint (epoch- or step-based filename).
        '''
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'options': self.options
        }

        if global_step is not None:
            checkpoint_path = os.path.join(
                save_dir, f"checkpoint_step_{global_step:07d}.pt"
            )
        else:
            checkpoint_path = os.path.join(
                save_dir, f"checkpoint_epoch_{epoch}.pt"
            )
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        '''
        Load model checkpoint
        '''
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch']


def train_elmo(
    options,
    train_prefix,
    vocab_file,
    save_dir,
    test_prefix=None,
    valid_prefix=None,
    num_epochs=10,
    max_steps=None,
):
    '''
    Train an ELMo model
    
    Args:
        options: dictionary containing model hyperparameters
        train_prefix: prefix for training data files
        vocab_file: path to vocabulary file
        save_dir: directory to save model checkpoints
        test_prefix: prefix for test data files (optional)
        valid_prefix: glob for validation shards (optional; val loss / perplexity in metrics)
        num_epochs: number of training epochs (if max_steps is None)
        max_steps: train for this many optimizer steps (takes precedence over epochs)
    '''
    # Initialize trainer
    trainer = LanguageModelTrainer(
        options, train_prefix, vocab_file, test_prefix, valid_prefix=valid_prefix
    )
    
    # Train model
    if max_steps is not None:
        trainer.train_max_steps(max_steps, save_dir)
    else:
        trainer.train(num_epochs, save_dir)
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'options': options,
        'vocab_file': vocab_file
    }, final_model_path)
    
    logger.info(f"Final model saved to {final_model_path}")
    
    return trainer