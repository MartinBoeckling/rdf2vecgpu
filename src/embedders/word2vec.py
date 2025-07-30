import torch
import torch.nn as nn
from torch.optim import Adam
import lightning as L
import numpy as np

torch.cuda.set_per_process_memory_fraction(0.9)

class SkipGram(L.LightningModule):
    def __init__(self, vocab_size, embedding_dim, neg_samples, learning_rate=0.003, use_sparse=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=use_sparse)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=use_sparse)
        
        bound = 1.0 / embedding_dim

        nn.init.uniform_(self.in_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.output_embeddings.weight, -bound, bound)

        self.neg_samples = neg_samples
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.use_sparse = use_sparse

    def forward(self, center, context, negative):
        center_embed = self.in_embeddings(center)  # (batch, embed)
        context_embed = self.output_embeddings(context)  # (batch, embed)
        negative_embed = self.output_embeddings(negative)  # (batch, neg, embed)

        # Positive score
        pos_logits = (center_embed * context_embed).sum(dim=1)
        pos_targets = torch.ones_like(pos_logits)
        # pos_score = torch.sum(torch.mul(center_embed, context_embed), dim=1)
        pos_loss = self.loss_fn(pos_logits, pos_targets)

        # Negative score
        neg_score = torch.bmm(negative_embed, center_embed.unsqueeze(-1)).squeeze(-1)
        neg_targets = torch.zeros_like(neg_score)
        neg_loss = self.loss_fn(neg_score, neg_targets).sum(dim=1)

        return torch.mean(pos_loss + neg_loss)

    def training_step(self, batch, batch_idx):
        center, context = batch
        negative = torch.randint(0, self.vocab_size, (center.size(0), self.neg_samples), device=self.device)
        loss = self(center, context, negative)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt_function = torch.optim.SparseAdam if self.use_sparse else torch.optim.Adam
        return opt_function(self.parameters(), lr=self.learning_rate)
    

class CBOW(L.LightningModule):
    def __init__(self, vocab_size, embedding_dim, learning_rate, use_sparse=True):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.use_sparse = use_sparse

        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

        bound = 1.0 / embedding_dim
        nn.init.uniform_(self.in_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.output_embeddings.weight, -bound, bound)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, center, context, negative):
        center_embed = self.in_embeddings(center)  # (batch, embed)
        context_embed = self.output_embeddings(context)  # (batch, embed)
        negative_embed = self.output_embeddings(negative)


        pos_logits = torch.sum(torch.mul(center_embed, context_embed), dim=-1)
        pos_targets = torch.ones_like(pos_logits)
        pos_loss = self.loss_fn(pos_logits, pos_targets)

        neg_logits = torch.bmm(center_embed, context_embed.unsqueeze(-1)).squeeze(-1)
        neg_targets = torch.zeros_like(neg_logits)
        neg_loss = self.loss_fn(neg_logits, neg_targets)

        return torch.mean(pos_loss + neg_loss)

    def training_step(self, batch, batch_idx):
        center, context = batch
        batch_size, device =  center.size(0), center.device

        negative = torch.randint(0, self.vocab_size, (batch_size, self.window_size), device=device)
        loss = self(center, context, negative)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt_function = torch.optim.SparseAdam if self.use_sparse else torch.optim.Adam
        return opt_function(self.parameters(), lr=self.learning_rate)
    

class OrderAwareSkipgram:
    def __init__(self):
        super().__init__()
        pass