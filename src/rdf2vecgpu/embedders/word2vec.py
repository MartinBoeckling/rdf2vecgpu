import torch
import torch.nn as nn
from torch.optim import Adam
import lightning as L
import numpy as np
from torch.nn import functional as F

torch.cuda.set_per_process_memory_fraction(0.9)


class SkipGram(L.LightningModule):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        neg_samples,
        learning_rate=0.003,
        use_sparse=False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=use_sparse)
        self.output_embeddings = nn.Embedding(
            vocab_size, embedding_dim, sparse=use_sparse
        )

        bound = 1.0 / embedding_dim

        nn.init.uniform_(self.in_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.output_embeddings.weight, -bound, bound)

        self.neg_samples = neg_samples
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
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
        negative = torch.randint(
            0, self.vocab_size, (center.size(0), self.neg_samples), device=self.device
        )
        loss = self(center, context, negative)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt_function = torch.optim.SparseAdam if self.use_sparse else torch.optim.Adam
        return opt_function(self.parameters(), lr=self.learning_rate)


class CBOW(L.LightningModule):
    def __init__(
        self, vocab_size, embedding_dim, learning_rate=0.003, use_sparse=False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.use_sparse = use_sparse

        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=use_sparse)
        self.output_embeddings = nn.Linear(embedding_dim, vocab_size)

        bound = 1.0 / embedding_dim
        nn.init.uniform_(self.in_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.output_embeddings.weight, -bound, bound)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, context):
        mask = (context != -1).float()

        context_embeds = self.in_embeddings(context.clamp(min=0))

        masked_embeds = context_embeds * mask.unsqueeze(-1)
        sum_embeds = masked_embeds.sum(dim=1)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        context_mean = sum_embeds / counts

        logits = self.output_embeddings(context_mean)

        return logits

    def training_step(self, batch, batch_idx):
        context, center = batch
        center = center.long()
        logits = self(context)
        loss = self.loss_fn(logits, center)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class OrderAwareSkipgram(L.LightningModule):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        neg_samples,
        max_distance=5,
        learning_rate=0.003,
        use_sparse=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.neg_samples = neg_samples
        self.max_distance = max_distance
        self.learning_rate = learning_rate
        self.use_sparse = use_sparse

        # Center word embeddings
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=use_sparse)

        # Context embeddings with position-specific weights
        # We create separate output embeddings for each possible distance
        self.output_embeddings = nn.ModuleList(
            [
                nn.Embedding(vocab_size, embedding_dim, sparse=use_sparse)
                for _ in range(2 * max_distance + 1)  # -max_distance to +max_distance
            ]
        )

        # Position embeddings to encode relative distance
        self.position_embeddings = nn.Embedding(2 * max_distance + 1, embedding_dim)

        # Initialize embeddings
        bound = 1.0 / embedding_dim
        nn.init.uniform_(self.in_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.position_embeddings.weight, -bound, bound)

        for output_emb in self.output_embeddings:
            nn.init.uniform_(output_emb.weight, -bound, bound)

        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def _get_position_index(self, distance):
        """Convert relative distance to position index"""
        return distance + self.max_distance

    def forward(self, center, context, distances, negative, neg_distances=None):
        """
        Args:
            center: (batch_size,) center word indices
            context: (batch_size,) context word indices
            distances: (batch_size,) relative distances from center to context
            negative: (batch_size, neg_samples) negative sample indices
            neg_distances: (batch_size, neg_samples) distances for negative samples
        """
        batch_size = center.size(0)

        # Get center embeddings
        center_embed = self.in_embeddings(center)  # (batch_size, embed_dim)

        # Get position-aware context embeddings
        pos_indices = self._get_position_index(distances)  # (batch_size,)

        # Compute positive scores with position awareness
        pos_logits = []
        for i in range(batch_size):
            pos_idx = pos_indices[i].item()
            context_embed = self.output_embeddings[pos_idx](
                context[i : i + 1]
            )  # (1, embed_dim)
            pos_embed = self.position_embeddings(
                pos_indices[i : i + 1]
            )  # (1, embed_dim)

            # Combine context and position embeddings
            combined_embed = context_embed + pos_embed

            # Compute similarity score
            logit = (center_embed[i : i + 1] * combined_embed).sum(dim=1)
            pos_logits.append(logit)

        pos_logits = torch.cat(pos_logits, dim=0)  # (batch_size,)
        pos_targets = torch.ones_like(pos_logits)
        pos_loss = self.loss_fn(pos_logits, pos_targets)

        # Compute negative scores
        if neg_distances is None:
            # Sample random distances for negative samples
            neg_distances = torch.randint(
                -self.max_distance,
                self.max_distance + 1,
                (batch_size, self.neg_samples),
                device=center.device,
            )

        neg_pos_indices = self._get_position_index(
            neg_distances
        )  # (batch_size, neg_samples)

        neg_logits = []
        for i in range(batch_size):
            center_i = center_embed[i : i + 1]  # (1, embed_dim)
            neg_logits_i = []

            for j in range(self.neg_samples):
                pos_idx = neg_pos_indices[i, j].item()
                neg_embed = self.output_embeddings[pos_idx](
                    negative[i : i + 1, j : j + 1]
                )  # (1, 1, embed_dim)
                neg_pos_embed = self.position_embeddings(
                    neg_pos_indices[i : i + 1, j : j + 1]
                )  # (1, 1, embed_dim)

                # Combine negative context and position embeddings
                combined_neg_embed = neg_embed.squeeze(1) + neg_pos_embed.squeeze(
                    1
                )  # (1, embed_dim)

                # Compute similarity score
                logit = (center_i * combined_neg_embed).sum(dim=1)
                neg_logits_i.append(logit)

            neg_logits.append(torch.cat(neg_logits_i, dim=0))

        neg_logits = torch.stack(neg_logits, dim=0)  # (batch_size, neg_samples)
        neg_targets = torch.zeros_like(neg_logits)
        neg_loss = self.loss_fn(neg_logits, neg_targets).sum(dim=1)

        return torch.mean(pos_loss + neg_loss)

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            center, context, distances = batch
            neg_distances = None
        else:
            center, context, distances, neg_distances = batch

        batch_size = center.size(0)

        # Generate negative samples
        negative = torch.randint(
            0, self.vocab_size, (batch_size, self.neg_samples), device=self.device
        )

        # Generate negative distances if not provided
        if neg_distances is None:
            neg_distances = torch.randint(
                -self.max_distance,
                self.max_distance + 1,
                (batch_size, self.neg_samples),
                device=self.device,
            )

        loss = self(center, context, distances, negative, neg_distances)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt_function = torch.optim.SparseAdam if self.use_sparse else torch.optim.Adam
        return opt_function(self.parameters(), lr=self.learning_rate)

    def get_embeddings(self):
        """Return the center word embeddings for downstream tasks"""
        return self.in_embeddings.weight.data

    def get_context_embeddings(self, distance=0):
        """Get context embeddings for a specific distance"""
        pos_idx = self._get_position_index(torch.tensor(distance))
        return self.output_embeddings[pos_idx.item()].weight.data


class OrderAwareCBOW(L.LightningModule):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        neg_samples,
        max_distance=5,
        learning_rate=0.003,
        use_sparse=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.neg_samples = neg_samples
        self.max_distance = max_distance
        self.learning_rate = learning_rate
        self.use_sparse = use_sparse

        # Context word embeddings with position-specific weights
        self.in_embeddings = nn.ModuleList(
            [
                nn.Embedding(vocab_size, embedding_dim, sparse=use_sparse)
                for _ in range(2 * max_distance + 1)  # -max_distance to +max_distance
            ]
        )

        # Output embeddings for center words
        self.output_embeddings = nn.Embedding(
            vocab_size, embedding_dim, sparse=use_sparse
        )

        # Position embeddings to encode relative distance
        self.position_embeddings = nn.Embedding(2 * max_distance + 1, embedding_dim)

        # Initialize embeddings
        bound = 1.0 / embedding_dim
        for input_emb in self.in_embeddings:
            nn.init.uniform_(input_emb.weight, -bound, bound)
        nn.init.uniform_(self.output_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.position_embeddings.weight, -bound, bound)

        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def _get_position_index(self, distance):
        """Convert relative distance to position index"""
        return distance + self.max_distance

    def forward(
        self, context_words, context_distances, center, negative, neg_distances=None
    ):
        """
        Args:
            context_words: (batch_size, context_size) context word indices
            context_distances: (batch_size, context_size) relative distances from center to each context word
            center: (batch_size,) center word indices
            negative: (batch_size, neg_samples) negative sample indices
            neg_distances: (batch_size, neg_samples) distances for negative samples (not used in CBOW)
        """
        batch_size, context_size = context_words.size()

        # Get position-aware context embeddings and combine them
        context_embeds = []
        for i in range(batch_size):
            batch_context_embeds = []
            for j in range(context_size):
                # Get position index for this context word
                distance = context_distances[i, j]
                pos_idx = self._get_position_index(distance).item()

                # Get context embedding from position-specific embedding matrix
                ctx_embed = self.in_embeddings[pos_idx](
                    context_words[i : i + 1, j : j + 1]
                )  # (1, 1, embed_dim)
                pos_embed = self.position_embeddings(
                    self._get_position_index(distance).unsqueeze(0)
                )  # (1, embed_dim)

                # Combine context and position embeddings
                combined_embed = ctx_embed.squeeze(1) + pos_embed  # (1, embed_dim)
                batch_context_embeds.append(combined_embed)

            # Average the context embeddings for this batch item
            avg_context = torch.mean(
                torch.cat(batch_context_embeds, dim=0), dim=0, keepdim=True
            )  # (1, embed_dim)
            context_embeds.append(avg_context)

        context_embeds = torch.cat(context_embeds, dim=0)  # (batch_size, embed_dim)

        # Get center word embeddings
        center_embed = self.output_embeddings(center)  # (batch_size, embed_dim)

        # Compute positive scores
        pos_logits = (context_embeds * center_embed).sum(dim=1)  # (batch_size,)
        pos_targets = torch.ones_like(pos_logits)
        pos_loss = self.loss_fn(pos_logits, pos_targets)

        # Compute negative scores
        negative_embed = self.output_embeddings(
            negative
        )  # (batch_size, neg_samples, embed_dim)

        # Expand context embeddings to match negative samples shape
        context_embeds_expanded = context_embeds.unsqueeze(
            1
        )  # (batch_size, 1, embed_dim)

        # Compute negative logits
        neg_logits = (context_embeds_expanded * negative_embed).sum(
            dim=2
        )  # (batch_size, neg_samples)
        neg_targets = torch.zeros_like(neg_logits)
        neg_loss = self.loss_fn(neg_logits, neg_targets).sum(dim=1)

        return torch.mean(pos_loss + neg_loss)

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            context_words, context_distances, center = batch
        else:
            context_words, context_distances, center, _ = (
                batch  # Ignore neg_distances if provided
            )

        batch_size = center.size(0)

        # Generate negative samples
        negative = torch.randint(
            0, self.vocab_size, (batch_size, self.neg_samples), device=self.device
        )

        loss = self(context_words, context_distances, center, negative)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt_function = torch.optim.SparseAdam if self.use_sparse else torch.optim.Adam
        return opt_function(self.parameters(), lr=self.learning_rate)

    def get_embeddings(self):
        """Return the center word embeddings for downstream tasks"""
        return self.output_embeddings.weight.data

    def get_context_embeddings(self, distance=0):
        """Get context embeddings for a specific distance"""
        pos_idx = self._get_position_index(torch.tensor(distance))
        return self.in_embeddings[pos_idx.item()].weight.data

    def predict_center(self, context_words, context_distances, top_k=5):
        """Predict the most likely center words given context"""
        self.eval()
        with torch.no_grad():
            batch_size, context_size = context_words.size()

            # Get position-aware context embeddings
            context_embeds = []
            for i in range(batch_size):
                batch_context_embeds = []
                for j in range(context_size):
                    distance = context_distances[i, j]
                    pos_idx = self._get_position_index(distance).item()

                    ctx_embed = self.in_embeddings[pos_idx](
                        context_words[i : i + 1, j : j + 1]
                    )
                    pos_embed = self.position_embeddings(
                        self._get_position_index(distance).unsqueeze(0)
                    )

                    combined_embed = ctx_embed.squeeze(1) + pos_embed
                    batch_context_embeds.append(combined_embed)

                avg_context = torch.mean(
                    torch.cat(batch_context_embeds, dim=0), dim=0, keepdim=True
                )
                context_embeds.append(avg_context)

            context_embeds = torch.cat(context_embeds, dim=0)

            # Compute similarity with all center words
            all_center_embeds = self.output_embeddings.weight  # (vocab_size, embed_dim)
            scores = torch.mm(
                context_embeds, all_center_embeds.t()
            )  # (batch_size, vocab_size)

            # Get top-k predictions
            _, top_indices = torch.topk(scores, top_k, dim=1)

            return top_indices
