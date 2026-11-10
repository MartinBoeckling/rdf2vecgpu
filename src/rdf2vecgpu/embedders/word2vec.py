import torch
import torch.nn as nn
from torch.optim import Adam
import lightning as L
import numpy as np
from torch.nn import functional as F


class SkipGram(L.LightningModule):
    """Skip-gram with negative sampling, trained the way word2vec and gensim do.

    * The loss is the **sum** over a batch's (center, context) pairs, so
      ``learning_rate`` is the step size per pair, as gensim's ``alpha``.
      A mean would divide each pair's step by the batch size.
    * Plain SGD with the learning rate decaying linearly per optimiser step
      from ``learning_rate`` to ``min_learning_rate`` over all epochs
      (gensim's ``alpha`` schedule). SGD keeps no per-parameter state, and with
      ``use_sparse=True`` each step only touches the rows of the batch.
    * Gradients are clipped per component to ``grad_clip_value`` after the
      backward pass. With a summed loss a few very frequent tokens (typically
      predicates) collect a large gradient in every batch; unclipped, their
      step can overflow.
    * ``sample`` drops frequent token occurrences as gensim's ``sample`` does,
      and ``negative_sampling="unigram"`` draws negatives from the corpus
      frequencies to the power 0.75. Both need ``token_counts``.

    A batch is either ``(vertices, predicates)`` walk rows, shape
    ``(n, depth + 1)`` and ``(n, depth)`` (``-1`` pads walks that ended early),
    whose pairs are built on the GPU from each walk's interleaved
    vertex/predicate sequence; or flat ``(center, context)`` pairs.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        neg_samples,
        learning_rate=0.025,
        use_sparse=True,
        epochs=5,
        min_learning_rate=0.0001,
        lr_decay="linear",
        window=5,
        steps_per_epoch=None,
        grad_clip_value=0.1,
        sample=0.0,
        negative_sampling="uniform",
        token_counts=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["token_counts"])
        if lr_decay not in ("linear", "none"):
            raise ValueError(f"lr_decay must be 'linear' or 'none', got {lr_decay!r}")
        if negative_sampling not in ("uniform", "unigram"):
            raise ValueError(
                f"negative_sampling must be 'uniform' or 'unigram', got {negative_sampling!r}"
            )
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_learning_rate = min_learning_rate
        self.lr_decay = lr_decay
        self.window = window
        self.steps_per_epoch = steps_per_epoch
        self.grad_clip_value = grad_clip_value
        self.sample = float(sample)
        self.negative_sampling = negative_sampling
        if token_counts is not None:
            self._token_counts = torch.as_tensor(np.asarray(token_counts), dtype=torch.double).cpu()
            if self._token_counts.numel() != vocab_size:
                raise ValueError(
                    f"token_counts has {self._token_counts.numel()} entries, vocab_size is {vocab_size}"
                )
        else:
            self._token_counts = None
            if self.sample > 0 or self.negative_sampling == "unigram":
                raise ValueError("sample > 0 and negative_sampling='unigram' need token_counts")
        # built on first use, on the batch's device
        self._p_keep = None
        self._neg_weights = None
        self._pair_templates = {}

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

    def get_embeddings(self):
        """The input embedding table, which is the one returned as embeddings."""
        return self.in_embeddings.weight.data

    def _get_p_keep(self, device):
        """Keep probability per token for frequent-token subsampling, as in
        gensim: with corpus frequency z, (sqrt(z / sample) + 1) * sample / z,
        at most 1 (tokens rarer than ``sample`` are always kept)."""
        if self._p_keep is None or self._p_keep.device != device:
            counts = self._token_counts.to(device)
            z = (counts / counts.sum().clamp(min=1.0)).clamp(min=1e-30)
            p = (torch.sqrt(z / self.sample) + 1.0) * (self.sample / z)
            self._p_keep = p.clamp(max=1.0).to(torch.float32)
        return self._p_keep

    def _get_neg_weights(self, device):
        """Unnormalised count^0.75 weights; tokens absent from the corpus get 0."""
        if self._neg_weights is None or self._neg_weights.device != device:
            self._neg_weights = self._token_counts.to(device).clamp(min=0).pow(0.75).to(torch.float32)
        return self._neg_weights

    def _sample_negatives(self, n_rows, device):
        """(n_rows, neg_samples) negative token ids."""
        if self.negative_sampling == "unigram":
            weights = self._get_neg_weights(device)
            return torch.multinomial(
                weights, n_rows * self.neg_samples, replacement=True
            ).view(n_rows, self.neg_samples)
        return torch.randint(0, self.vocab_size, (n_rows, self.neg_samples), device=device)

    def forward(self, center, context, negative):
        center_embed = self.in_embeddings(center)  # (batch, embed)
        context_embed = self.output_embeddings(context)  # (batch, embed)
        negative_embed = self.output_embeddings(negative)  # (batch, neg, embed)

        pos_logits = (center_embed * context_embed).sum(dim=1)
        pos_loss = self.loss_fn(pos_logits, torch.ones_like(pos_logits))

        neg_score = torch.bmm(negative_embed, center_embed.unsqueeze(-1)).squeeze(-1)
        neg_loss = self.loss_fn(neg_score, torch.zeros_like(neg_score)).sum(dim=1)

        return torch.sum(pos_loss + neg_loss)

    def _pair_template_for(self, length, device):
        """(i, j) position pairs with 0 < |i - j| <= window in a sequence of
        `length` tokens, as long tensors on `device`; cached per length."""
        cached = self._pair_templates.get(length)
        if cached is not None and cached[0].device == device:
            return cached
        ii, jj = [], []
        for i in range(length):
            for j in range(max(0, i - self.window), min(length, i + self.window + 1)):
                if j != i:
                    ii.append(i)
                    jj.append(j)
        template = (
            torch.as_tensor(ii, dtype=torch.long, device=device),
            torch.as_tensor(jj, dtype=torch.long, device=device),
        )
        self._pair_templates[length] = template
        return template

    def _expand_skipgram(self, vertices, predicates):
        """All skip-gram (center, context) pairs of a batch of walks, built on
        the GPU. A walk is the sequence v0, p0, v1, p1, ..., cut where its
        vertices hit the ``-1`` padding; with ``sample > 0`` each token
        occurrence is dropped with probability 1 - p_keep, and with it every
        pair it is part of."""
        n, n_vertices = vertices.shape
        length = 2 * n_vertices - 1
        device = vertices.device
        seq = torch.full((n, length), -1, dtype=torch.long, device=device)
        seq[:, 0::2] = vertices
        seq[:, 1::2] = predicates
        padded = vertices.eq(-1)
        first_pad = torch.where(
            padded.any(dim=1),
            padded.long().argmax(dim=1),
            torch.full((n,), n_vertices, device=device, dtype=torch.long),
        )
        valid_len = (2 * first_pad - 1).clamp(min=0)
        ii, jj = self._pair_template_for(length, device)
        center = seq[:, ii].reshape(-1)
        context = seq[:, jj].reshape(-1)
        keep = (ii.unsqueeze(0) < valid_len.unsqueeze(1)) & (jj.unsqueeze(0) < valid_len.unsqueeze(1))
        if self.sample > 0:
            p_keep = self._get_p_keep(device)
            kept = torch.rand((n, length), device=device) < p_keep[seq.clamp(min=0)]
            keep = keep & kept[:, ii] & kept[:, jj]
        keep = keep.reshape(-1)
        return center[keep], context[keep]

    def training_step(self, batch, batch_idx):
        first, second = batch
        if first.dim() == 2:
            center, context = self._expand_skipgram(first, second)
        else:
            center, context = first.long(), second.long()
        if center.numel() == 0:
            # every pair of the batch was subsampled away (only possible for
            # tiny batches): a zero loss leaves the embeddings unchanged
            return torch.zeros((), device=center.device, requires_grad=True)
        negative = self._sample_negatives(center.size(0), center.device)
        loss = self(center, context, negative)
        self.log("train_loss", loss / center.size(0), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        if self.lr_decay != "linear":
            return optimizer
        end_factor = max(self.min_learning_rate / self.learning_rate, 1e-12)
        if self.steps_per_epoch is None:
            if self.epochs <= 1:
                return optimizer
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=end_factor, total_iters=self.epochs - 1
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        # the last step runs at min_learning_rate (LinearLR stays there after
        # total_iters), as gensim's alpha does
        total_iters = max(1, self.epochs * max(1, int(self.steps_per_epoch)) - 1)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=end_factor, total_iters=total_iters
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_after_backward(self):
        # Lightning's gradient_clip_val cannot clip sparse gradients, so clip
        # here, on the coalesced gradient (duplicate rows summed first).
        clip = self.grad_clip_value
        if not clip:
            return
        for embedding in (self.in_embeddings, self.output_embeddings):
            grad = embedding.weight.grad
            if grad is None:
                continue
            if grad.is_sparse:
                grad = grad.coalesce()
                grad._values().clamp_(-clip, clip)
                embedding.weight.grad = grad
            else:
                grad.clamp_(-clip, clip)


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

    def get_embeddings(self):
        """The input embedding table, which is the one returned as embeddings."""
        return self.in_embeddings.weight.data


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
