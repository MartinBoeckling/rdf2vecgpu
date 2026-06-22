import torch
import torch.nn as nn
from torch.optim import Adam
import lightning as L
import numpy as np
from torch.nn import functional as F


class SkipGram(L.LightningModule):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        neg_samples,
        learning_rate=0.025,
        use_sparse=True,
        epochs=5,
        min_learning_rate=0.0001,
        lr_decay="linear",            # "linear" | "none"
        window=5,                     # for the on-GPU walk-sequence pair expansion
        accumulate_grad_batches=1,    # mirrors lightning.Trainer's setting
        steps_per_epoch=None,         # training batches/epoch; drives the LR decay length
        grad_clip_value=0.1,          # grad clamp magnitude (see grad_clip_mode); 0 disables
        optimizer="sgd",              # "sgd" | "sparseadam"
        grad_clip_mode="element",     # "element": |grad|≤v per component; "row": ‖grad_row‖₂≤v
        sample=0.001,                 # frequent-token subsampling threshold (gensim's `sample`); 0 disables. Default 1e-3 = gensim's classic recipe (spec §7.4).
        negative_sampling="unigram",  # "uniform" | "unigram" (classic f^0.75; default per spec §7.4).
        token_counts=None,            # 1-D array/tensor of corpus token counts, len == vocab_size;
                                      # required when sample>0 or negative_sampling=="unigram"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["token_counts"])
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_learning_rate = min_learning_rate
        self.lr_decay = lr_decay
        self.window = window
        self.accumulate_grad_batches = max(1, int(accumulate_grad_batches))
        self.steps_per_epoch = steps_per_epoch
        self.grad_clip_value = grad_clip_value
        self.optimizer_name = optimizer
        self.grad_clip_mode = grad_clip_mode
        self.sample = float(sample)
        self.negative_sampling = negative_sampling
        # Token counts → lazily-built device tensors (built on first use, on the
        # batch's device): `_p_keep` (subsampling keep-probability per token) and
        # `_neg_cdf` (normalised cumulative f^0.75 weights for unigram negatives).
        if token_counts is not None:
            self._token_counts = torch.as_tensor(np.asarray(token_counts), dtype=torch.double).cpu()
            if self._token_counts.numel() != vocab_size:
                raise ValueError(
                    f"token_counts length {self._token_counts.numel()} != vocab_size {vocab_size}")
        else:
            self._token_counts = None
            if self.sample > 0 or self.negative_sampling == "unigram":
                raise ValueError("sample>0 / negative_sampling='unigram' require token_counts")
        self._p_keep = None
        self._neg_cdf = None

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
        # window-shift skip-gram pair templates, keyed by interleaved-walk length;
        # built lazily on the right device.
        self._pair_templates = {}

    def get_embeddings(self):
        """Entity-level embedding table used downstream (the input/syn0 table)."""
        return self.in_embeddings.weight.data

    # ── frequent-token subsampling + unigram^0.75 negatives (built lazily on the
    #    batch's device from `token_counts`) ────────────────────────────────────
    def _get_p_keep(self, device):
        """Per-token keep-probability for gensim-style subsampling: for token w
        with corpus frequency z = count(w)/total, p_keep = (sqrt(z/sample) + 1) *
        sample/z, clamped to ≤ 1 (rare tokens, z < sample, are always kept)."""
        if self._p_keep is None or self._p_keep.device != device:
            c = self._token_counts.to(device)
            z = (c / c.sum().clamp(min=1.0)).clamp(min=1e-30)
            p = (torch.sqrt(z / self.sample) + 1.0) * (self.sample / z)
            self._p_keep = p.clamp(max=1.0).to(torch.float32)
        return self._p_keep

    def _get_neg_weights(self, device):
        """Unnormalised f^0.75 unigram weights over the vocab (torch.multinomial
        normalises). Tokens absent from the corpus get weight 0."""
        if self._neg_cdf is None or self._neg_cdf.device != device:
            self._neg_cdf = self._token_counts.to(device).clamp(min=0).pow(0.75).to(torch.float32)
        return self._neg_cdf

    def _sample_negatives(self, n_rows, device):
        """(n_rows, neg_samples) negative token ids — uniform, or f^0.75 unigram."""
        if self.negative_sampling == "unigram":
            w = self._get_neg_weights(device)
            return torch.multinomial(
                w, n_rows * self.neg_samples, replacement=True
            ).view(n_rows, self.neg_samples)
        return torch.randint(0, self.vocab_size, (n_rows, self.neg_samples), device=device)

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

        # SUM over the batch of pairs, not mean: with `WalkSequenceDataModule`
        # batching ~20k walks → ~1M (center, context) pairs/step, `torch.mean`
        # divides every row's gradient by ~1M, so the effective per-pair SGD step
        # becomes lr·(few/1M)·grad ≈ 1e-10 vs an init magnitude ~1/dim — the
        # embeddings never move (loss flat at 0.693·(1+neg); spike's first DBpedia
        # run came out ≈ random init). With `sum`, `lr` (0.025, decaying linearly)
        # is the per-pair learning rate, matching classic word2vec / gensim.
        return torch.sum(pos_loss + neg_loss)

    def _pair_template_for(self, L, device):
        """(ii, jj) long tensors on `device` of all (i, j) with 0 < |i-j| <= window,
        0 <= i, j < L. Cached per (L, device)."""
        cached = self._pair_templates.get(L)
        if cached is not None and cached[0].device == device:
            return cached
        w = self.window
        ii, jj = [], []
        for i in range(L):
            for j in range(max(0, i - w), min(L, i + w + 1)):
                if j != i:
                    ii.append(i)
                    jj.append(j)
        t = (torch.as_tensor(ii, dtype=torch.long, device=device),
             torch.as_tensor(jj, dtype=torch.long, device=device))
        self._pair_templates[L] = t
        return t

    def _expand_skipgram(self, vertices, predicates):
        """vertices: (n, depth+1) int (cuGraph `-1` sentinels pad short walks);
        predicates: (n, depth) int (cuGraph pads with 0). Returns flat 1-D long
        tensors (center, context) of all valid skip-gram pairs."""
        n, depth_p1 = vertices.shape
        Lint = 2 * depth_p1 - 1
        device = vertices.device
        seq = vertices.new_full((n, Lint), -1)
        seq[:, 0::2] = vertices
        seq[:, 1::2] = predicates
        seq = seq.long()
        is_sent = vertices.eq(-1)
        first_sent = torch.where(
            is_sent.any(dim=1),
            is_sent.long().argmax(dim=1),
            torch.full((n,), depth_p1, device=device, dtype=torch.long),
        )
        valid_len = (2 * first_sent - 1).clamp(min=0)          # (n,)
        ii, jj = self._pair_template_for(Lint, device)         # (P,)
        center = seq[:, ii].reshape(-1)
        context = seq[:, jj].reshape(-1)
        keep = (
            (ii.unsqueeze(0) < valid_len.unsqueeze(1))
            & (jj.unsqueeze(0) < valid_len.unsqueeze(1))
        )                                                       # (n, P)
        if self.sample > 0:
            # Frequent-token subsampling: independently drop each token *occurrence*
            # in the walk with prob 1 - p_keep(token), then drop every skip-gram
            # pair that touches a dropped occurrence. (No window-shift — the
            # standard vectorised approximation; still thins the hot predicates.)
            p_keep = self._get_p_keep(device)                   # (vocab,)
            keep_tok = torch.rand((n, Lint), device=device) < p_keep[seq.clamp(min=0)]  # (n, Lint)
            keep = keep & keep_tok[:, ii] & keep_tok[:, jj]
        keep = keep.reshape(-1)
        return center[keep], context[keep]

    def _training_step_sequences(self, vertices, predicates):
        center, context = self._expand_skipgram(vertices, predicates)
        if center.numel() == 0:
            return torch.zeros((), device=center.device, requires_grad=True)
        negative = self._sample_negatives(center.size(0), center.device)
        loss = self(center, context, negative)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        a, b = batch
        if a.dim() == 2:                       # (vertices, predicates) — walk-sequence path
            return self._training_step_sequences(a, b)
        center, context = a, b                 # legacy (center, context) pair path
        negative = self._sample_negatives(center.size(0), self.device)
        loss = self(center, context, negative)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Classic word2vec: plain SGD + linear LR decay (gensim's alpha schedule).
        # SGD also has zero adaptive state ⇒ the optimiser is O(touched rows) not
        # O(vocab) when use_sparse=True. `optimizer="sparseadam"` swaps in
        # torch.optim.SparseAdam (adaptive per-parameter step sizes; requires all
        # params to produce sparse grads — true when use_sparse=True; O(vocab)
        # moment state, single-GPU only). The LR-decay schedule below applies to
        # either optimizer (for SparseAdam it's a mild warmdown — a small start LR
        # like 1e-3–5e-3 is what you'd pass via `learning_rate`).
        if self.optimizer_name == "sparseadam" and self.use_sparse:
            opt = torch.optim.SparseAdam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "sparseadam":
            opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)   # dense fallback
        else:
            opt = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        if self.lr_decay != "linear" or self.epochs <= 1:
            return opt
        end_factor = max(self.min_learning_rate / self.learning_rate, 1e-12)
        if self.steps_per_epoch is not None:
            opt_steps_per_epoch = -(-int(self.steps_per_epoch) // self.accumulate_grad_batches)  # ceil
            opt_steps_per_epoch = max(1, opt_steps_per_epoch)
            # last opt-step fires at min_lr (LinearLR clamps after total_iters), matching gensim's alpha clamp
            total_iters = max(1, self.epochs * opt_steps_per_epoch - 1)
            sched = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1.0, end_factor=end_factor, total_iters=total_iters
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1.0, end_factor=end_factor, total_iters=max(1, self.epochs - 1)
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

    def on_after_backward(self):
        # Gradient clipping for the two embedding tables. With the `sum`-over-the-
        # batch loss, a hot predicate token (RDF2Vec walk corpora have a few
        # predicates that appear in a large fraction of every batch's pairs)
        # accumulates a huge summed gradient → its SGD step ≫ init magnitude →
        # fp32 overflow. Two modes (config `grad_clip_mode`):
        #   "element": clamp each gradient component to [-c, +c] — bounds a single
        #     step's per-component move at lr*c; the long tail of rarely-touched
        #     rows (small gradients) is essentially unclipped.
        #   "row":     rescale each touched row's gradient so ‖row‖₂ ≤ c — a
        #     geometry-aware cap on how far a row drifts per step that keeps the
        #     gradient *direction* intact (element clamping distorts it toward the
        #     box corners). For sparse grads this acts on the coalesced per-row
        #     sums; rows with ‖·‖ ≤ c are left untouched.
        # Lightning's built-in `gradient_clip_val` uses `.clamp_`, which doesn't
        # support sparse tensors — hence this hook on the coalesced sparse grad.
        c = self.grad_clip_value
        if not c:
            return
        row_mode = self.grad_clip_mode == "row"
        for emb in (self.in_embeddings, self.output_embeddings):
            g = emb.weight.grad
            if g is None:
                continue
            if g.is_sparse:
                g = g.coalesce()              # sum duplicate-index contributions, THEN clip the per-row sum
                v = g._values()
                if row_mode:
                    # v: (nnz_rows, dim) — one summed gradient row per touched token.
                    norms = v.norm(dim=1, keepdim=True)
                    v.mul_((c / norms.clamp(min=1e-12)).clamp(max=1.0))
                else:
                    v.clamp_(-c, c)
                emb.weight.grad = g
            else:
                if row_mode:
                    norms = g.norm(dim=1, keepdim=True)
                    g.mul_((c / norms.clamp(min=1e-12)).clamp(max=1.0))
                else:
                    g.clamp_(-c, c)


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
        """Entity-level embedding table used downstream (the input/syn0 table)."""
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
