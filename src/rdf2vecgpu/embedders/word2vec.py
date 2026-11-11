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
        self._neg_cdf = None
        self._pair_templates = {}

        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=use_sparse)
        self.output_embeddings = nn.Embedding(
            vocab_size, embedding_dim, sparse=use_sparse
        )

        # as word2vec and gensim: small random input vectors, zero output vectors
        bound = 1.0 / embedding_dim
        nn.init.uniform_(self.in_embeddings.weight, -bound, bound)
        nn.init.zeros_(self.output_embeddings.weight)

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

    def _get_neg_cdf(self, device):
        """Cumulative count^0.75 distribution over the vocabulary, built once:
        drawing from it costs log(vocabulary) per sample, whereas
        torch.multinomial re-reads the whole distribution on every call."""
        if self._neg_cdf is None or self._neg_cdf.device != device:
            weights = self._token_counts.clamp(min=0).pow(0.75)
            cdf = torch.cumsum(weights, dim=0)
            self._neg_cdf = (cdf / cdf[-1]).to(device)  # float64: exact for large vocabularies
        return self._neg_cdf

    def _sample_negatives(self, n_rows, device):
        """(n_rows, neg_samples) negative token ids."""
        if self.negative_sampling == "unigram":
            cdf = self._get_neg_cdf(device)
            draws = torch.rand(n_rows * self.neg_samples, device=device, dtype=cdf.dtype)
            # the first token whose cumulative share exceeds the draw; tokens
            # with count 0 add nothing to the CDF and are never drawn
            tokens = torch.searchsorted(cdf, draws, right=True).clamp_(max=self.vocab_size - 1)
            return tokens.view(n_rows, self.neg_samples)
        return torch.randint(0, self.vocab_size, (n_rows, self.neg_samples), device=device)

    def forward(self, center, context, negative, negative_mask=None):
        """Summed negative-sampling loss. `negative_mask` (batch, neg), if
        given, leaves out negatives that are the positive's own token, as
        word2vec does: otherwise a frequent token is pushed towards and away
        from the same input in one step."""
        center_embed = self.in_embeddings(center)  # (batch, embed)
        context_embed = self.output_embeddings(context)  # (batch, embed)
        negative_embed = self.output_embeddings(negative)  # (batch, neg, embed)

        pos_logits = (center_embed * context_embed).sum(dim=1)
        pos_loss = self.loss_fn(pos_logits, torch.ones_like(pos_logits))

        neg_score = torch.bmm(negative_embed, center_embed.unsqueeze(-1)).squeeze(-1)
        neg_loss = self.loss_fn(neg_score, torch.zeros_like(neg_score))
        if negative_mask is not None:
            neg_loss = neg_loss * negative_mask
        return torch.sum(pos_loss + neg_loss.sum(dim=1))

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

    def _expand_skipgram(self, vertices, predicates, with_offsets=False):
        """All skip-gram (center, context) pairs of a batch of walks, built on
        the GPU, and with ``with_offsets`` each pair's offset (context
        position minus center position, in [-window, window] without 0).
        A walk is the sequence v0, p0, v1, p1, ..., cut where its
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
        if with_offsets:
            offsets = (jj - ii).repeat(n)
            return center[keep], context[keep], offsets[keep]
        return center[keep], context[keep]

    def training_step(self, batch, batch_idx):
        first, second = batch
        if first.dim() == 2:
            center, context = self._expand_skipgram(first, second)
        else:
            center, context = first.long(), second.long()
        # A batch whose pairs were all subsampled away (possible for tiny
        # batches) runs through the same computation on empty tensors: its zero
        # loss stays connected to the embeddings, so every DDP rank still joins
        # the gradient all-reduce (a detached zero would leave the others waiting).
        negative = self._sample_negatives(center.size(0), center.device)
        loss = self(center, context, negative, negative != context.unsqueeze(1))
        self.log("train_loss", loss / max(1, center.size(0)), prog_bar=True)
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


class CBOWNegativeSampling(SkipGram):
    """CBOW with negative sampling, trained like `SkipGram` (summed loss, SGD
    with linear decay, clipping, subsampling, unigram negatives), as gensim's
    ``sg=0`` with ``cbow_mean=1`` does.

    Each position of a walk is one example: the mean of the input embeddings
    of the tokens within ``window`` of it predicts the token there. Batches
    are ``(vertices, predicates)`` walk rows; the examples are built on the
    GPU.
    """

    def _context_template_for(self, length, device):
        """(length, 2 * window) positions around each centre position, -1
        where the window runs past the sequence; cached per length."""
        key = ("cbow", length)
        cached = self._pair_templates.get(key)
        if cached is not None and cached.device == device:
            return cached
        offsets = [d for d in range(-self.window, self.window + 1) if d != 0]
        rows = [[i + d if 0 <= i + d < length else -1 for d in offsets] for i in range(length)]
        template = torch.as_tensor(rows, dtype=torch.long, device=device)
        self._pair_templates[key] = template
        return template

    def _expand_cbow(self, vertices, predicates):
        """(context, mask, center) of every example in a batch of walks:
        context (m, 2 * window) token ids, mask (m, 2 * window) which of them
        count, center (m,) the token to predict. Positions past a walk's end
        and token occurrences dropped by subsampling take part in nothing;
        examples without any context are left out."""
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
        positions = torch.arange(length, device=device)
        live = positions.unsqueeze(0) < valid_len.unsqueeze(1)  # (n, length)
        if self.sample > 0:
            p_keep = self._get_p_keep(device)
            live &= torch.rand((n, length), device=device) < p_keep[seq.clamp(min=0)]
        template = self._context_template_for(length, device)  # (length, 2w)
        in_range = template >= 0
        context = seq[:, template.clamp(min=0)]  # (n, length, 2w)
        mask = in_range.unsqueeze(0) & live[:, template.clamp(min=0)]
        keep = live & mask.any(dim=2)  # centre counts and has some context
        return context[keep], mask[keep], seq[keep]

    def _cbow_loss(self, context, mask, center):
        weights = mask.to(self.in_embeddings.weight.dtype)
        # masked slots may hold the -1 padding: look up row 0 there, weight 0
        summed = (self.in_embeddings(context.clamp(min=0)) * weights.unsqueeze(-1)).sum(dim=1)
        hidden = summed / weights.sum(dim=1, keepdim=True)  # (m, dim)
        positive = (hidden * self.output_embeddings(center)).sum(dim=1)
        negative = self._sample_negatives(center.size(0), center.device)
        negative_scores = torch.bmm(self.output_embeddings(negative), hidden.unsqueeze(-1)).squeeze(-1)
        loss = self.loss_fn(positive, torch.ones_like(positive))
        # leave out negatives that are the token to predict, see SkipGram.forward
        negative_loss = self.loss_fn(negative_scores, torch.zeros_like(negative_scores))
        loss = loss + (negative_loss * (negative != center.unsqueeze(1))).sum(dim=1)
        return loss.sum()

    def training_step(self, batch, batch_idx):
        vertices, predicates = batch
        context, mask, center = self._expand_cbow(vertices, predicates)
        # empty batches stay connected to the embeddings, see SkipGram.training_step
        loss = self._cbow_loss(context, mask, center)
        self.log("train_loss", loss / max(1, center.size(0)), prog_bar=True)
        return loss


class OrderAwareSkipGram(SkipGram):
    """Order-aware skip-gram (structured skip-gram, as in RDF2Vec_oa): one
    output embedding per relative position, so the model learns not only
    which tokens occur near a token but where (for walks: which hop, and
    whether a vertex or a predicate). Trained like `SkipGram`, from batches
    of ``(vertices, predicates)`` walk rows.

    The 2 * ``window`` output tables (offsets -window..-1, 1..window) are one
    sparse embedding of 2 * window * vocab_size rows; row
    ``offset_index * vocab_size + token``. The returned embeddings are the
    input table, as for `SkipGram`.
    """

    def __init__(self, vocab_size, embedding_dim, neg_samples, window=5, **kwargs):
        super().__init__(vocab_size, embedding_dim, neg_samples, window=window, **kwargs)
        self.output_embeddings = nn.Embedding(
            2 * window * vocab_size, embedding_dim, sparse=self.use_sparse
        )
        nn.init.zeros_(self.output_embeddings.weight)

    def _offset_rows(self, offsets, tokens):
        """Rows of the output table: offsets -w..-1 map to 0..w-1, 1..w to w..2w-1."""
        index = torch.where(offsets < 0, offsets + self.window, offsets + self.window - 1)
        return index * self.vocab_size + tokens

    def training_step(self, batch, batch_idx):
        first, second = batch
        if first.dim() != 2:
            raise ValueError("order-aware skip-gram trains on walks, not on flat (center, context) pairs")
        center, context, offsets = self._expand_skipgram(first, second, with_offsets=True)
        # empty batches stay connected to the embeddings, see SkipGram.training_step
        negatives = self._sample_negatives(center.size(0), center.device)
        loss = self(
            center,
            self._offset_rows(offsets, context),
            self._offset_rows(offsets.unsqueeze(1), negatives),
            negatives != context.unsqueeze(1),
        )
        self.log("train_loss", loss / max(1, center.size(0)), prog_bar=True)
        return loss


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
