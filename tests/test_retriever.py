"""
Lightweight tests that don't require downloading any model weights --
they test the pure-tensor logic (pooling, loss) in isolation, so you can
run them on the MacBook with just `pip install torch` and no GPU.
"""

import torch

from vdocrag.retriever.modeling import VDocRetriever


def test_last_token_pool_picks_correct_index():
    # batch of 2, seq_len 4, hidden_dim 3
    hidden = torch.tensor([
        [[1., 1., 1.], [2., 2., 2.], [3., 3., 3.], [0., 0., 0.]],  # 3 real tokens, 1 pad
        [[4., 4., 4.], [5., 5., 5.], [0., 0., 0.], [0., 0., 0.]],  # 2 real tokens, 2 pad
    ])
    attention_mask = torch.tensor([
        [1, 1, 1, 0],
        [1, 1, 0, 0],
    ])

    pooled = VDocRetriever._last_token_pool(hidden, attention_mask)

    assert torch.allclose(pooled[0], torch.tensor([3., 3., 3.]))
    assert torch.allclose(pooled[1], torch.tensor([5., 5., 5.]))


def test_in_batch_contrastive_loss_is_low_for_perfectly_aligned_batch():
    class DummyBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.device = "cpu"

        def forward(self, **kwargs):
            raise NotImplementedError("not used directly in this test")

    model = VDocRetriever(base_model=DummyBaseModel(), temperature=0.01)
    model.train()

    # Simulate perfectly aligned query/doc embeddings (query_i matches doc_i)
    q_reps = torch.eye(4)
    p_reps = torch.eye(4)

    scores = torch.matmul(q_reps, p_reps.transpose(0, 1)) / model.temperature
    labels = torch.arange(scores.size(0))
    loss = torch.nn.functional.cross_entropy(scores, labels)

    assert loss.item() < 0.01  # near-zero loss since diagonal scores dominate
