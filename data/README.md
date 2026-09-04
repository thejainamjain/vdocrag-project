# Data

Run this on the MacBook (CPU is fine, it's just downloading + saving images):

```bash
python data/download_opendocvqa.py --configs chartqa --max_examples 500
```

This creates `data/processed/`:
- `images/` — page images
- `corpus.jsonl` — `{id, image}` — the retrieval corpus
- `test_queries.jsonl` — `{id, query}` — questions to retrieve/answer for
- `retriever_train.jsonl` — `{query, pos_image}` — contrastive training pairs
- `generator_train.jsonl` — `{query, context_images, answer}` — SFT pairs

**Important**: field names differ slightly across the 9 OpenDocVQA sub-datasets
(ChartQA, SlideVQA, InfoVQA, DUDE, etc). Before training, run:

```python
from datasets import load_dataset
ds = load_dataset("NTT-hil-insight/OpenDocVQA", "chartqa", split="test")
print(ds[0])
```

...and adjust the field-mapping lines in `download_opendocvqa.py` if a config
you add uses different key names for the answer / image id.

Start with just `chartqa` (single config, ~few hundred examples) to keep the
first end-to-end run small and fast to debug on the MacBook before moving
training to the RTX GPU.
