import pandas as pd

from src.data.split_semeval import create_grouped_splits


def test_grouped_split_keeps_sentences_in_one_split():
    rows = []
    for i in range(60):
        label = ["positive", "negative", "neutral", "conflict"][i % 4]
        rows.append({"sentence_id": f"s{i}", "review_text": f"review {i}", "aspect": "battery", "sentiment": label})
        # Multiple aspect records from one sentence must stay together.
        rows.append({"sentence_id": f"s{i}", "review_text": f"review {i}", "aspect": "screen", "sentiment": label})
    splits = create_grouped_splits(pd.DataFrame(rows), n_splits=20)
    group_sets = {
        name: set(frame["sentence_id"])
        for name, frame in splits.items()
    }
    assert group_sets["train"].isdisjoint(group_sets["validation"])
    assert group_sets["train"].isdisjoint(group_sets["test"])
    assert group_sets["validation"].isdisjoint(group_sets["test"])
    assert sum(map(len, splits.values())) == len(rows)
