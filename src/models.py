from dataclasses import dataclass
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

@dataclass
class TfidfComplaintModel:
    pipeline: Pipeline

    @classmethod
    def build(cls, max_features=20000, ngram_range=(1, 2)):
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=True,
                min_df=1
            )),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            ))
        ])
        return cls(pipe)

    def fit(self, x, y):
        self.pipeline.fit(x, y)
        return self

    def predict(self, x):
        return self.pipeline.predict(x)

    def predict_proba(self, x):
        return self.pipeline.predict_proba(x)

    def save(self, path):
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path):
        return cls(joblib.load(path))
