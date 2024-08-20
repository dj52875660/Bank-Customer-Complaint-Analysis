import re

import numpy as np
import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import BertTokenizer


class TextDataset(Dataset):
    def __init__(
        self,
        root: str,
        seed: int = 42,
        mode: str = "train",
        test_ratio: float = 0.1,
        val_ratio: float = 0.2,
        max_length: int = 128,
        N: int = 64,  # number of top features
        NTA: int = 128,  # new text length limit
    ):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.seed = seed
        self.mode = mode
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.max_length = max_length
        self.N = N
        self.NTA = NTA
        self.df = pd.read_csv(root)

        self.label_encoder = LabelEncoder()
        self.df["encoded_product"] = self.label_encoder.fit_transform(
            self.df["product"]
        )
        self.num_classes = len(self.label_encoder.classes_)

        self.df = self.preprocess_dataframe(self.df)
        # self.sITFL = self.get_feature_importance_list(self.df)

        # self.df["processed_narrative"] = self.df["narrative"].apply(
        #     lambda x: self.extract_important_tokens(x, self.sITFL, self.NTA)
        # )

        x = self.df["narrative"].values
        # x = self.df["processed_narrative"].values
        y = self.df["encoded_product"].values

        x_train_val, x_test, y_train_val, y_test = train_test_split(
            x, y, test_size=self.test_ratio, random_state=self.seed, stratify=y
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val,
            y_train_val,
            test_size=self.val_ratio,
            random_state=self.seed,
            stratify=y_train_val,
        )

        if mode == "test":
            self.x, self.y = x_test, y_test
        elif mode == "train":
            self.x, self.y = x_train, y_train
        else:
            self.x, self.y = x_val, y_val

    def preprocess_dataframe(self, df):
        df = df.dropna()
        df["narrative"] = df["narrative"].str.lower()
        df["narrative"] = df["narrative"].apply(self.remove_url)
        df = df.reset_index()

        return df

    def remove_url(self, text):
        pattern = re.compile(r"https?://\S+|www\.\S+")
        return pattern.sub(r"", text)

    def get_feature_importance_list(self, df):
        vectorizer = CountVectorizer(min_df=0.1, stop_words="english")
        X = vectorizer.fit_transform(df["narrative"])
        y = df["encoded_product"]

        mi = mutual_info_classif(X, y)
        top_N_features = np.argsort(mi)[-self.N :]

        X_top_N = X[:, top_N_features].toarray()
        clf = GradientBoostingClassifier()
        clf.fit(X_top_N, y)

        result = permutation_importance(clf, X_top_N, y, n_repeats=3, random_state=42)
        importance = result.importances_mean

        sorted_indices = np.argsort(importance)[::-1]
        sorted_features = top_N_features[sorted_indices]

        feature_names = np.array(vectorizer.get_feature_names_out())[sorted_features]
        sITFL = feature_names.tolist()

        return sITFL

    def extract_important_tokens(self, text, sITFL, NTA):
        tokens = word_tokenize(text)
        selected_tokens = []

        for feature in sITFL:
            for i, token in enumerate(tokens):
                if re.fullmatch(feature, token, flags=re.IGNORECASE):
                    start = max(0, i - 5)
                    end = min(len(tokens), i + 5)
                    selected_tokens.extend(tokens[start:end])
                    tokens = tokens[:start] + tokens[end:]
                    break

            if len(selected_tokens) >= NTA:
                break

        return " ".join(selected_tokens[:NTA])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        narrative = self.x[idx]
        label = self.y[idx]

        encoding = self.tokenizer.encode_plus(
            narrative,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        sample = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }
        return sample
