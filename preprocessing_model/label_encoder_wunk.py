from sklearn.preprocessing import LabelEncoder
import numpy as np
UNK_TOKEN = '<UNK>'
class LabelEncoderWithUnk(LabelEncoder):
    def __init__(self, unk_token=UNK_TOKEN):
        super().__init__()
        self.unk_token = unk_token
        self.token_to_index = None
    def fit(self, y):
        unique_tokens = sorted(set(y))
        super().fit(unique_tokens)
        if self.unk_token not in self.classes_:
            self.classes_ = np.append(self.classes_, self.unk_token)
        self.token_to_index = {token: idx for idx, token in enumerate(self.classes_)}
        return self
    def transform(self, y):
        unknown_index = self.token_to_index.get(self.unk_token, -1)
        return np.array([self.token_to_index.get(token, unknown_index) for token in y])
    def inverse_transform(self, y):
        """Inverse transform labels back to original tokens."""
        index_to_token = {idx: token for token, idx in self.token_to_index.items()}
        return np.array([index_to_token.get(index, self.unk_token) for index in y])
