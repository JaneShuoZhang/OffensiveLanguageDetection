import pandas as pd
import numpy as np
from preprocessing import process_train_data, process_test_data, process_trial_data
from utils import load_train_data, load_test_data_a, load_trial_data, change_to_binary
from feature_embedding import generate_glove_embedding, build_LSTM_dataset
from sklearn.metrics import classification_report, accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.utils.data
from torch_model_base import TorchModelBase
from utils import progress_bar

__author__ = "zhexuan"
__version__ = "CS224u, Stanford, Spring 2020"

class TorchLSTMDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, seq_lengths, y):
        assert len(sequences) == len(y)
        assert len(sequences) == len(seq_lengths)
        self.sequences = sequences
        self.seq_lengths = seq_lengths
        self.y = y

    @staticmethod
    def collate_fn(batch):
        X, seq_lengths, y = zip(*batch)
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return X, seq_lengths, y

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (self.sequences[idx], self.seq_lengths[idx], self.y[idx])


class TorchLSTMClassifierModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 embedding,
                 hidden_dim,
                 output_dim,
                 bidirectional,
                 device):
        super(TorchLSTMClassifierModel, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional

        embedding = torch.tensor(embedding, dtype=torch.float)
        self.embedding = nn.Embedding.from_pretrained(embedding)

        # Graph
        self.rnn = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional)
        if bidirectional:
            classifier_dim = hidden_dim * 2
        else:
            classifier_dim = hidden_dim
        self.classifier_layer = nn.Linear(classifier_dim, output_dim)

    def forward(self, X, seq_lengths):
        state = self.LSTM_forward(X, seq_lengths, self.rnn)
        logits = self.classifier_layer(state)
        return logits

    def LSTM_forward(self, X, seq_lengths, rnn):
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        X = X.to(self.device, non_blocking=True)
        seq_lengths = seq_lengths.to(self.device)
        seq_lengths, sort_idx = seq_lengths.sort(0, descending=True)
        X = X[sort_idx]

        embs = self.embedding(X)
        embs = torch.nn.utils.rnn.pack_padded_sequence(
            embs, batch_first=True, lengths=seq_lengths)

        outputs, state = rnn(embs)

        state = state[0].squeeze(0)

        if self.bidirectional:
            state = torch.cat((state[0], state[1]), dim=1)

        _, unsort_idx = sort_idx.sort(0)
        state = state[unsort_idx]
        return state


class TorchLSTMClassifier(TorchModelBase):
    """LSTM-based Recurrent Neural Network for classification problems.
    The network will work for any kind of classification task.

    Parameters
    ----------
    vocab : list of str
        This should be the vocabulary. It needs to be aligned with
         `embedding` in the sense that the ith element of vocab
        should be represented by the ith row of `embedding`.
    embedding : np.array or None
        Each row represents a word in `vocab`, as described above.
    embed_dim : int
        Dimensionality for the initial embeddings.
    hidden_dim : int
        Dimensionality of the hidden layer.
    bidirectional : bool
        If True, then the final hidden states from passes in both
        directions are used.
    max_iter : int
        Maximum number of training epochs.
    eta : float
        Learning rate.
    optimizer : PyTorch optimizer
        Default is `torch.optim.Adam`.
    l2_strength : float
        L2 regularization strength. Default 0 is no regularization.
    device : 'cpu' or 'cuda'
        The default is to use 'cuda' iff available
    warm_start : bool
        If True, calling `fit` will resume training with previously
        defined trainable parameters. If False, calling `fit` will
        reinitialize all trainable parameters. Default: False.

    """

    def __init__(self,
                 vocab,
                 embedding,
                 embed_dim=300,
                 bidirectional=False,
                 **kwargs):
        self.vocab = vocab
        self.embedding = embedding
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        super(TorchLSTMClassifier, self).__init__(**kwargs)
        self.params += ['embedding', 'embed_dim', 'bidirectional']
        # The base class has this attribute, but this model doesn't,
        # so we remove it to avoid misleading people:
        delattr(self, 'hidden_activation')
        self.params.remove('hidden_activation')

    def build_dataset(self, X, y):
        X, seq_lengths = self._prepare_dataset(X)
        return TorchLSTMDataset(X, seq_lengths, y)

    def build_graph(self):
        return TorchLSTMClassifierModel(
            vocab_size=len(self.vocab),
            embedding=self.embedding,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_classes_,
            bidirectional=self.bidirectional,
            device=self.device)

    def fit(self, X, y, **kwargs):
        """Standard `fit` method.

        Parameters
        ----------
        X : np.array
        y : array-like
        kwargs : dict
            For passing other parameters. If 'X_dev' is included,
            then performance is monitored every 10 epochs; use
            `dev_iter` to control this number.

        Returns
        -------
        self

        """
        # Incremental performance:
        X_dev = kwargs.get('X_dev')
        if X_dev is not None:
            dev_iter = kwargs.get('dev_iter', 10)
        # Data prep:
        self.classes_ = sorted(set(y))
        self.n_classes_ = len(self.classes_)
        class2index = dict(zip(self.classes_, range(self.n_classes_)))
        y = [class2index[label] for label in y]
        dataset = self.build_dataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn)

        # Graph:
        if not self.warm_start or not hasattr(self, "model"):
            self.model = self.build_graph()
        self.model.to(self.device)
        self.model.train()
        # Make sure this value is up-to-date; self.`model` might change
        # it if it creates an embedding:
        self.embed_dim = self.model.embed_dim
        # Optimization:
        loss = nn.CrossEntropyLoss()
        optimizer = self.optimizer(
            self.model.parameters(),
            lr=self.eta,
            weight_decay=self.l2_strength)
        # Train:
        for iteration in range(1, self.max_iter + 1):
            epoch_error = 0.0
            for X_batch, batch_seq_lengths, y_batch in dataloader:
                y_batch = y_batch.to(self.device, non_blocking=True)
                batch_preds = self.model(X_batch, batch_seq_lengths)
                err = loss(batch_preds, y_batch)
                epoch_error += err.item()
                # Backprop:
                optimizer.zero_grad()
                err.backward()
                optimizer.step()
            # Incremental predictions where possible:
            if X_dev is not None and iteration > 0 and iteration % dev_iter == 0:
                self.dev_predictions[iteration] = self.predict(X_dev)
                self.model.train()
            self.errors.append(epoch_error)
            progress_bar("Finished epoch {} of {}; error is {}".format(
                iteration, self.max_iter, epoch_error))
        return self

    def predict_proba(self, X):
        """Predicted probabilities for the examples in `X`.

        Parameters
        ----------
        X : np.array

        Returns
        -------
        np.array with shape (len(X), self.n_classes_)

        """
        self.model.eval()
        with torch.no_grad():
            self.model.to(self.device)
            X, seq_lengths = self._prepare_dataset(X)
            preds = self.model(X, seq_lengths)
            preds = torch.softmax(preds, dim=1).cpu().numpy()
            return preds

    def predict(self, X):
        """Predicted labels for the examples in `X`. These are converted
        from the integers that PyTorch needs back to their original
        values in `self.classes_`.

        Parameters
        ----------
        X : np.array

        Returns
        -------
        list of length len(X)

        """
        probs = self.predict_proba(X)
        return [self.classes_[i] for i in probs.argmax(axis=1)]

    def _prepare_dataset(self, X):
        """Internal method for preprocessing a set of examples.
        `X` is transformed into a list of lists of indices. And
        we measure the lengths of the sequences in `X`.

        Parameters
        ----------
        X : list of lists of tokens

        Returns
        -------
        list of lists of ints
        and `torch.LongTensor` of sequence lengths.

        """
        new_X = []
        seq_lengths = []

        index = dict(zip(self.vocab, range(len(self.vocab))))
        unk_index = index['$UNK']
        for ex in X:
            seq = [index.get(w, unk_index) for w in ex]
            seq = torch.tensor(seq, dtype=torch.long)
            new_X.append(seq)
            seq_lengths.append(len(seq))

        return new_X, torch.LongTensor(seq_lengths)


def LSTM_model(embed_dim=300):
    vocab, embedding = generate_glove_embedding(embed_dim)

    train_data = load_train_data()
    train_data = process_train_data(train_data)
    X_train, y_train = build_LSTM_dataset(train_data, 128)

    mod = TorchLSTMClassifier(
        vocab=vocab,
        embedding=embedding,
        embed_dim=embed_dim,
        max_iter=100,
        bidirectional=False,
        hidden_dim=50)

    print(mod)

    mod.fit(X_train, y_train)

    test_data = load_test_data_a()
    test_data = process_test_data(test_data)
    X_test, y_test = build_LSTM_dataset(test_data, 128)

    predictions = mod.predict(X_test)

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    F1_score = f1_score(change_to_binary(y_test), change_to_binary(predictions))

    print("LSTM embedding dim: {}, f1 score: {}".format(embed_dim, F1_score))
    return F1_score


if __name__ == '__main__':
   LSTM_model(300)
