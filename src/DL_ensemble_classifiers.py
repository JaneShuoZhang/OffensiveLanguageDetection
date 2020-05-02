import pandas as pd
import numpy as np
from preprocessing import process_train_data, process_test_data, process_trial_data
from utils import load_train_data, load_test_data_a, load_trial_data, change_to_binary, RESULT_FOLDER, progress_bar
from feature_embedding import generate_glove_embedding, build_LSTM_dataset
from sklearn.metrics import classification_report, accuracy_score, f1_score
import os
import time
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch_model_base import TorchModelBase
from LSTM_classifiers import TorchLSTMDataset


__author__ = "Shuo Zhang"
__version__ = "CS224u, Stanford, Spring 2020"


class TorchLSTM_CNNClassifierModel(nn.Module):
    def __init__(self,
                 embed_dim,
                 embedding,
                 hidden_dim,
                 output_dim,
                 out_channels,
                 kernel_sizes,
                 dropout_prob,
                 device,
                 bidirectional=True):
        super(TorchLSTM_CNNClassifierModel, self).__init__()
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
            cnn_input_dim = hidden_dim * 2
        else:
            cnn_input_dim = hidden_dim
        self.conv_kernels = nn.ModuleList([nn.Conv2d(1, out_channels, (kernel_size, cnn_input_dim)) for kernel_size in kernel_sizes])
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier_layer = nn.Linear(len(kernel_sizes)*out_channels, output_dim)

    def forward(self, X, seq_lengths):
        lstm_output = self.LSTM_forward(X, seq_lengths, self.rnn)   # (batch_size, seq_len, hidden_dim*num_directions)
        state = self.CNN_forward(lstm_output)
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

        outputs, _ = rnn(embs)
        output_sequence, sequence_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)    

        _, unsort_idx = sort_idx.sort(0)
        output_sequence = output_sequence[unsort_idx]
        return output_sequence

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)     # (batch_size, out_channels, sequence_length - kernal_size + 1, 1)
        activation = F.relu(conv_out.squeeze(3))     # (batch_size, out_channels, sequence_length - kernal_size + 1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)   # (batch_size, out_channels)
        return max_out

    def CNN_forward(self, X):
        #X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        X = X.to(self.device, non_blocking=True)
        input = X.unsqueeze(1)    # (batch_size, 1, seq_length, hidden_dim*num_directions)        
        max_outs = [self.conv_block(input, conv) for conv in self.conv_kernels]
        all_out = torch.cat(max_outs, 1)     # (batch_size, num_kernels*out_channels)
        state = self.dropout(all_out)
        return state


class TorchLSTM_CNNClassifier(TorchModelBase):
    """LSTM + CNN ensemble model for classification problems.
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
        Dimensionality of the LSTM hidden layer.
    bidirectional : bool
        If True, then the final hidden states from passes in both
        directions are used.
    out_channels : int
        Number of CNN kernals for each size.
    kernel_sizes : list of int
        Sizes of CNN kernels.
    dropout_prob : float
        Dropout probability for the dropout layer before final classification layer.
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
                 out_channels,
                 kernel_sizes,
                 dropout_prob,
                 embedding,
                 embed_dim=300,
                 bidirectional=True,
                 **kwargs):
        self.vocab = vocab
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.dropout_prob = dropout_prob
        self.embedding = embedding
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        super(TorchLSTM_CNNClassifier, self).__init__(**kwargs)
        self.params += ['embedding', 'embed_dim', 'bidirectional', 'kernel_sizes', 'out_channels', 'dropout_prob']
        # The base class has this attribute, but this model doesn't,
        # so we remove it to avoid misleading people:
        delattr(self, 'hidden_activation')
        self.params.remove('hidden_activation')

    def build_dataset(self, X, y):
        X, seq_lengths = self._prepare_dataset(X)
        return TorchLSTMDataset(X, seq_lengths, y)

    def build_graph(self):
        return TorchLSTM_CNNClassifierModel(
            embedding=self.embedding,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_classes_,
            bidirectional=self.bidirectional,
            out_channels=self.out_channels,
            kernel_sizes=self.kernel_sizes,
            dropout_prob=self.dropout_prob,
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


def BiLSTM_CNN_model(embed_dim=50, batch_size=1024, max_iter=10, hidden_dim=50, bidirectional=True, out_channels=30, kernel_sizes=[3,4,5], dropout_prob=0.1):
    start_time = time.time()
    vocab, embedding = generate_glove_embedding(embed_dim)

    train_data = load_train_data()
    train_data = process_train_data(train_data)
    X_train, y_train = build_LSTM_dataset(train_data, 128)

    mod = TorchLSTM_CNNClassifier(
        vocab=vocab,
        embedding=embedding,
        embed_dim=embed_dim,
        max_iter=max_iter,
        bidirectional=bidirectional,
        hidden_dim=hidden_dim,
        out_channels=out_channels,
        kernel_sizes=kernel_sizes,
        dropout_prob=dropout_prob,
        batch_size=batch_size)

    mod.fit(X_train, y_train)

    test_data = load_test_data_a()
    test_data = process_test_data(test_data)
    X_test, y_test = build_LSTM_dataset(test_data, 128)

    predictions = mod.predict(X_test)
    test_data['prediction'] = np.array(predictions)
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
    output_file_path = os.path.join(RESULT_FOLDER, "BiLSTM_CNN_{}-embedding_{}-batchsize_{}-hidden_{}-filters_{}-iter_prediction.csv". \
        format(embed_dim, batch_size, hidden_dim, out_channels, max_iter))
    test_data.to_csv(output_file_path, index=False)

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    f1_macro = f1_score(change_to_binary(y_test), change_to_binary(predictions), average='macro')
    print("BiLSTM+CNN embedding dim: {}, batch size: {}, hiddend dim: {}, out channels: {}, max_iter: {}, dropout: {}, macro f1 score: {}" \
        .format(embed_dim, batch_size, hidden_dim, out_channels, max_iter, dropout_prob, f1_macro))

    end_time = time.time()
    print("Finish BiLSTM+CNN in {} mins.".format((end_time - start_time)/60))
    return f1_macro


if __name__ == '__main__':
    BiLSTM_CNN_model()