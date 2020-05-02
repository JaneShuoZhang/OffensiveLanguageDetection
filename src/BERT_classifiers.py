import pandas as pd
import numpy as np
import random
import os
from preprocessing import process_train_data, process_test_data, process_trial_data
from utils import load_train_data, load_test_data_a, load_trial_data, change_to_binary
from sklearn.metrics import classification_report, accuracy_score, f1_score
import torch
from torch.utils.data import TensorDataset,DataLoader
from torch_model_base import TorchModelBase
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, AdamW, BertConfig
from utils import progress_bar, RESULT_FOLDER, format_time
import time
import datetime

class TorchBertClassifier(TorchModelBase):
    """LSTM-based Recurrent Neural Network for classification problems.
    The network will work for any kind of classification task.

    Parameters
    ----------
    tokenizer: Bert pretrained Tokenizer
    model: interfaces of BERT mode
    max_set_length: int
        Maximum sentence length. Pad & truncate all sentences to this value.
    max_iter : int
        Maximum number of training epochs.
    eta : float
        Learning rate.
    eps : float
        A very small number to prevent any division by zero in the implementation.
    optimizer : PyTorch optimizer
        Default is `torch.optim.Adam`.
    l2_strength : float
        L2 regularization strength. Default 0 is no regularization.
    batch_size: int
        Default = 1028
    warm_start : bool
        If True, calling `fit` will resume training with previously
        defined trainable parameters. If False, calling `fit` will
        reinitialize all trainable parameters. Default: False.

    """

    def __init__(self,
                 tokenizer,
                 model,
                 max_set_length=128,
                 eps=1e-8,
                 **kwargs):
        self.tokenizer = tokenizer
        self.model = model
        self.max_set_length = max_set_length
        self.eps = eps
        super(TorchBertClassifier, self).__init__(**kwargs)
        self.params += ['max_set_length', 'eps']
        # The base class has this attribute, but this model doesn't,
        # so we remove it to avoid misleading people:
        delattr(self, 'hidden_activation')
        self.params.remove('hidden_activation')
        delattr(self, 'hidden_dim')
        self.params.remove('hidden_dim')

    def _prepare_label(self, y):
        # Data prep:
        self.classes_ = sorted(set(y))
        self.n_classes_ = len(self.classes_)
        class2index = dict(zip(self.classes_, range(self.n_classes_)))
        y = [class2index[label] for label in y]
        return torch.tensor(y, dtype=torch.long)

    def _prepare_dataset(self, X):
        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in X:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=self.max_set_length,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return input_ids, attention_masks

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
        labels = self._prepare_label(y)
        input_ids, attention_masks = self._prepare_dataset(X)

        dataset = TensorDataset(input_ids, attention_masks, labels)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True)

        # Graph:
        self.model.to(self.device)
        self.model.train()

        optimizer = self.optimizer(
            self.model.parameters(),
            lr=self.eta,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
            eps=self.eps  # args.adam_epsilon  - default is 1e-8.
        )

        total_steps = len(dataloader) * self.max_iter
        print("batch number: {}, total_steps: {}".format(len(dataloader), total_steps))

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

        # Train:
        t0 = time.time()
        for iteration in range(1, self.max_iter + 1):
            epoch_error = 0.0
            for step, batch in enumerate(dataloader):
                # `batch` contains three pytorch tensors:
                #   [0]: input ids    [1]: attention masks   [2]: labels
                X_input_ids = batch[0].to(self.device)
                X_input_mask = batch[1].to(self.device)
                X_labels = batch[2].to(self.device)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                self.model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # arge given and what flags are set. For our useage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                loss, logits = self.model(X_input_ids,
                                          token_type_ids=None,
                                          attention_mask=X_input_mask,
                                          labels=X_labels)

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                epoch_error += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

                # Incremental predictions where possible:
                if step % 50 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)

                    print('Interation: {}, Finished batch {}, loss is {}. Elapsed: {}'.format(iteration, step, loss.item(), elapsed))

            self.errors.append(epoch_error)
            progress_bar("Finished epoch {} of {}; error is {}".format(
                iteration, self.max_iter, epoch_error))
        return self

    def predict_proba(self, X):
        """Predicted probabilities for the examples in `X`.
        Returns
        -------
        np.array with shape (len(X), self.n_classes_)

        """
        self.model.eval()
        with torch.no_grad():
            self.model.to(self.device)
            input_ids, attention_masks = self._prepare_dataset(X)
            input_ids = input_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)

            output = self.model(input_ids,
                                token_type_ids=None,
                                attention_mask=attention_masks)

            preds = torch.softmax(output[0], dim=1).cpu().numpy()
            return preds

    def predict(self, X):
        """Predicted labels for the examples in `X`. These are converted
        from the integers that PyTorch needs back to their original
        values in `self.classes_`.

        Returns
        -------
        list of length len(X)

        """
        probs = self.predict_proba(X)
        return [self.classes_[i] for i in probs.argmax(axis=1)]

def BERT_model(max_set_length=128,max_iter=2,batch_size=32,eta=2e-5,eps=1e-8):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    train_data = load_train_data()
    train_data = process_train_data(train_data)
    X_train, y_train = list(train_data['tweet']), list(train_data['subtask_a'])

    BertClassifier = TorchBertClassifier(
        tokenizer=tokenizer,
        model=model,
        optimizer=AdamW,
        max_set_length=max_set_length,
        max_iter=max_iter,
        batch_size=batch_size,
        eta=eta,
        eps=eps)
    print(BertClassifier)

    BertClassifier.fit(X_train, y_train)

    test_data = load_test_data_a()
    test_data = process_test_data(test_data)
    X_test, y_test = list(test_data['tweet']), list(test_data['subtask_a'])

    predictions = BertClassifier.predict(X_test)
    test_data['prediction'] = np.array(predictions)
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
    output_file_path = os.path.join(RESULT_FOLDER, "BERT_Iter_{}_prediction.csv".format(max_iter))
    test_data.to_csv(output_file_path, index=False)

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    F1_score = f1_score(change_to_binary(y_test), change_to_binary(predictions), average='macro')
    print("f1 score: {}".format(F1_score))
    return F1_score

if __name__ == '__main__':
    seed_val = 0

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    BERT_model(max_iter=3)
