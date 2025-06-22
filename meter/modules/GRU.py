import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class StackedGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dims, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.num_layers = len(hidden_dims)
        self.grus = nn.ModuleList()

        input_dim = embed_dim
        for i in range(self.num_layers):
            layer_gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dims[i],
                num_layers=1,
                batch_first=True,
                dropout=0.0,
                bidirectional=False
            )
            self.grus.append(layer_gru)
            input_dim = hidden_dims[i]

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_ids, input_mask=None):
        embedded = self.embedding(input_ids)
        if input_mask is not None:
            lengths = input_mask.sum(dim=1).cpu()
            embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        current_input = embedded
        all_outputs = []
        all_hiddens = []

        for i, layer_gru in enumerate(self.grus):
            layer_output, layer_hidden = layer_gru(current_input)

            # 如果是打包形式，解包回原 shape
            if isinstance(layer_output, nn.utils.rnn.PackedSequence):
                layer_output, _ = pad_packed_sequence(layer_output, batch_first=True)

            layer_output = self.dropout_layer(layer_output)

            all_outputs.append(layer_output)
            all_hiddens.append(layer_hidden)

            current_input = layer_output

        return all_outputs, all_hiddens