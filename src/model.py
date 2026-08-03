import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import PAD_IDX, SOS_IDX, EOS_IDX, get_device, set_seed
from dataset import get_dataloaders, MAX_SOURCE_LEN, MAX_TARGET_LEN


class Encoder(nn.Module):
    def __init__(self, source_vocab_size, embedding_dim, encoder_hidden_dim, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(
            source_vocab_size,
            embedding_dim,
            padding_idx=PAD_IDX
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=encoder_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        # Because encoder is bidirectional, forward + backward hidden states are concatenated.
        self.hidden_bridge = nn.Linear(encoder_hidden_dim * 2, encoder_hidden_dim * 2)
        self.cell_bridge = nn.Linear(encoder_hidden_dim * 2, encoder_hidden_dim * 2)

    def forward(self, source_ids):
        embedded = self.dropout(self.embedding(source_ids))

        # Pack by true length so the final hidden state is read off the last real
        # token, not the trailing <pad> positions.
        lengths = (source_ids != PAD_IDX).sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_outputs, (hidden, cell) = self.lstm(packed)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=source_ids.shape[1]
        )

        hidden_forward = hidden[-2]
        hidden_backward = hidden[-1]
        hidden_combined = torch.cat((hidden_forward, hidden_backward), dim=1)

        cell_forward = cell[-2]
        cell_backward = cell[-1]
        cell_combined = torch.cat((cell_forward, cell_backward), dim=1)

        decoder_hidden = torch.tanh(self.hidden_bridge(hidden_combined)).unsqueeze(0)
        decoder_cell = torch.tanh(self.cell_bridge(cell_combined)).unsqueeze(0)

        return encoder_outputs, decoder_hidden, decoder_cell


class Attention(nn.Module):
    def __init__(self, decoder_hidden_dim):
        super().__init__()

        self.attn = nn.Linear(decoder_hidden_dim * 2, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, source_mask):
        source_len = encoder_outputs.shape[1]

        hidden = decoder_hidden[-1].unsqueeze(1).repeat(1, source_len, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention_scores = self.v(energy).squeeze(2)

        # Mask padding tokens so the model does not attend to <pad>.
        attention_scores = attention_scores.masked_fill(source_mask == 0, -1e10)

        attention_weights = F.softmax(attention_scores, dim=1)

        return attention_weights


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embedding_dim, decoder_hidden_dim, dropout=0.3, use_attention=True):
        super().__init__()

        self.use_attention = use_attention

        self.embedding = nn.Embedding(
            target_vocab_size,
            embedding_dim,
            padding_idx=PAD_IDX
        )

        self.attention = Attention(decoder_hidden_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim + decoder_hidden_dim,
            hidden_size=decoder_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.output_projection = nn.Linear(
            embedding_dim + decoder_hidden_dim + decoder_hidden_dim,
            target_vocab_size
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs, source_mask):
        # input_token shape: [batch_size]
        input_token = input_token.unsqueeze(1)

        embedded = self.dropout(self.embedding(input_token))

        if self.use_attention:
            attention_weights = self.attention(hidden, encoder_outputs, source_mask)
            attention_weights = attention_weights.unsqueeze(1)
            context = torch.bmm(attention_weights, encoder_outputs)
            attn_out = attention_weights.squeeze(1)
        else:
            # Ablation: no attention. Use a masked mean of the encoder outputs.
            mask = source_mask.unsqueeze(2).float()
            summed = (encoder_outputs * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            context = (summed / counts).unsqueeze(1)
            attn_out = None

        lstm_input = torch.cat((embedded, context), dim=2)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        prediction_input = torch.cat(
            (
                output.squeeze(1),
                context.squeeze(1),
                embedded.squeeze(1)
            ),
            dim=1
        )

        prediction = self.output_projection(prediction_input)

        return prediction, hidden, cell, attn_out


class Seq2SeqAttentionModel(nn.Module):
    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        embedding_dim=128,
        encoder_hidden_dim=128,
        dropout=0.3,
        use_attention=True
    ):
        super().__init__()

        decoder_hidden_dim = encoder_hidden_dim * 2

        self.encoder = Encoder(
            source_vocab_size=source_vocab_size,
            embedding_dim=embedding_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            dropout=dropout
        )

        self.decoder = Decoder(
            target_vocab_size=target_vocab_size,
            embedding_dim=embedding_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            dropout=dropout,
            use_attention=use_attention
        )

        self.target_vocab_size = target_vocab_size

    def create_source_mask(self, source_ids):
        return (source_ids != PAD_IDX)

    def forward(self, source_ids, target_input_ids):
        batch_size = source_ids.shape[0]
        target_len = target_input_ids.shape[1]

        outputs = torch.zeros(
            batch_size,
            target_len,
            self.target_vocab_size,
            device=source_ids.device
        )

        source_mask = self.create_source_mask(source_ids)

        encoder_outputs, hidden, cell = self.encoder(source_ids)

        for t in range(target_len):
            input_token = target_input_ids[:, t]

            prediction, hidden, cell, _ = self.decoder(
                input_token,
                hidden,
                cell,
                encoder_outputs,
                source_mask
            )

            outputs[:, t, :] = prediction

        return outputs

    def generate(self, source_ids, max_len=MAX_TARGET_LEN):
        self.eval()

        source_mask = self.create_source_mask(source_ids)

        encoder_outputs, hidden, cell = self.encoder(source_ids)

        batch_size = source_ids.shape[0]
        input_token = torch.full(
            (batch_size,),
            SOS_IDX,
            dtype=torch.long,
            device=source_ids.device
        )

        generated_ids = []

        with torch.no_grad():
            for _ in range(max_len):
                prediction, hidden, cell, _ = self.decoder(
                    input_token,
                    hidden,
                    cell,
                    encoder_outputs,
                    source_mask
                )

                next_token = prediction.argmax(dim=1)
                generated_ids.append(next_token)

                input_token = next_token

        generated_ids = torch.stack(generated_ids, dim=1)

        return generated_ids


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    set_seed()
    device = get_device()

    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, source_vocab, target_vocab = get_dataloaders()

    model = Seq2SeqAttentionModel(
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        embedding_dim=128,
        encoder_hidden_dim=128,
        dropout=0.3
    ).to(device)

    print("Model created successfully.")
    print(f"Trainable parameters: {count_parameters(model):,}")

    batch = next(iter(train_loader))

    source_ids = batch["source_ids"].to(device)
    target_input_ids = batch["target_input_ids"].to(device)

    outputs = model(source_ids, target_input_ids)

    print("\nForward pass successful.")
    print("source_ids shape:", source_ids.shape)
    print("target_input_ids shape:", target_input_ids.shape)
    print("model output shape:", outputs.shape)

    generated_ids = model.generate(source_ids[:2])

    print("\nGreedy generation test successful.")
    print("Generated IDs shape:", generated_ids.shape)

    print("\nExample untrained generated headline:")
    print(target_vocab.decode(generated_ids[0].cpu()))


if __name__ == "__main__":
    main()
