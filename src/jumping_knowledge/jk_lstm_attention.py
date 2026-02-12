import torch
import torch.nn as nn

class JKLSTM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lstm = nn.LSTM(dim, dim, batch_first=True)
        self.att = nn.Linear(dim, 1)

    def forward(self, outputs):
        seq = torch.stack(outputs, dim=1)
        lstm_out, _ = self.lstm(seq)

        scores = self.att(lstm_out).squeeze(-1)
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)

        return (seq * alpha).sum(dim=1)
