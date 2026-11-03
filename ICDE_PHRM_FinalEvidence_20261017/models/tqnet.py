import torch
import torch.nn as nn


class Model(nn.Module):
    """The official TQNet implementation used by the base checkpoints."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin

        self.use_tq = True
        self.channel_aggre = True
        self.temporalQuery = nn.Parameter(torch.zeros(self.cycle_len, self.enc_in))
        self.channelAggregator = nn.MultiheadAttention(
            embed_dim=self.seq_len, num_heads=4, batch_first=True, dropout=0.5
        )
        self.input_proj = nn.Linear(self.seq_len, self.d_model)
        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout), nn.Linear(self.d_model, self.pred_len)
        )

    def forward(self, x, cycle_index):
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x_input = x.permute(0, 2, 1)
        gather_index = (
            cycle_index.view(-1, 1)
            + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)
        ) % self.cycle_len
        query_input = self.temporalQuery[gather_index].permute(0, 2, 1)
        channel_information = self.channelAggregator(
            query=query_input, key=x_input, value=x_input
        )[0]
        inp = self.input_proj(x_input + channel_information)
        hidden = self.model(inp)
        output = self.output_proj(hidden + inp).permute(0, 2, 1)

        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean
        return output
