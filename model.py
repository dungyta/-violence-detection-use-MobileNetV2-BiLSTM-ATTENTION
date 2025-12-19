import torch
import torch.nn as nn
import torchvision.models as models

class AttentionBlock(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [Batch, Seq, Features]
        weights = self.attention(x)
        context = torch.sum(x * weights, dim=1)
        return context

class ViolenceModel(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1, dropout=0.5):
        super(ViolenceModel, self).__init__()

        # 1. Backbone: MobileNetV2
        mobilenet = models.mobilenet_v2(weights='DEFAULT')
        self.features = nn.Sequential(*list(mobilenet.features.children()))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # KHÓA CHẶT BACKBONE để giamr loss
        for p in self.features.parameters():
            p.requires_grad = False

        # 2. Adapter (Cầu nối )
        self.adapter = nn.Sequential(
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(1280, 512), # Giảm chiều xuống 512 cho nhẹ
            nn.ReLU()
        )

        # 3. Bi-LSTM (dùng lstm 2 chiều)
        self.lstm = nn.LSTM(
            input_size=512, # Đã giảm chiều từ adapter
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # 4. Attention (Giúp model biết frame nào là đánh nhau)
        self.attention = AttentionBlock(hidden_size * 2)

        # 5. Classifier Head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        b, s, c, h, w = x.size()

        # CNN Phase
        x = x.view(b * s, c, h, w)
        with torch.no_grad(): # Freeze tuyệt đối
            x = self.features(x)
            x = self.pool(x)

        # Adapter
        x = self.adapter(x)
        x = x.view(b, s, -1)

        # RNN Phase
        lstm_out, _ = self.lstm(x)

        # Attention Pooling (Tốt hơn Mean/Max)
        context = self.attention(lstm_out)

        # Head
        logits = self.head(context)
        return logits