import torch
import torch.nn as nn
import torch.nn.functional as F

def x_to_cnn(input ) : 
    data =torch.zeros(input.shape[0] ,input.shape[1]  , 9 , 9 ,device=input.device)
    grids = [(1,3) , (2,0) , (4,2) , (3,1) , (4,0) , (6,0) , (8,3) , (8,5) , (6,8) , (4,8) , (3,7) , (2,6) , (2,8) , (1,5)]
    for g, (i, j) in enumerate(grids):
        data[:, :, i, j] = input[:, :, g]
    return data


class EmotionCaps(nn.Module):
    def __init__(self, num_emotions, out_dim, num_iterations=3):
        super(EmotionCaps, self).__init__()
        self.num_emotions = num_emotions
        self.out_dim = out_dim
        self.num_iterations = num_iterations
        self.W = None

    def squash(self, s, dim=-1):
        norm = torch.norm(s, dim=dim, keepdim=True)
        scale = (norm ** 2) / (1 + norm ** 2)
        return scale * s / (norm + 1e-8)

    def forward(self, u):
        B, N, in_dim = u.size()
        if self.W is None:
            self.num_capsules = N
            self.in_dim = in_dim
            self.W = nn.Parameter(torch.randn(1, N, self.num_emotions, self.out_dim, in_dim, device=u.device))
        u_exp = u.unsqueeze(2).unsqueeze(-1)   # (B, N, 1, in_dim, 1)
        u_hat = torch.matmul(self.W, u_exp).squeeze(-1)  # (B, N, num_emotions, out_dim)
        b = torch.zeros(B, N, self.num_emotions, device=u.device)
        for r in range(self.num_iterations):
            c = F.softmax(b, dim=2).unsqueeze(-1)
            s = (c * u_hat).sum(dim=1)
            v = self.squash(s)
            if r < self.num_iterations - 1:
                v_exp = v.unsqueeze(1).unsqueeze(-1)
                uv = torch.matmul(u_hat.unsqueeze(-2), v_exp).squeeze(-1).squeeze(-1)
                b = b + uv
        return v


class model(nn.Module):
    def __init__(self, num_filter, time_len, caps_len, num_emotions, out_dim):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filter, kernel_size=(6,1), stride=(2,1), padding=0)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size=(5,1), stride=1, padding=(2,0))
        self.conv3 = nn.Conv2d(2*num_filter, num_filter, kernel_size=1)
        self.relu = nn.ReLU()
        self.caps_len = caps_len
        self.emotion_caps = EmotionCaps(num_emotions=num_emotions, out_dim=out_dim)

    def forward(self, x):
        # x: (B, T, C)
        x = x.unsqueeze(1)  # (B, 1, T, C)
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x = torch.cat((x1, x2), dim=1)
        x = self.relu(self.conv3(x))
        B, C, H, W = x.size()
        x = x.view(B, -1, self.caps_len)  # Flatten برای Caps
        v = self.emotion_caps(x)
        v_abs = torch.norm(v, dim=-1)
        return v_abs
