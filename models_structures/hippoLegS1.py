import torch 
import torch.nn as nn 
import math


def matrix(hippo_type , dim) :
    if hippo_type =='legs' : 
        A = torch.zeros(dim , dim)
        b  = torch.zeros(dim , 1)
        for i in range(dim) : 
            A[i , i] = i+1 
            b[i] = (2*(i+1))**(1/2)
        for i in range(dim) : 
            for j in range(0 , i , 1) : 
                A[i , j] = ((2*i+1)*(2*j+1))**0.5
        return -A , b
    elif hippo_type == 'random' : 
        A = torch.randn(dim , dim)
        b = torch.randn(dim , 1)
        return A , b 
        
def discretization(A: torch.Tensor, B: torch.Tensor, n_samples: int):
    dim = A.shape[0]
    I = torch.eye(dim, dtype=A.dtype, device=A.device)

    A_stack = []
    B_stack = []

    for k in range(1, n_samples + 1):
        # effective timestep due to 1/t scaling
        dt = math.log(k + 1) - math.log(k)
        
        # discrete system matrix
        Ad = torch.matrix_exp(A * dt)
        
        # discrete input matrix: solve A * X = (Ad - I) * B
        try:
            Bd = torch.linalg.solve(A, Ad @ B - B)
        except RuntimeError:
            # in case A is singular, add small regularization
            Bd = torch.linalg.solve(A + 1e-12 * I, Ad @ B - B)

        # subtract identity to match the form c_{k+1} = c_k + A_d c_k + B_d f_k
        A_stack.append(Ad - I)
        B_stack.append(Bd)

    A_stack = torch.stack(A_stack, dim=0)  # (n_samples, dim, dim)
    B_stack = torch.stack(B_stack, dim=0)  # (n_samples, dim, 1)

    return A_stack, B_stack



class RNN_block(nn.Module):
    def __init__(self, x_dim, h_dim, c_dim, y_dim):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.c_dim = c_dim
        self.y_dim = y_dim

        # Encoders to construct scalar u
        self.input_encoders  = nn.Parameter(torch.randn(x_dim, 1))
        self.hidden_encoders = nn.Parameter(torch.randn(h_dim, 1))
        self.memory_encoders = nn.Parameter(torch.randn(c_dim, 1))

        # Hidden update
        self.W_hh = nn.Parameter(torch.randn(h_dim, h_dim))
        self.W_xh = nn.Parameter(torch.randn(x_dim + c_dim, h_dim))
        self.b_h = nn.Parameter(torch.zeros(1, h_dim))

        # Output
        self.W_hy = nn.Parameter(torch.randn(h_dim, y_dim))
        self.b_y = nn.Parameter(torch.zeros(1, y_dim))
        self.tanh = nn.Tanh()
        self.f = nn.Linear(h_dim, 1)  # optional, can be identity

    def forward(self, x, h_prev, c_prev, A_d, B_d):

        # 1️⃣ Compute scalar u for memory update
        u = x @ self.input_encoders + h_prev @ self.hidden_encoders + c_prev @ self.memory_encoders  # (batch,1)

        # 2️⃣ Memory update
        c_next =  c_prev @ A_d.T + u @ B_d.T  # (batch, c_dim)

        # 3️⃣ Hidden update
        xc = torch.cat([x, c_next], dim=1)
        h_next = self.tanh(xc @ self.W_xh + h_prev @ self.W_hh + self.b_h)  # (batch, h_dim)

        # 4️⃣ Output
        y = h_next @ self.W_hy + self.b_y  # (batch, y_dim)

        return h_next, c_next, y



class RNN(nn.Module):
    def __init__(self, x_dim, h_dim, c_dim, y_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.h_dim = h_dim
        self.c_dim = c_dim

        # RNN block
        self.rnnCell = RNN_block(x_dim, h_dim, c_dim, y_dim)

        # HiPPO matrices
        A, B = matrix('legs', c_dim)  # use your LegS matrix function
        A_stack, B_stack = discretization(A, B, seq_len)  # ZOH discretization

        # store as buffers to avoid updating them during training
        self.register_buffer("A_stack", A_stack)  # (seq_len, c_dim, c_dim)
        self.register_buffer("B_stack", B_stack)  # (seq_len, c_dim, 1)

    def forward(self, x):
        """
        x: (batch, seq_len, x_dim)
        Returns: (batch, seq_len, y_dim)
        """
        batch_size = x.shape[0]
        device = x.device

        # Initialize hidden and memory
        h = torch.zeros(batch_size, self.h_dim, device=device)
        c = torch.zeros(batch_size, self.c_dim, device=device)

        out = []

        for t in range(self.seq_len):
            x_t = x[:, t, :]  # (batch, x_dim)
            A_d = self.A_stack[t]  # (c_dim, c_dim)
            B_d = self.B_stack[t]  # (c_dim, 1)

            # Forward through RNN block
            h, c, y = self.rnnCell(x_t, h, c, A_d, B_d)
            out.append(y)

        # Stack outputs along sequence dimension
        out = torch.stack(out, dim=1)  # (batch, seq_len, y_dim)
        return out




        
class model(nn.Module):
    def __init__(self, x_dim, h_dim, c_dim, seq_len, dim2, dim3, dim_out):
        super().__init__()
        self.hippo_rnn = RNN(x_dim, h_dim, c_dim, 1, seq_len)

        self.fc1 = nn.Linear(seq_len, dim2)
        self.bn1 = nn.BatchNorm1d(dim2)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(dim2, dim3)
        self.bn2 = nn.BatchNorm1d(dim3)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(dim3, dim_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hippo_rnn(x).squeeze(-1)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x

