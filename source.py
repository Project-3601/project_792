# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

# ==============================
# 2. Data Preprocessing
# ==============================
class InsiderDataset(Dataset):
    def __init__(self, csv_path, seq_len=20):
        df = pd.read_csv(csv_path)
        
        # Encode categorical features
        self.user_encoder = LabelEncoder()
        self.pc_encoder = LabelEncoder()
        self.url_encoder = LabelEncoder()
        
        df["user"] = self.user_encoder.fit_transform(df["user"])
        df["pc"] = self.pc_encoder.fit_transform(df["pc"])
        df["url"] = self.url_encoder.fit_transform(df["url"])
        
        # Convert date to timestamp seconds
        df["timestamp"] = pd.to_datetime(df["date"]).astype(int) // 10**9
        
        # Scale numerical features
        scaler = MinMaxScaler()
        df[["user", "pc", "url"]] = scaler.fit_transform(df[["user", "pc", "url"]])
        
        # Group by ID for sequences
        grouped = []
        for _, group in df.groupby("id"):
            features = group[["user", "pc", "url"]].values
            timestamps = group["timestamp"].values
            time_diffs = np.diff(timestamps, prepend=timestamps[0])
            seq = np.hstack([features, time_diffs.reshape(-1, 1)])
            grouped.append(seq)
        
        # Pad/trim sequences
        self.sequences = []
        for seq in grouped:
            if len(seq) < seq_len:
                pad = np.zeros((seq_len - len(seq), seq.shape[1]))
                seq = np.vstack([seq, pad])
            else:
                seq = seq[:seq_len]
            self.sequences.append(seq)
        
        self.sequences = torch.tensor(self.sequences, dtype=torch.float32)
        
        # Placeholder binary labels (replace with actual insider labels)
        self.labels = torch.randint(0, 2, (len(self.sequences),), dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# ==============================
# 3. Time-Aware Transformer
# ==============================
class TimeAwareAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_w = nn.Linear(1, embed_dim)

    def forward(self, x, time_diffs):
        time_emb = self.time_w(time_diffs.unsqueeze(-1))
        return x + time_emb

class TSTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim - 1, embed_dim)
        self.time_aware = TimeAwareAttention(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=256,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        features = x[:, :, :-1]  # last column is time diff
        time_diffs = x[:, :, -1]
        
        emb = self.embedding(features)
        emb = self.time_aware(emb, time_diffs)
        out = self.transformer(emb)
        return self.fc(out.mean(dim=1))

# ==============================
# 4. Graph Neural Network
# ==============================
class ActivityGraph(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x):
        adj = torch.corrcoef(x.T)
        adj = torch.nan_to_num(adj)
        adj = torch.relu(adj)
        edge_index, edge_weight = dense_to_sparse(adj)
        h = F.relu(self.gcn1(x, edge_index, edge_weight))
        h = self.gcn2(h, edge_index, edge_weight)
        return h

# ==============================
# 5. Hybrid Loss Function
# ==============================
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, contrastive_margin=1.0, lambda_graph=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.margin = contrastive_margin
        self.lambda_graph = lambda_graph

    def focal_loss(self, preds, targets):
        BCE = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
        pt = torch.exp(-BCE)
        return (self.alpha * (1 - pt) ** self.gamma * BCE).mean()

    def contrastive_loss(self, features, labels):
        dist = torch.cdist(features, features)
        loss = 0
        for i in range(len(labels)):
            for j in range(len(labels)):
                if labels[i] == labels[j]:
                    loss += dist[i, j] ** 2
                else:
                    loss += torch.clamp(self.margin - dist[i, j], min=0) ** 2
        return loss / (len(labels) ** 2)

    def graph_regularization_loss(self, graph_features):
        diff = torch.norm(graph_features[1:] - graph_features[:-1], p=2, dim=1)
        return diff.mean()

    def forward(self, preds, targets, features, graph_features):
        return (
            self.focal_loss(preds, targets) +
            self.contrastive_loss(features, targets) +
            self.lambda_graph * self.graph_regularization_loss(graph_features)
        )

# ==============================
# 6. Full Model
# ==============================
class InsiderThreatModel(nn.Module):
    def __init__(self, input_dim, embed_dim=64, heads=4, layers=2):
        super().__init__()
        self.transformer = TSTransformer(input_dim, embed_dim, heads, layers)
        self.graph = ActivityGraph(embed_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        transformer_out = self.transformer(x)
        graph_out = self.graph(transformer_out)
        logits = self.classifier(transformer_out)
        return logits.squeeze(), transformer_out, graph_out

# ==============================
# 7. Training & Evaluation
# ==============================
def train_model(csv_path, epochs=5, batch_size=16, lr=1e-3):
    dataset = InsiderDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = InsiderThreatModel(input_dim=4)  # user, pc, url, time_diff
    loss_fn = HybridLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            preds, features, graph_features = model(x)
            loss = loss_fn(preds, y, features, graph_features)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(dataloader):.4f}")
    
    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            preds, _, _ = model(x)
            all_preds.extend(torch.sigmoid(preds).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    auc = roc_auc_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, np.round(all_preds)).ravel()
    detection_rate = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    print(f"AUC: {auc:.4f}, Detection Rate: {detection_rate:.4f}, FPR: {fpr:.4f}")

# ==============================
# 8. Run Training
# ==============================
if __name__ == "__main__":
    train_model(r"D:\MAHESH\time\12841247\r4.2\r4.2\http.csv", epochs=10)
