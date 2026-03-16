import torch.nn.functional as F
import torch
import torch.nn as nn

class test_v1(torch.nn.Module):
    def __init__(self, modalities, hid_dim1=16, hid_dim11=16, hid_dim2=16, out_dim=2):
        super(test_v1, self).__init__()
        self.hid_dim1 = hid_dim1
        self.modalities = {str(k): v for k, v in modalities.items()}

        self.input_projections = nn.ModuleDict({
                 k: nn.Linear(v, hid_dim1) for k, v in self.modalities.items()})

        # Q, K, V projections per modality
        self.Qs = nn.ModuleDict({k: nn.Linear(hid_dim1, hid_dim1) for k in self.modalities})
        self.Ks = nn.ModuleDict({k: nn.Linear(hid_dim1, hid_dim1) for k in self.modalities})
        self.Vs = nn.ModuleDict({k: nn.Linear(hid_dim1, hid_dim1) for k in self.modalities})

        # Cross-modality fusion
        self.fusion_att = nn.Linear(hid_dim1, 1)

        # Prediction MLP
        self.pred0 = nn.Linear(hid_dim11, hid_dim2)  # integrate clinical_data
        self.pred1 = nn.Linear(hid_dim2, out_dim)

    def forward(self, data_dic,clinical_data):
        embeddings = []

        for key, x in data_dic.items():
            key_str = str(key)

            H = F.leaky_relu_(self.input_projections[key_str](x))


            Q = (self.Qs[key_str](H))
            K = (self.Ks[key_str](H))
            V = (self.Vs[key_str](H))

            # ---- Step 3: self-attention across features ----
            scores = torch.matmul(Q, K.transpose(0, 1)) / (self.hid_dim1 ** 0.5)
            attn = F.softmax(scores, dim=1)
            emb = torch.matmul(attn, V)
            embeddings.append(emb)

        # ---- Step 5: cross-modality fusion ----
        E = torch.stack(embeddings, dim=1)
        scores = self.fusion_att(E)
        weights = torch.softmax(scores, dim=1)
        integrated = torch.sum(weights*E, dim=1)

        # ---- Step 6: integrate clinical data ----
        integrated = torch.cat((integrated, clinical_data), dim=1)  # B x (H + clinical_dim)
        out0 = F.tanh(self.pred0(integrated))
        out1 = F.sigmoid(self.pred1(out0))

        return weights, out1.squeeze(1)

# class test_v1(torch.nn.Module):
#     def __init__(self, modalities, hid_dim1=16, hid_dim11=16, hid_dim2=16, out_dim=2):
#         super(test_v1, self).__init__()
#         self.hid_dim1 = hid_dim1
#         self.modalities = {str(k): v for k, v in modalities.items()}
#
#         self.input_projections = nn.ModuleDict({
#                  k: nn.Linear(v, hid_dim1) for k, v in self.modalities.items()})
#
#         # Q, K, V projections per modality
#         self.Qs = nn.ModuleDict({k: nn.Linear(hid_dim1, hid_dim1) for k in self.modalities})
#         self.Ks = nn.ModuleDict({k: nn.Linear(hid_dim1, hid_dim1) for k in self.modalities})
#         self.Vs = nn.ModuleDict({k: nn.Linear(hid_dim1, hid_dim1) for k in self.modalities})
#
#         # Cross-modality fusion
#         self.fusion_att = nn.Linear(hid_dim1, 1)
#
#         # Prediction MLP
#         self.pred0 = nn.Linear(hid_dim11, hid_dim2)  # integrate clinical_data
#         self.pred1 = nn.Linear(hid_dim2, out_dim)
#
#     def forward(self, data_dic,clinical_data):
#         embeddings = []
#
#         for key, x in data_dic.items():
#             key_str = str(key)
#
#             H = F.leaky_relu_(self.input_projections[key_str](x))
#
#
#             Q = (self.Qs[key_str](H))
#             K = (self.Ks[key_str](H))
#             V = (self.Vs[key_str](H))
#
#             # ---- Step 3: self-attention across features ----
#             scores = torch.matmul(Q, K.transpose(0, 1)) / (self.hid_dim1 ** 0.5)
#             attn = F.softmax(scores, dim=1)
#             emb = torch.matmul(attn, V)
#             embeddings.append(emb)
#
#         # ---- Step 5: cross-modality fusion ----
#         E = torch.stack(embeddings, dim=1)
#         scores = self.fusion_att(E)
#         weights = torch.softmax(scores, dim=1)
#         integrated = torch.sum(weights * E, dim=1)/20
#
#         # ---- Step 6: integrate clinical data ----
#         integrated = torch.cat((integrated, clinical_data), dim=1)  # B x (H + clinical_dim)
#         out0 = F.relu(self.pred0(integrated))
#         out1 = F.sigmoid(self.pred1(out0))
#
#         return weights, out1.squeeze(1)


