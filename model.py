import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils import rnn as rnn_utils
from attention import MultiHeadAttention
from transformers import CamembertModel, CamembertConfig, FlaubertConfig, FlaubertModel, \
    XLMRobertaConfig, XLMRobertaModel, BertModel, BertConfig


class CoAttention(nn.Module):
    """Co-Attention Architecture"""

    def __init__(self, ques_enc_params, img_enc_params, K, mlp_dim=512):
        super().__init__()
        self.hidden_dim = ques_enc_params['hidden_dim']
        self.image_encoder = ImageCoAttentionEncoder(**img_enc_params)
        self.question_encoder = QuestionCoAttentionEncoder.create(**ques_enc_params)
        self.co_attention = CoAttention(self.hidden_dim)
        self.mlp_classify = FeedForward(self.hidden_dim, mlp_dim, K)

    def forward(self, x_img, x_input_id, x_input_mask, x_input_type_ids):
        # Word features
        x_word = self.question_encoder(x_input_id, x_input_mask, x_input_type_ids)   # [batch, max_seq_len, hidden_dim]
        # Question Features ([word])
        x_ques_features = [x_word]                            # [batch, max_seq_len, hidden_dim]
        # Image Features
        x_img_features = self.image_encoder(x_img)                                  # [batch, spatial_locs, hidden_dim]
        # Attention weighted image & question features
        x_img_attn, x_ques_attn = self.co_attention(x_img_features, x_ques_features)    # [B, hid_dim], [B, hid_dim]
        # Predict Answer (logits)
        x_logits = self.mlp_classify(x_img_attn, x_ques_attn)                       # [batch_size, K]

        return x_logits


class ImageCoAttentionEncoder(nn.Module):


    def __init__(self, is_require_grad):
        super(ImageCoAttentionEncoder, self).__init__()

        self.is_require_grad = is_require_grad

        # Resnet Encoder
        self.resnet_encoder = self.build_resnet_encoder()

        # Flatten the feature map grid [B, D, H, W] --> [B, D, H*W]
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)


    def forward(self, x_img):
        x_feat_map = self.resnet_encoder(x_img)

        # Flatten (16 x 16 x 2048) --> (16*16, 2048)
        x_feat = self.flatten(x_feat_map)

        x_feat = x_feat.permute(0, 2, 1)                            # [batch_size, spatial_locs, 2048]

        return x_feat

    def build_encoder(self):
        """
        Given Resnet backbone, build the encoder network from all layers except the last 2 layers.

        :return: model (nn.Module)
        """
        resnet = models.resnet152(pretrained=True)

        modules = list(resnet.children())[:-2]

        resnet_encoder = nn.Sequential(*modules)

        for param in resnet_encoder.parameters():
            param.requires_grad = self.is_require_grad

        return resnet_encoder



class QuestionCoAttentionEncoder(nn.Module):
    """
    Encode question phrases using  BERT based embeddings
    and apply an BILSTM to encode the question.
    """
    def __init__(self, config, model, lstm,attn):
        super(QuestionCoAttentionEncoder, self).__init__()

        self.configuration = config
        self.model = model
        self.lstm  = lstm
        self.attn = attn
        if self.configuration['mode'] == "weighted":
            self.bert_weights = torch.nn.Parameter(torch.FloatTensor(12, 1))
            self.bert_gamma = torch.nn.Parameter(torch.FloatTensor(1, 1))
        self.init_weights()

    def init_weights(self):
        if self.configuration["mode"] == "weighted":
            torch.nn.init.xavier_normal(self.bert_gamma)
            torch.nn.init.xavier_normal(self.bert_weights)


    @classmethod
    def create(cls,
               model_type ='camem',
               model_name ="camembert-base",
               embedding_size = 768,
               hidden_dim = 512,
               rnn_layers = 1,
               lstm_dropout = 0.5,
               device="cuda",
               mode="weighted",
               key_dim=64,
               val_dim=64,
               num_heads=3,
               attn_dropout=0.3,
               self_attention=False,
               is_require_grad=False):
        configuration = {
            'model_type' : model_type,
            "model_name": model_name,
            "device": device,
            "mode": mode,
            "self_attention":self_attention,
            "is_freeze": is_require_grad
        }

        if 'camem' in model_type:
            config_bert = CamembertConfig.from_pretrained(model_name, output_hidden_states=True)
            model = CamembertModel.from_pretrained(model_name, config=config_bert)
            model.to(device)
        elif 'flaubert' in model_type:
            config_bert = FlaubertConfig.from_pretrained(model_name, output_hidden_states=True)
            model = FlaubertModel.from_pretrained(model_name, config=config_bert)
            model.to(device)
        elif 'XLMRoberta' in model_type:
            config_bert = XLMRobertaConfig.from_pretrained(model_name, output_hidden_states=True)
            model = XLMRobertaModel.from_pretrained(model_name, config=config_bert)
            model.to(device)
        elif 'M-Bert' in model_type:
            config_bert = BertConfig.from_pretrained(model_name, output_hidden_states=True)
            model = BertModel.from_pretrained(model_name, config=config_bert)
            model.to(device)

        lstm = BiLSTM.create(embedding_size=embedding_size, hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)

        attn = MultiHeadAttention(key_dim, val_dim, hidden_dim, num_heads, attn_dropout)
        model.train()
        self = cls(model=model, config=configuration, lstm=lstm,attn=attn)
        # if is_freeze:
        self.freeze()

        return self

    def freeze(self):
        
        for param in self.model.parameters():
            param.requires_grad = self.configuration['is_freeze']

    def forward(self, x_input_id, x_input_mask, x_input_type_ids):

        if 'camem' in self.configuration['model_type']:
            encoded_layers, _, all_layer_embeddings = self.model(input_ids = x_input_id,
                                                                 token_type_ids = x_input_type_ids,
                                                                 attention_mask= x_input_mask)
        elif 'flaubert' in self.configuration['model_type']:
            encoded_layers, all_layer_embeddings = self.model(input_ids = x_input_id,
                                                                 token_type_ids = x_input_type_ids,
                                                                 attention_mask= x_input_mask)

        elif 'XLMRoberta' in self.configuration['model_type']:
            encoded_layers,_, all_layer_embeddings = self.model(input_ids = x_input_id,
                                                                 token_type_ids = x_input_type_ids,
                                                                 attention_mask= x_input_mask)

        elif 'M-Bert' in self.configuration['model_type']:
            encoded_layers,_, all_layer_embeddings = self.model(input_ids = x_input_id,
                                                                 token_type_ids = x_input_type_ids,
                                                                 attention_mask= x_input_mask)



        if self.configuration["mode"] == "weighted":
            encoded_embeddings = torch.stack([a * b for a, b in zip(all_layer_embeddings, self.bert_weights)])
            input_embeddings = self.bert_gamma * torch.sum(encoded_embeddings, dim=0)
        else:
            input_embeddings = encoded_layers[-1, :, :, :]
        output, _ = self.lstm.forward(input_embeddings, x_input_mask)
        if self.configuration['self_attention']:
            output, _ = self.attn(output, output, output, None)
        return output


class BiLSTM(nn.Module):

    def __init__(self, embedding_size=768, hidden_dim=512, rnn_layers=1, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_dim // 2,
            rnn_layers, batch_first=True, bidirectional=True)

    def forward(self, input_, input_mask):
        length = input_mask.sum(-1)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_ = input_[sorted_idx]
        packed_input = rnn_utils.pack_padded_sequence(input_, sorted_lengths.data.tolist(), batch_first=True)
        self.lstm.flatten_parameters()
        output, (hidden, _) = self.lstm(packed_input)
        padded_outputs = rnn_utils.pad_packed_sequence(output, batch_first=True)[0]
        _, reversed_idx = torch.sort(sorted_idx)
        return padded_outputs[reversed_idx], hidden[:, reversed_idx]

    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)



class CoAttention(nn.Module):
    """
    Implements Co-Attention mechanism
    given image & question features.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Affinity layer
        self.W_b = nn.Linear(self.hidden_dim, self.hidden_dim)
        # Attention layers
        self.W_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.w_v = nn.Linear(self.hidden_dim, 1)
        self.w_q = nn.Linear(self.hidden_dim, 1)

    def forward(self, x_img, x_ques_hierarchy):
        img_feats = []
        quest_feats = []

        for x_ques in x_ques_hierarchy:
            Q = x_ques                                              # [batch_size, max_seq_len, hidden_dim]
            V = x_img.permute(0, 2, 1)                              # [batch_size,  hidden_dim, spatial_locs]

            # Affinity matrix
            C = F.tanh(torch.bmm(Q, V))                             # [batch_size, max_seq_len, spatial_locs]
            ##Tranpose the image spatial matrix
            V = V.permute(0, 2, 1)                                  # [batch_size, spatial_locs, hidden_dim]

            H_v = F.tanh(self.W_v(V) +                              # [batch_size, spatial_locs, hidden_dim]
                         torch.bmm(C.transpose(2, 1), self.W_q(Q)))

            H_q = F.tanh(self.W_q(Q) +                              # [batch_size, max_seq_len, hidden_dim]
                         torch.bmm(C, self.W_v(V)))

            # Attention weights
            a_v = F.softmax(self.w_v(H_v), dim=1)                   # [batch_size, spatial_locs, 1]
            a_q = F.softmax(self.w_q(H_q), dim=1)                   # [batch_size, max_seq_len, 1]

            # Compute attention-weighted features
            v = torch.sum(a_v * V, dim=1)                           # [batch_size, hidden_dim]
            q = torch.sum(a_q * Q, dim=1)                           # [batch_size, hidden_dim]

            img_feats.append(v)
            quest_feats.append(q)

        return img_feats, quest_feats                               # 3*[batch, hidden_dim], 3*[batch, hidden_dim]


class FeedForward(nn.Module):
    """
    Feed forward neural network that concatenates the image and word attention weights
    """
    def __init__(self, hidden_dim, K):
        super().__init__()

        self.W_w = nn.Linear(hidden_dim, hidden_dim)
        # self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.W_h = nn.Linear(hidden_dim, K)

    def forward(self, x_img_feats, x_ques_feats):
        q_w  = x_ques_feats[0]                                    # [batch_size, hidden_dim]
        v_w = x_img_feats [0]                                    # [batch_size, hidden_dim]
        h_w = F.tanh(self.W_w(q_w + v_w))                               # [batch_size, hidden_dim]
        # Final answer (classification logit)
        # logit = self.W_h(self.batch_norm(h_w))                                       # [batch_size, K]

        logit = self.W_h(h_w)
        return logit

