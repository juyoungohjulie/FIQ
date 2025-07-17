import numpy as np
from torch.nn import functional as F

from .utils import *
# from timm.models import create_model
# from Vim.vim.models_mamba import VisionMamba, vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2
from einops.layers.torch import Rearrange
from Vim.vim.wrapper import BiDrectionalMamba
import copy
from typing import Optional, List
from torch import nn, Tensor

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

###########################Transformer Decoder, for question embedding ###########################
class TransformerDecoderLayer_VQ(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        self_attn_output = tgt2
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
        #                            key=self.with_pos_embed(memory, pos),
        #                            value=memory, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)[0]
        # import ipdb; ipdb.set_trace()
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos), # pos is optional
                                   value=memory)[0]
        multihead_attn_output = tgt2
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        linear2_output = tgt2
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        # import ipdb; ipdb.set_trace()  
        return tgt

    def forward_pre(self, tgt, memory,
                    # tgt_mask: Optional[Tensor] = None,
                    # memory_mask: Optional[Tensor] = None,
                    # tgt_key_padding_mask: Optional[Tensor] = None,
                    # memory_key_padding_mask: Optional[Tensor] = None, # Exclude mask-related parameters
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
        #                            key=self.with_pos_embed(memory, pos),
        #                            value=memory, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # if self.normalize_before:
        #     return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
        #                             tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        # return self.forward_post(tgt, memory, tgt_mask, memory_mask,
        #                          tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        # import ipdb; ipdb.set_trace()
        if self.normalize_before:
            # print("normalize_before True")
            return self.forward_pre(tgt, memory, pos, query_pos)
        # print("normalize_before False")
        return self.forward_post(tgt, memory, pos, query_pos)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


###############################################################
# Add new LearnablePositionalEmbedding class
class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(LearnablePositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        
        # Initialize learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        
        # Initialize with small random values
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        # x shape: [seq_len, batch, features] - transformer format
        
        # Convert to batch-first for easier positional embedding addition
        x = x.permute(1, 0, 2)  # [batch, seq, features]
        
        # Get sequence length
        seq_len = x.shape[1]
        
        # Add positional embeddings
        if seq_len <= self.max_len:
            # Use only embeddings for required sequence length
            x = x + self.pos_embedding[:, :seq_len, :]
        else:
            # Either truncate sequence or interpolate embeddings
            print(f"Warning: Sequence length {seq_len} exceeds maximum positional embeddings {self.max_len}")
            x = x[:, :self.max_len, :] + self.pos_embedding
        
        # Apply dropout
        x = self.dropout(x)
        
        # Convert back to sequence-first format
        x = x.permute(1, 0, 2)  # [seq, batch, features]
        
        return x
###############################################################


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :] 
        return self.dropout(x)


class TempAligner(nn.Module):
    def __init__(self, vision_dim, module_dim, use_mamba=False):
        super(TempAligner, self).__init__()

        self.positional_encoding = PositionalEncoding(module_dim, dropout=0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=module_dim, nhead=16)
        self.use_mamba = use_mamba
        if use_mamba:
            self.vision_mamba = BiDrectionalMamba(embed_dim=module_dim, depth=8)
        else:
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        decoder_layer = nn.TransformerDecoderLayer(d_model=module_dim, nhead=16) 
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.linear = nn.Linear(2304, module_dim)
        self.norm = nn.LayerNorm(module_dim)
        self.tgt_mask = None
        self.src_key_padding_mask = None
        self.question_decoder = TransformerDecoderLayer_VQ(d_model=module_dim, nhead=16, normalize_before=False) # nhead: 8 in original code, but 16 is correct
        ###########################################################################################
        # Add learnable positional embeddings (max sequence length 128)
        self.learnable_pos_embedding = LearnablePositionalEmbedding(module_dim, dropout=0.2, max_len=128)

        # self.use_mamba = True
        ###########################################################################################
        # for the learnable embedding
        # self.cls_token = nn.Parameter(torch.randn(1, 1, module_dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, 32, module_dim))
        init_modules(self.modules(), w_init="xavier_uniform")

    def forward(self, answers, ans_candidates, video_appearance_feat, question,
                ):
        """
        Args:
            ans_candidates: [Tensor] (batch_size, 5, max_ans_candidates_length)
            ans_candidates_len: [Tensor] (batch_size, 5)
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            logits.
        """
        batch_size = question.size(0)

        correct_answers = []
        for ind in range(batch_size):
            batch_answer = ans_candidates[ind]
            correct_answer = batch_answer[answers[ind]]
            correct_answers.append(correct_answer)
        correct_answers = torch.stack(correct_answers)
        correct_answers = correct_answers.unsqueeze(1)
        correct_answers = correct_answers.permute(1,0,2)
        question_embedding = question
        # import ipdb; ipdb.set_trace() # video_appearance_feat.shape : [128, 128, 512]

        # # The frame has not passed as 128 frames
        # feat_dim = video_appearance_feat.shape[-1]
        # video_appearance_feat = video_appearance_feat.contiguous().view(batch_size, -1, feat_dim) # [128, 128, 512]로 여기서도 clip x frame으로 풀어서 계산함 -> torch.Size([128, 2304, 128]) # batch, frame, featur인데 순서가 바뀜
        # video_appearance_feat = self.norm(self.linear(video_appearance_feat))
        # video_appearance_feat_ori = video_appearance_feat
        # _, nframes, _ = video_appearance_feat.shape
        # video_appearance_feat = video_appearance_feat.permute(1,0,2)
        # video_appearance_feat = self.positional_encoding(video_appearance_feat)
        # # The frame has not passed as 128 frames

        # ###################When the frame has passed as 128 frames###########################
        feat_dim = video_appearance_feat.shape[-1]
        video_appearance_feat = video_appearance_feat.view(batch_size, -1, feat_dim) # video_appearance_feat.shape : [128, 128, 512]
        # video_appearance_feat = video_appearance_feat.permute(0,2,1) #1,0,2 ->0,2,1
        # video_appearance_feat = self.norm(self.linear(video_appearance_feat))
        video_appearance_feat_ori = video_appearance_feat
        _, nframes, _ = video_appearance_feat.shape
        # import ipdb; ipdb.set_trace()
        video_appearance_feat_before_learnable = video_appearance_feat.permute(1,0,2)
        # original positional encoding
        # video_appearance_feat = self.positional_encoding(video_appearance_feat) 
        
        # learnable positional embedding
        video_appearance_feat = self.learnable_pos_embedding(video_appearance_feat_before_learnable)
        
        
        # # Add positional embeddings
        # # video_appearance_feat = video_appearance_feat + self.pos_embedding[:, :nframes, :]
        # if nframes <= self.max_seq_length:
        # # Use only the positional embeddings we need
        #     video_appearance_feat = video_appearance_feat + self.pos_embedding[:, :nframes, :]
        # else:
        #     # Handle case where sequence is longer than our embeddings
        #     print(f"Warning: Sequence length {nframes} exceeds maximum positional embeddings {self.max_seq_length}")
        #     # Either truncate sequence or interpolate positional embeddings
        #     video_appearance_feat = video_appearance_feat[:, :self.max_seq_length, :] + self.pos_embedding
        
        # # Convert back to [seq_len, batch, dim] for transformer
        # # video_appearance_feat = video_appearance_feat.permute(1, 0, 2)  # [seq_len, batch, dim]





        # import ipdb; ipdb.set_trace()
        # ###################When the frame has passed as 128 frames###########################

        if self.tgt_mask is not None and self.tgt_mask.size(0) == video_appearance_feat.size(0):
            tgt_mask = self.tgt_mask
        else:
            self.tgt_mask = nn.Transformer().generate_square_subsequent_mask(video_appearance_feat.size(0))
            self.tgt_mask = self.tgt_mask.to(video_appearance_feat.device)
            tgt_mask = self.tgt_mask
        if self.src_key_padding_mask is not None and self.src_key_padding_mask.size(0) == batch_size:
            src_key_padding_mask = self.src_key_padding_mask
        else:
            src_key_padding_mask = torch.ones((batch_size, nframes), dtype=bool, device=video_appearance_feat.device)
            self.src_key_padding_mask = src_key_padding_mask

        # print(src_key_padding_mask.shape)
        # import ipdb; ipdb.set_trace()

        # exit()
        tgt_key_padding_mask = src_key_padding_mask
        # video_appearance_feat.shape: torch.Size([128, 128, 512])
        # src_key_padding_mask.shape: torch.Size([128, 128])
        # import ipdb; ipdb.set_trace()
        # Choose between transformer encoder and vision mamba
        if self.use_mamba:
            # print("Use Mamba")
            # For Mamba: Need to convert to batch-first format
            # video_appearance_feat_mamba = video_appearance_feat.permute(1, 0, 2)  # [batch, seq, feature]
            
            # Pass through VisionMamba
            visual_embedding = self.vision_mamba(video_appearance_feat)
            
            # VisionMamba with return_features=True returns [batch, seq+cls, feature]
            # The seq_reducer in VisionMamba should handle the cls token dimension
            
            # # Convert back to transformer format
            # visual_embedding = visual_embedding.permute(1, 0, 2)  # [seq, batch, feature]
        else:
            ############################### Add question embedding to visual embedding ###############################
            # question_embedding = question_embedding.permute(2,0,1)
            # import ipdb; ipdb.set_trace()
        # ######################Test######################
            question_embedding_ori = question_embedding
            question_embedding = question_embedding.repeat(1, batch_size, 1)

            # visual_embedding = self.question_decoder(tgt=video_appearance_feat_before_learnable, memory=question_embedding, query_pos=video_appearance_feat)
            visual_embedding = self.question_decoder(tgt=video_appearance_feat, memory=question_embedding, query_pos=None)
        # ######################Test######################
            visual_embedding_before_transformer = visual_embedding
            # # transformer encoder
            # visual_embedding = self.transformer_encoder(src=video_appearance_feat, src_key_padding_mask=~src_key_padding_mask) 
            # visual_embedding = self.transformer_encoder(src=visual_embedding, src_key_padding_mask=~src_key_padding_mask) 
        
        # vision mamba encoder
        # video_appearance_feat = video_appearance_feat.permute(1,0,2)
        # visual_embedding = self.vision_mamba.forward(x=video_appearance_feat.permute(1,0,2), return_features=True) # it will return features if return_features is True
        # video_appearance_feat = video_appearance_feat.permute(1,0,2)

        ###################### Originally added visual embedding and correct answers ######################
        visual_embedding_answer = visual_embedding + correct_answers 

        # # ###################### Now added visual embedding and question embedding ######################
        # visual_embedding_answer = visual_embedding + question_embedding
        # import ipdb; ipdb.set_trace()
        visual_embedding_decoder = self.transformer_decoder(tgt=video_appearance_feat, memory=visual_embedding_answer,tgt_key_padding_mask=~tgt_key_padding_mask) #  tgt_mask=tgt_mask, 
        visual_embedding_decoder_before_decoder = visual_embedding_decoder
        # import ipdb; ipdb.set_trace()
        visual_embedding = visual_embedding.permute(1,0,2)
        visual_embedding_decoder = visual_embedding_decoder.permute(1,0,2)

        # print(visual_embedding.shape, visual_embedding_decoder.shape) 
        # exit()

        visual_embedding = torch.mean(visual_embedding, dim=1, keepdim=True) 
        out = torch.matmul(ans_candidates, visual_embedding.permute(0,2,1)).view(batch_size*4, -1) 
        # out = out.mean(dim=1, keepdim=True) 
        # return out, visual_embedding_decoder
        # import ipdb; ipdb.set_trace()
        return out, visual_embedding_decoder, video_appearance_feat_ori

