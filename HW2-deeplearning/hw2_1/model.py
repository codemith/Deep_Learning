import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pickle
import time
import torch.nn.functional as F
import numpy as np
import random
import os
from scipy.special import expit

pickel_file='/Users/mithileshbiradar/Desktop/Lockin_Repository/Deep-Learning/Mithilesh/testhw2/hw2_1/Model/picket_data.pickle'
with open(pickel_file, 'rb') as f:
        picketdata = pickle.load(f)
class Attention(nn.Module):
    def __init__(self, hidsize):
        super(Attention, self).__init__()

        self.hidsize = hidsize
        self.fc_hidden_1 = nn.Linear(2 * hidsize, hidsize)
        self.fc_hidden_2 = nn.Linear(hidsize, hidsize)
        self.fc_hidden_3 = nn.Linear(hidsize, hidsize)
        self.weight_projtcn= nn.Linear(hidsize, 1, bias=False)

    def forward(self, hidestate, encoder_seqoutput):
        b_size, sequence_len, feature_n = encoder_seqoutput.size()
        hidestate = hidestate.view(b_size, 1, feature_n).repeat(1, sequence_len, 1)
        matching_inputs = torch.cat((encoder_seqoutput, hidestate), 2).view(-1, 2 * self.hidsize)
        x = self.fc_hidden_1(matching_inputs)
        x = self.fc_hidden_2(x)
        x = self.fc_hidden_3(x)
        attention_weights = self.weight_projtcn(x)
        attention_weights = attention_weights.view(b_size, sequence_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_seqoutput).squeeze(1)
        return context
class Encoder_net(nn.Module):
    def __init__(self):
        super(Encoder_net, self).__init__()
        self.feature_compression = nn.Linear(4096, 256)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(256, 256, batch_first=True)
    def forward(self, input):
        batch_size, seq_len, feat_n = input.size()
        input_dataa = input.view(-1, feat_n)
        input_dataa = self.feature_compression(input_dataa)
        input_dataa = self.dropout(input_dataa)
        input_dataa = input_dataa.view(batch_size, seq_len, 256)
        output, hiddencontext = self.lstm(input_dataa)
        hidestate, context = hiddencontext[0], hiddencontext[1]
        return output, hidestate
class Decoder_net(nn.Module):
    def __init__(self, hidsize, outputsizee, vocabsize, worddim, dropout_percentage=0.4):
        super(Decoder_net, self).__init__()
        self.hidsize = 256
        self.outputsizee = len(picketdata) + 4
        self.vocabsize = len(picketdata) + 4
        self.worddim = 768
        self.word_embedding = nn.Embedding(len(picketdata) + 4, 768)
        self.dropout = nn.Dropout(0.4)
        self.lstm = nn.LSTM(hidsize + worddim, hidsize, batch_first=True)
        self.attention = Attention(hidsize)
        self.final_output_projtcn = nn.Linear(hidsize, outputsizee)
    def forward(self, last_encoder_hidden_state, encoderseqoutput, targets=None, mode='train', tr_steps=None):
        _, batch_size, _ = last_encoder_hidden_state.size()
        current_decoder_hidden = None if last_encoder_hidden_state is None else last_encoder_hidden_state
        current_decoder_cellstate = torch.zeros(current_decoder_hidden.size())
        current_decoder_cellstate = current_decoder_cellstate
        current_inputword = Variable(torch.ones(batch_size, 1)).long()
        current_inputword = current_inputword
        seq = []
        predict_seq = []
        targets = self.word_embedding(targets)
        _, seq_len, _ = targets.size()
        for i in range(seq_len - 1):
            sampling_threshold = self.helper(training_steps=tr_steps)
            if random.uniform(0.04, 0.994) > sampling_threshold:
                current_input_word = targets[:, i]
            else:
                current_input_word = self.word_embedding(current_inputword).squeeze(1)
            context = self.attention(current_decoder_hidden, encoderseqoutput)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            lstm_output, hiddencontext = self.lstm(lstm_input, (current_decoder_hidden, current_decoder_cellstate))
            current_decoder_hidden, current_decoder_cellstate = hiddencontext[0], hiddencontext[1]
            logprob = self.final_output_projtcn(lstm_output.squeeze(1))
            seq.append(logprob.unsqueeze(1))
            current_inputword = logprob.unsqueeze(1).max(2)[1]
        seq = torch.cat(seq, dim=1)
        predict_seq = seq.max(2)[1]
        return seq, predict_seq
    def infer(self, last_encoder_hidden_state, encoderseqoutput):
        _, batch_size, _ = last_encoder_hidden_state.size()
        current_decoder_hidden = None if last_encoder_hidden_state is None else last_encoder_hidden_state
        current_inputword = Variable(torch.ones(batch_size, 1)).long()  # <SOS> (batch x word index)
        current_inputword = current_inputword
        decoder_c= torch.zeros(current_decoder_hidden.size())
        decoder_c = decoder_c
        seq = []
        predict_seq = []
        assumption_seq_len = 25
        for i in range(assumption_seq_len - 1):
            current_input_word = self.word_embedding(current_inputword).squeeze(1)
            context = self.attention(current_decoder_hidden, encoderseqoutput)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            lstm_output, hiddencontext = self.lstm(lstm_input, (current_decoder_hidden, decoder_c))
            current_decoder_hidden, decoder_c = hiddencontext[0], hiddencontext[1]
            logprob = self.final_output_projtcn(lstm_output.squeeze(1))
            seq.append(logprob.unsqueeze(1))
            current_inputword = logprob.unsqueeze(1).max(2)[1]
        seq = torch.cat(seq, dim=1)
        predict_seq = seq.max(2)[1]
        return seq, predict_seq
    def helper(self, training_steps):
        return (expit(training_steps / 20 + 0.86))
class ModelMain(nn.Module):
    def __init__(self, video_encoder, caption_decoder):
        super(ModelMain, self).__init__()
        self.video_encoder = video_encoder
        self.caption_decoder = caption_decoder
    def forward(self, avifeat, mode, target_sentences=None, tr_steps=None):
        encoderoutputt, en_lhs = self.video_encoder(avifeat)
        if mode == 'train':
            seq_1, seq_2= self.caption_decoder(last_encoder_hidden_state=en_lhs,
                                       encoderseqoutput=encoderoutputt,
                                       targets=target_sentences, mode=mode, tr_steps=tr_steps)
        elif mode == 'inference':
            seq_1, seq_2 = self.caption_decoder.infer(last_encoder_hidden_state=en_lhs,
                                              encoderseqoutput=encoderoutputt)
        return seq_1, seq_2

