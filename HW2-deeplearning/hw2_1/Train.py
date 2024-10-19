import torch.optim as optim
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import os
from scipy.special import expit
import random
import sys
import json
import re
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import model
data_path='./MLDS_hw2_1_data'
model_path='./Model'
pickel_file='./Model/picket_data.pickle'
class Dataprocessor(Dataset):
    def __init__(self, caption_file, vidfeature_dir, dictnry, wordtoindex):
        self.caption_file = caption_file
        self.vidfeature_dir = vidfeature_dir
        self.video_features = filesreader(caption_file)
        self.wordtoindex = wordtoindex
        self.dictnry = dictnry
        self.vidcaption_pairs = helper1(vidfeature_dir, dictnry, wordtoindex)
    def __len__(self):
        return len(self.vidcaption_pairs)
    def __getitem__(self, idx):
        assert (idx < self.__len__())
        avi_file_name, caption_seq = self.vidcaption_pairs[idx]
        data = torch.Tensor(self.video_features[avi_file_name])
        data += torch.Tensor(data.size()).random_(0, 2000) / 10000
        return torch.Tensor(data), torch.Tensor(caption_seq)
class test_dataloader(Dataset):
    def __init__(self, test_data_path):
        self.video_features = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            vidval = np.load(os.path.join(test_data_path, file))
            self.video_features.append([key, vidval])
    def __len__(self):
        return len(self.video_features)
    def __getitem__(self, idx):
        return self.video_features[idx]
def dictonaryFunc(word_min):
    with open(
            './MLDS_hw2_1_data/training_label.json','r') as f:
        file = json.load(f)
    wordcount = {}
    for d in file:
        for s in d['caption']:
            ws = re.sub('[.!,;?]]', ' ', s).split()
            for word in ws:
                word = word.replace('.', '') if '.' in word else word
                if word in wordcount:
                    wordcount[word] += 1
                else:
                    wordcount[word] = 1
    dict1 = {}
    for word in wordcount:
        if wordcount[word] > word_min:
            dict1[word] = wordcount[word]
    special_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    indextoword = {index + len(special_tokens): word for index, word in enumerate(dict1)}
    wordtoindex = {word: index + len(special_tokens) for index, word in enumerate(dict1)}
    for token, index in special_tokens:
        indextoword[index] = token
        wordtoindex[token] = index
    return indextoword, wordtoindex, dict1
def string_split(caption_seq, dictnry, wordtoindex): 
    caption_seq = re.sub(r'[.!,;?]', ' ', caption_seq).split()
    for index in range(len(caption_seq)):
        if caption_seq[index] not in dictnry:
            caption_seq[index] = 3
        else:
            caption_seq[index] = wordtoindex[caption_seq[index]]
    caption_seq.insert(0, 1)
    caption_seq.append(2)
    return caption_seq
def helper1(caption_file, dictnry, wordtoindex):
    label_json = caption_file
    caption_annotations = []
    with open(label_json, 'r') as f:
        label = json.load(f)
    for d in label:
        for s in d['caption']:
            s = string_split(s, dictnry, wordtoindex)
            caption_annotations.append((d['id'], s))
    return caption_annotations
def helper2(wordtoindex, word):
    return wordtoindex[word]

def helper3(indextoword, index):
    return indextoword[index]

def helper4(wordtoindex, caption_seq):
    return [wordtoindex[word] for word in caption_seq]

def helper5(indextoword, index_seq):
    return [indextoword[int(index)] for index in index_seq]
def filesreader(vidfeature_dir):
    avidata = {}
    trainfeat = vidfeature_dir
    files = os.listdir(trainfeat)
    for file in files:
        vidval = np.load(os.path.join(trainfeat, file))
        avidata[file.split('.npy')[0]] = vidval
    return avidata
def train(model, epoch, train_loader, loss_func):
    model.train()
    print(epoch)
    model = model
    parameters = model.parameters()
    optimiser = optim.Adam(parameters, lr=0.001)
 
    batch_running_loss = 0.0
    for batch_idx, train_batch in enumerate(train_loader):
        avi_feats, captiontarget, seq_len = train_batch
        avi_feats, captiontarget = avi_feats, captiontarget
        avi_feats, captiontarget = Variable(avi_feats), Variable(captiontarget)

        optimiser.zero_grad()
        seq_logProb, seq_predictions = model(avi_feats, target_sentences=captiontarget, mode='train', tr_steps=epoch)

        captiontarget = captiontarget[:, 1:]
        loss = loss_cal(seq_logProb, captiontarget, seq_len, loss_func)
        loss.backward()
        optimiser.step()
        batch_running_loss += loss.item()
        if batch_idx % 10 == 9:
            print(f"Batch: {batch_idx+1}/{len(train_loader)}, Loss: {batch_running_loss/10:.3f}")
            batch_running_loss = 0.0
 
    loss = loss.item()
    print(f'Epoch:{epoch} & loss:{np.round(loss, 3)}')
def evaluate(test_loader, model):
    model.eval()
    for batch_idx, train_batch in enumerate(test_loader):
        eval_vidfeatures, eval_targets, seq_len = train_batch
        eval_vidfeatures, eval_targets = eval_vidfeatures, eval_targets
        eval_vidfeatures, eval_targets = Variable(eval_vidfeatures), Variable(eval_targets)
        seq_logProb, seq_predictions = model(eval_vidfeatures, mode='inference')
        eval_targets = eval_targets[:, 1:]
        testpredict = seq_predictions[:3]
        testtruth = eval_targets[:3]
        break
def testfun(test_loader, model, indextoword):
    model.eval()
    ss = []
    for batch_idx, train_batch in enumerate(test_loader):
        id, vid_feature = train_batch
        id, vid_feature = id, Variable(vid_feature).float()
        seq_logProb, seq_predictions = model(vid_feature, mode='inference')
        testpredict = seq_predictions
        generated_captions = [[indextoword[x.item()] if indextoword[x.item()] != '<UNK>' else 'something' for x in s] for s in testpredict]
        generated_captions = [' '.join(s).split('<EOS>')[0] for s in generated_captions]

        rr = zip(id, generated_captions)
        for r in rr:
            ss.append(r)
    return ss
def loss_cal(x, y, seq_len, lossfunction):
    batchsizee = len(x)
    predicted_cat = None
    ground_truth_cat = None
    flag = True
    for train_batch in range(batchsizee):
        predict = x[train_batch]
        ground_truth = y[train_batch]
        seq_length = seq_len[train_batch] - 1
        predict = predict[:seq_length]
        ground_truth = ground_truth[:seq_length]
        if flag:
            predicted_cat = predict
            ground_truth_cat = ground_truth
            flag = False
        else:
            predicted_cat = torch.cat((predicted_cat, predict), dim=0)
            ground_truth_cat = torch.cat((ground_truth_cat, ground_truth), dim=0)
    loss = lossfunction(predicted_cat, ground_truth_cat)
    avg_loss = loss / batchsizee
    return loss
def minibatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avidata, caption_seq = zip(*data)
    avidata = torch.stack(avidata, 0)
    seq_len = [len(cap) for cap in caption_seq]
    target = torch.zeros(len(caption_seq), max(seq_len)).long()
    for index, cap in enumerate(caption_seq):
        end = seq_len[index]
        target[index, :end] = cap[:end]
    return avidata, target, seq_len
def main():
    caption_file = './MLDS_hw2_1_data/training_data/feat'
    vidfeature_dir = './MLDS_hw2_1_data/training_label.json'
    indextoword,wordtoindex,dictnry = dictonaryFunc(4)
    train_dataset = Dataprocessor(caption_file, vidfeature_dir,dictnry, wordtoindex)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=minibatch)
    
    caption_file = './MLDS_hw2_1_data/testing_data/feat'
    vidfeature_dir = './MLDS_hw2_1_data/testing_label.json'
    test_dataset = Dataprocessor(caption_file,vidfeature_dir,dictnry, wordtoindex)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=minibatch)
   
    epochs_n = 20
    model_save_directory = (model_path)
    with open(pickel_file, 'wb') as f:
         pickle.dump(indextoword, f)

    x = len(indextoword)+4
    if not os.path.exists(model_save_directory):
        os.mkdir(model_save_directory)
    lossfunction = nn.CrossEntropyLoss()
    encode =model.Encoder_net()
    decode = model.Decoder_net(256, x, x, 768, 0.4)
    modeltrain = model.ModelMain(video_encoder = encode,caption_decoder = decode) 
    

    start = time.time()
    for epoch in range(epochs_n):
        train(modeltrain,epoch+1, train_loader=train_dataloader, loss_func=lossfunction)
        evaluate(test_dataloader, modeltrain)

    end = time.time()
    torch.save(modeltrain.state_dict(), "{}/{}.pth".format(model_save_directory, 'model1'))
    print("Training finished {}  elapsed time: {: .3f} seconds. \n".format('test', end-start))
if __name__=="__main__":
    main()
