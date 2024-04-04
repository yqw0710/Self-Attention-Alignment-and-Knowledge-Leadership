import argparse
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from utils import *
import torch.nn.functional as F
from transformers import AdamW
from transformers import BertTokenizer, BertForPreTraining, BertLMHeadModel
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel
from transformers import AlbertTokenizer, AlbertForPreTraining


def JSDivergence(hidden_1, hidden_2):
    h1 = F.softmax(hidden_1, dim=-1)
    h2 = F.softmax(hidden_2, dim=-1)

    avg_hidden = (h1 + h2) / 2.0

    jsd = 0.0
    jsd += F.kl_div(input=F.log_softmax(hidden_1, dim=-1), target=avg_hidden, reduction="batchmean")
    jsd += F.kl_div(input=F.log_softmax(hidden_2, dim=-1), target=avg_hidden, reduction="batchmean")

    return jsd / 2.0


def get_bias_loss(jsd_runner: JSDivergence, hidden_1: torch.FloatTensor, hidden_2: torch.FloatTensor):
    bias_hidden_jsd = jsd_runner(hidden_1=hidden_1, hidden_2=hidden_2)
    bias_hidden_cossim = F.cosine_similarity(hidden_1, hidden_2).mean()

    # print(bias_hidden_jsd,bias_hidden_cossim)
    return bias_hidden_jsd - bias_hidden_cossim


def get_lm_loss(hidden_1: torch.FloatTensor, hidden_2: torch.FloatTensor):
    """Get KLD and cosine similarity loss for non-stereotype inputs. """

    lm_hidden_kld = F.kl_div(
        input=F.log_softmax(hidden_1, dim=-1), target=F.softmax(hidden_2, dim=-1), reduction="batchmean"
    )
    lm_hidden_cossim = F.cosine_similarity(hidden_1, hidden_2).mean()

    # print(lm_hidden_kld,lm_hidden_cossim)
    return lm_hidden_kld - lm_hidden_cossim


def get_sentence_pair(tokenizer):
    tar1_sen = []
    tar2_sen = []
    diff = []
    tar_neutral_sen = []
    sentence1_path = "./data/alignment/sentence_all_1.txt"
    sentence2_path = "./data/alignment/sentence_all_2.txt"
    neutral_path = "./data/alignment/no_stereotype_data.tsv"

    with open(sentence1_path, 'r', encoding='utf-8') as fin:
        with open(sentence2_path, 'r', encoding='utf-8') as finn:
            for line1, line2 in zip(fin, finn):
                tokenized1 = tokenizer(line1, padding=True, truncation=True, return_tensors="pt").input_ids[0]
                tokenized2 = tokenizer(line2, padding=True, truncation=True, return_tensors="pt").input_ids[0]
                for i in range(len(tokenized1)):
                    if tokenized1[i] != tokenized2[i]:
                        diff.append(i)
                        tar1_sen.append(line1)
                        tar2_sen.append(line2)
                        break

    with open(neutral_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            tar_neutral_sen.append(line)

    tar_neutral_sen = tar_neutral_sen[: len(tar1_sen)]

    # diff=torch.tensor(diff)
    tar1_tokenized = tokenizer(tar1_sen, padding=True, truncation=True, return_tensors="pt")
    tar2_tokenized = tokenizer(tar2_sen, padding=True, truncation=True, return_tensors="pt")
    tar_neutral_tokenized = tokenizer(tar_neutral_sen, padding=True, truncation=True, return_tensors="pt")
    tar1_tokenized, tar2_tokenized, tar_neutral_tokenized = send_to_cuda(tar1_tokenized, tar2_tokenized,
                                                                         tar_neutral_tokenized)

    diff = np.array(diff)
    return tar1_tokenized, tar2_tokenized, diff, tar_neutral_tokenized


def send_to_cuda(tar1_tokenized, tar2_tokenized, tar_neutral_tokenized):
    for key in tar1_tokenized.keys():
        tar1_tokenized[key] = tar1_tokenized[key].cuda()
        tar2_tokenized[key] = tar2_tokenized[key].cuda()
    for key in tar_neutral_tokenized.keys():
        tar_neutral_tokenized[key] = tar_neutral_tokenized[key].cuda()
    return tar1_tokenized, tar2_tokenized, tar_neutral_tokenized


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_name_or_path",
    default="bert-base-uncased",
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)

parser.add_argument(
    "--model_type",
    default="bert",
    type=str,
    help="choose from ['bert','roberta','albert']",
)

parser.add_argument(
    "--data_path",
    default="data/alignment",
    type=str,
    help="data path to put the taget/attribute word list",
)

parser.add_argument(
    "--batch_size",
    default=4,
    type=int,
    help="batch size in auto-debias fine-tuning",
)

parser.add_argument(
    "--lr",
    default=5e-5,
    type=float,
    help="learning rate in auto-debias fine-tuning",
)

parser.add_argument(
    "--epochs",
    default=1,
    type=int,
    help="number of epochs in auto-debias fine-tuning",
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.model_type == 'bert':
        tokenizer_path = './model/bert'
        model = BertLMHeadModel.from_pretrained(args.model_name_or_path, output_attentions=True)
        original_model = BertLMHeadModel.from_pretrained(args.model_name_or_path)
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    elif args.model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        model = RobertaForMaskedLM.from_pretrained(args.model_name_or_path)
        new_roberta = RobertaModel.from_pretrained(args.model_name_or_path)  # make the add_pooling_layer=True
        model.roberta = new_roberta
    elif args.model_type == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained(args.model_name_or_path)
        model = AlbertForPreTraining.from_pretrained(args.model_name_or_path)
    else:
        raise NotImplementedError("not implemented!")
    model.train()
    model.cuda()
    original_model.cuda()

    tar1_tokenized, tar2_tokenized, diff, tar_neutral_tokenized = get_sentence_pair(tokenizer)

    assert tar1_tokenized['input_ids'].shape[0] == tar2_tokenized['input_ids'].shape[0]
    data_len = tar1_tokenized['input_ids'].shape[0]
    idx_ds = DataLoader([i for i in range(data_len)], batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    for i in range(1, args.epochs + 1):
        num = 0
        for idx in tqdm(idx_ds):
            num += 1
            diff_idx = diff[idx]
            tar1_inputs = {}
            tar2_inputs = {}
            tar_neutral_inputs = {}
            for key in tar1_tokenized.keys():  # key:input_ids token_type_ids  attention_mask
                tar1_inputs[key] = tar1_tokenized[key][idx]
                tar2_inputs[key] = tar2_tokenized[key][idx]
                tar_neutral_inputs[key] = tar_neutral_tokenized[key][idx]
            optimizer.zero_grad()
            tar1_attentions = model.bert(**tar1_inputs).attentions
            tar2_attentions = model.bert(**tar2_inputs).attentions

            tar_neutral_biased_embedding = model.bert(**tar_neutral_inputs).last_hidden_state
            tar_neutral_original_embedding = original_model.bert(**tar_neutral_inputs).last_hidden_state

            # attention1_score = []
            # attention2_score = []
            bias_loss = 0
            for layer in range(len(tar1_attentions)):  # 每一层

                attention1 = tar1_attentions[layer]
                attention2 = tar2_attentions[layer]  # [batch_size,12head,len,len]

                attention1_score = attention1[torch.arange(attention1.size(0)), :, :, diff_idx]
                attention2_score = attention2[torch.arange(attention2.size(0)), :, :, diff_idx]
                bias_loss += get_bias_loss(jsd_runner=JSDivergence, hidden_1=attention1_score,
                                           hidden_2=attention2_score)

                # attention1_score.append(attention1[torch.arange(attention1.size(0)), :, :, diff_idx])
                # attention2_score.append(attention2[torch.arange(attention2.size(0)), :, :, diff_idx])  # [batch_size,12head,len]

            # attention1_score = torch.stack(attention1_score)
            # attention2_score = torch.stack(attention2_score)

            # bias_loss=get_bias_loss(jsd_runner=JSDivergence, hidden_1=attention1_score,hidden_2=attention2_score)
            embed_loss = get_lm_loss(hidden_1=tar_neutral_biased_embedding, hidden_2=tar_neutral_original_embedding)

            loss = bias_loss + 2 * embed_loss

            # print(bias_loss)
            # exit(0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('loss {}'.format(loss))
            if num % 10000 == 0:
                model.save_pretrained('model/alignment/bert/1020/bert_32_5e-5_{}_0.1_{}'.format(i, num))
            # if num>40000:
            #     break
        print("model_save")
        model.save_pretrained('model/alignment/bert/1020/bert_32_5e-5_{}_0.1'.format(i))

