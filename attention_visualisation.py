import argparse

import pandas as pd
from transformers import BertTokenizer, BertLMHeadModel


def load_word_list(f_path):
    lst = []
    with open(f_path,'r') as f:
        line = f.readline()
        while line:
            lst.append(line.strip())
            line = f.readline()
    return lst

def load_word_tokenize(stereotype_word,tokenizer):
    lst = []
    for word in stereotype_word:
        # line_tokenize = tokenizer(word, padding=True, truncation=True, return_tensors="pt").input_ids[0]
        line_tokenize = tokenizer(word, truncation=True, return_tensors="pt").input_ids[0]
        print(line_tokenize)
        lst.append(line_tokenize)
    return lst

def clean_word_list2(tar1_words_,tar2_words_,tokenizer):
    tar1_words = []
    tar2_words = []
    for i in range(len(tar1_words_)):
        if tokenizer.convert_tokens_to_ids(tar1_words_[i])!=tokenizer.unk_token_id and tokenizer.convert_tokens_to_ids(tar2_words_[i])!=tokenizer.unk_token_id:
            tar1_words.append(tar1_words_[i])
            tar2_words.append(tar2_words_[i])
    return tar1_words, tar2_words

def get_stereotype_sentence(text_path,stereotype_word_list):
    store_data_path="./data/visual/stereotype_word_sentence.txt"
    with open(store_data_path,'w', encoding='utf-8') as fin:
        with open(text_path, 'r', encoding='utf-8') as fout:
            line = fout.readline()
            while line:
                line_list=line.lower().replace(".", "").replace("%", "").replace(":", "").replace("“", "").replace("–", " ").replace(")", "").replace("”", "").replace("$", "").replace("’", " ").replace("?", "").replace("'", "").replace(";", "").replace(",", "").replace("-", "").replace("(", "").replace('"', '').split(" ")
                for word in line_list:
                    if word in stereotype_word_list:
                        print(word)
                        fin.write(line)
                        break
                line = fout.readline()

def get_geneder_sentence(text_path, male_words,female_words):
    store_data_path = "./data/visual/gender_word_sentence.txt"
    with open(store_data_path, 'w', encoding='utf-8') as fin:
        with open(text_path, 'r', encoding='utf-8') as fout:
            line = fout.readline()
            while line:
                flag=False
                new_line=line
                line_list = line.lower().replace(".", "").replace("%", "").replace(":", "").replace("“", "").replace(
                    "–", " ").replace(")", "").replace("”", "").replace("$", "").replace("’", " ").replace("?","").replace(
                    "'", "").replace(";", "").replace(",", "").replace("-", "").replace("(", "").replace('"', '').split(" ")

                for index,word in enumerate(line_list):
                    if word in male_words:
                        word_index=male_words.index(word)
                        new_word=female_words[word_index]
                        if index==0:
                            new_line=new_line.replace(word.capitalize(),new_word.capitalize())
                        elif index==len(line_list)-1:
                            new_line=new_line.replace(" "+word," "+new_word)
                        else:
                            new_line=new_line.replace(" "+word+" "," "+new_word+" ")
                        flag=True
                    elif word in female_words:
                        word_index=female_words.index(word)
                        new_word=male_words[word_index]
                        if index == 0:
                            new_line = new_line.replace(word.capitalize(), new_word.capitalize())
                        elif index == len(line_list) - 1:
                            new_line = new_line.replace(" " + word, " " + new_word)
                        else:
                            new_line = new_line.replace(" " + word + " ", " " + new_word + " ")
                        flag=True
                if flag:
                    fin.write(line)
                    fin.write(new_line)
                line = fout.readline()




def list_token_tokenize(tokenizer,word_list):
    token_list=[]
    for word in word_list:
        token_list.append(tokenizer.convert_tokens_to_ids(word))
    return token_list


def pre_processed_data(text_path,stereotype_word,stereotype_data_path,male_words, female_words):
    get_stereotype_sentence(text_path,stereotype_word)
    get_geneder_sentence(stereotype_data_path, male_words, female_words)


def pre_attention_score(gender_sentence_path,stereotype_word,male_words, female_words,tokenizer,model):
    stereotype_token_list=list_token_tokenize(tokenizer, stereotype_word)
    male_token_list=list_token_tokenize(tokenizer, male_words)
    female_token_list=list_token_tokenize(tokenizer, female_words)

    attention_score_save_path="./data/visual/sakg/attention_score_save_sakg_1.txt"
    count=0
    with open(attention_score_save_path, 'w', encoding='utf-8') as fin:
        with open(gender_sentence_path, 'r', encoding='utf-8') as fout:
            line = fout.readline()
            while line:
                print(count)
                count+=1
                stereotype_token_index=[]
                male_token_index=[]
                female_token_index=[]

                line_one=line
                line_two=fout.readline()
                line_one_input=tokenizer(line_one, padding=True, truncation=True, return_tensors="pt")
                line_two_input = tokenizer(line_two, padding=True, truncation=True, return_tensors="pt")
                for key in line_one_input.keys():
                    line_one_input[key] = line_one_input[key].cuda()
                    line_two_input[key] = line_two_input[key].cuda()

                line_one_attention = model.bert(**line_one_input).attentions
                line_two_attention=model.bert(**line_two_input).attentions

                for index,token in enumerate(line_one_input.input_ids[0]):
                    if token in stereotype_token_list:
                        stereotype_token_index.append(index)
                    elif token in male_token_list:
                        male_token_index.append(index)
                    elif token in female_token_list:
                        female_token_index.append(index)

                for stereotype_index in stereotype_token_index:
                    attention_score_male=0
                    attention_score_female=0
                    for male_index in male_token_index:
                        for layer in range(len(line_one_attention)):  # 每一层
                            attention_one = line_one_attention[layer]
                            attention_score_male+=attention_one[:, :, stereotype_index, male_index].sum().item()
                            attention_two = line_two_attention[layer]
                            attention_score_female+=attention_two[:, :, stereotype_index, male_index].sum().item()

                    for female_index in female_token_index:
                        for layer in range(len(line_two_attention)):  # 每一层
                            attention_two = line_two_attention[layer]
                            attention_score_male+=attention_two[:, :, stereotype_index, female_index].sum().item()
                            attention_one = line_one_attention[layer]
                            attention_score_female+=attention_one[:, :, stereotype_index, female_index].sum().item()

                    fin.write(tokenizer.convert_ids_to_tokens(line_one_input.input_ids[0][stereotype_index].item())+"\t"+str(attention_score_male)+"\t"+str(attention_score_female))
                    fin.write("\n")

                line = fout.readline()



def count_attention_score():
    text_path="./data/visual/sakg/attention_score_save_sakg_1.txt"
    path="./data/visual/sakg/count_attention_score_sakg_1.txt"
    stereotype=set()
    stereotype_list=[]
    male_score=[]
    female_score=[]

    with open(text_path, 'r', encoding='utf-8') as fout:
        line = fout.readline()
        while line:
            line=line.split("\t")
            print(line)
            stereotype.add(line[0])
            stereotype_list.append(line[0])
            male_score.append(float(line[1]))
            female_score.append(float(line[2]))
            line=fout.readline()
    with open(path, 'w', encoding='utf-8') as fin:
        for stereotype_word in stereotype:
            count=0
            male_score_temp=0
            female_score_temp = 0
            for index,appear_stereotype_word in enumerate(stereotype_list):
                if stereotype_word==appear_stereotype_word:
                    count+=1
                    male_score_temp+=male_score[index]
                    female_score_temp+=female_score[index]
            fin.write(stereotype_word+"\t"+str(male_score_temp/count)+"\t"+str(female_score_temp/count))
            fin.write("\n")


def visual():
    path = "./data/visual/sakg/count_attention_score_sakg_1.txt"
    stereotype = set()
    stereotype_list=[]
    male_score=[]
    female_score=[]

    with open(path, 'r', encoding='utf-8') as fout:
        line = fout.readline()
        while line:
            line=line.split("\t")
            # print(line[1])
            print(line[2],end="")
            line=fout.readline()


if __name__ == "__main__":
    # path='F:\\model\\bert_original'

    path='./model/bert_32_5e-5_1_0.1_5000'
    tokenizer = BertTokenizer.from_pretrained(path)
    model = BertLMHeadModel.from_pretrained(path, output_attentions=True)
    model.eval()
    model.cuda()

    data_path = "./data/visual/"
    text_path = data_path + "News-Commentary.de-en.en"
    male_words = load_word_list(data_path + "male_word_list.txt")
    female_words = load_word_list(data_path + "female_word_list.txt")
    stereotype_word = load_word_list(data_path + "stereotype.txt")
    stereotype_data_path = data_path+"stereotype_word_sentence.txt"
    gender_sentence_path = data_path+"gender_word_sentence.txt"
    male_words, female_words = clean_word_list2(male_words, female_words, tokenizer)


    pre_processed_data(text_path,stereotype_word,stereotype_data_path,male_words, female_words)

    pre_attention_score(gender_sentence_path,stereotype_word,male_words, female_words,tokenizer,model)

    count_attention_score()
    visual()


