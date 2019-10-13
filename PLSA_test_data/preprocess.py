#coding=utf8
import numpy as np
import os
import json

def preprocess(filepath, output_path, stop_words, min_fq = 5):
    matrix_path = os.path.join(output_path, 'data.txt') 
    voc_path = os.path.join(output_path, 'voc.json')
    
    line_num = 0
    word_dict = {} 
    with open(filepath, 'r') as f:
         #过滤停用词
         for line in f.readlines():
             line_num += 1 
             words = line.strip().split()
             for w in words:
                if w in stop_words:
                   continue
                if w not in word_dict:
                   word_dict[w] = 1
                else:
                   word_dict[w] += 1
    
    #构建词典
    word2idx = {}
    idx2word = {}
    
    voc = {}   
    word_num = 0
    for key, value in word_dict.items():
        if value < min_fq:
           continue
   
        word2idx[key] = word_num
        idx2word[word_num] = key
  
        word_num += 1
 
    voc['word2idx'] = word2idx
    voc['idx2word'] = idx2word
    with open(voc_path, 'w') as f:
         json.dump(voc, f)
    
    assert word_num == len(word2idx.keys())
    
    #构建共现矩阵
    comatrix = np.zeros((line_num, word_num), dtype=np.int32) 
    with open(filepath, 'r') as f:
         for idx, line in enumerate(f.readlines()):
             words = line.strip().split()
             for word in words:
                 if word in word2idx:
                    comatrix[idx, word2idx[word]] += 1
         np.savetxt(matrix_path, comatrix) 


if __name__ == "__main__":
   """
     mkdir data
     python preprocess.py text.txt data
   """
   import sys
    
   file_path = sys.argv[1]
   output_path = sys.argv[2]

   stop_words = ['the', 'to', 'of', 'a', 'and', 'i', 'in', 'is', 'that', 'it', 'for', 'you', 'on', 's', 'this', 'be', 'are', 'not', 'have', 'with', 't', 
                           'as', 'or', 'if', 'was', 'but', 'they', 'can', 'from', 'by', 'at', 're', 'an', 'what', 'there', 'my', 'all', 'will', 'we', 'one', 'would', 
                           'do', 'he', 'about', 'x', 'so', 'your', 'no', 'has', 'any', 'me', 'some', 'who', 'out', 'which', 'don', 'more', 'like', 
                           'when', 'just', 'their', 'm', 'were', 'up', 'how', 'other', 'only', 'them', 'than', 'had', 'been', 'his', 'c', 
                           'also', 'does', 'd', 'then', 'these', 'should', 'could', 'well', 'am', 'because', 'even', 'why', 'very', 'may', 'now', 'us', 'apr', 'into', 
                           'two', 'way', 'first', 'many', 'e', 'make', 'much', 'most', 'such', 'those', 'right', 'where', 'say', 'here', 've', 'want', 'our', 'anyone', 
                           'used', 'said', 'being', 'its', 'did', 'same', 'after', 'over', 'need', 'b', 'r', 'too', 'something', 'o', 'please', 
                           'really', 'him', 'ca', 'off', 'since', 'back', 'believe', 'still', 'going', 'p', 'using', 'find', 'n', 
                           'thanks', 'last', 'before', 'must', 'll', 'never', 'while', 'things', 'might', 'own', 'both', 'another', 
                           'sure', 'without', 'etc', 'down', 'key', 'through', 'got', 'made', 'k', 'let', 'someone', 'under', 'u', 'doesn'] 

   preprocess(file_path, output_path, stop_words)
   
