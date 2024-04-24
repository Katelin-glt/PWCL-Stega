import torch
import sys

import utils
import Huffman
import os
import lm
import numpy as np
from loguru import logger

import argparse

parser = argparse.ArgumentParser(description='HC')


parser.add_argument('-dataset', type=str, default=None, \
                    help='The training corpus [default:None]')
parser.add_argument('-generate-num', type=int, default=None, \
                    help='The number of generated stego text [default: None]')
parser.add_argument('-idx-gpu', type=str, default=None,\
                    help='the index of the gpu for runing [default:None]')
parser.add_argument('-seed', type=int, default=123, \
                    help='The random seed for initialization [default:123]')


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.idx_gpu

#Setting the random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def prob_sort(model, inp, pos=False):
    log_prob = model(inp)
    prob = torch.exp(log_prob)[:, -1 :].reshape(-1)
    prob[1] = 0
    probs = prob / prob.sum()
    probs, indices = prob.sort(descending=True)
    return probs, indices


def main(args):

    # ==================
    # hyper-parameters
    # ==================
    DATASET = args.dataset
    WORD_DROP = 10
    MIN_LEN = 5
    MAX_LEN = 200
    EMBED_SIZE = 800
    HIDDEN_DIM = 800
    NUM_LAYERS = 3
    DROPOUT_RATE = 0.8
    MAX_GENERATE_LENGTH = 200
    GENERATE_NUM = args.generate_num
    
    if DATASET == "IMDB":
        LOAD_EPOCH = 29
    if DATASET == "News":
        LOAD_EPOCH = 30
    if DATASET == 'Twitter':
        LOAD_EPOCH = 28

    all_var = locals()
    print()
    for var in all_var:
        if var != 'var_name':
            print("{0:15} ".format(var), all_var[var])
    print()

    # ===========
    # data
    # ===========
    data_path = './train_corpora/original/' + DATASET + '2022.txt'
    vocabulary = utils.Vocabulary(
                    data_path,
                    max_len = MAX_LEN,
                    min_len = MIN_LEN,
                    word_drop = WORD_DROP,
                    )

    # ===========
    # building model
    # ===========

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    print()
    model = lm.LM(
                    vocab_size = vocabulary.vocab_size,
                    embed_size = EMBED_SIZE,
                    hidden_dim = HIDDEN_DIM,
                    num_layers = NUM_LAYERS,
                    dropout_rate = DROPOUT_RATE
                    )
    model.to(device)
    model.load_state_dict(torch.load('models/' + DATASET + '-' + \
                            str(LOAD_EPOCH) + '.pkl', map_location=device))
    print('checkpoint loaded...')
    print()

    print('start steganography...')
    
    num_bits_list = [1,2,3,4,5]
    logger.add(DATASET+'_HC_{time}.log')
    for num_bits in num_bits_list:
        logger.info("num_bits is: " + str(num_bits))
        os.makedirs('stego/' + DATASET, exist_ok=True)
        # read bit streams
        with open('bit_stream/bit_stream.txt', 'r', encoding='utf8') as f:
            bit_stream = f.read().strip()
            bit_stream += bit_stream

        bit_index = int(torch.randint(0, high=1000, size=(1,)))
        bit_index_pos = bit_index

        model.eval()
        with torch.no_grad():
            stega_text = []
            stega_text_pos = []
            stega_bits = []
            # stega_bits_pos = []
            import time
            start = time.time()
            while len(stega_text) < GENERATE_NUM or len(stega_text_pos) < GENERATE_NUM:
                # if len(stega_text) % 10 == 0:
                #     Log = 'The number of generated: ' + str(len(stega_text))
                #     logger.info(Log)
                if len(stega_text) % 1000 == 0:
                    print('generated length: ', len(stega_text))
                stega_sentence = []
                stega_sentence_pos = []
                embed_bit = ''
                embed_bit_pos =''
                x = torch.LongTensor([[vocabulary.w2i['_BOS']]]).to(device)
                samp = model.sample(x) # 推理第一个词
                stega_sentence.append(vocabulary.i2w[samp.reshape(-1).cpu().numpy()[0]])
                stega_sentence_pos.append(vocabulary.i2w[samp.reshape(-1).cpu().numpy()[0]])

                x = torch.cat([x, samp], dim = 1)
                # print("xxxx!!: ", x, len(stega_text), len(stega_text_pos))
                x_pos = x
                for i in range(MAX_GENERATE_LENGTH - 1):
                    probs, indices = prob_sort(model, x)
                    probs_pos, indices_pos = prob_sort(model, x_pos)

                    C_probs = probs[:2**num_bits]       #Candidate probs 候选的概率
                    C_indices = indices[:2**num_bits]   #Candidate indiceso # 候选词索引

                    C_probs_pos = probs_pos[:2**num_bits]       #Candidate probs
                    C_indices_pos = indices_pos[:2**num_bits]   #Candidate indiceso

                    C_probs = list(C_probs.cpu().numpy())
                    C_probs_pos = list(C_probs_pos.cpu().numpy())

                    nodes = Huffman.createNodes(p for p in C_probs)
                    root = Huffman.createHuffmanTree(nodes)
                    codes = Huffman.huffmanEncoding(nodes, root)


                    nodes_pos = Huffman.createNodes(p for p in C_probs_pos)
                    root_pos = Huffman.createHuffmanTree(nodes_pos)
                    codes_pos = Huffman.huffmanEncoding_pos(nodes_pos, root_pos)


                    for b in range(2**int(num_bits)):
                        current_embed_bits = bit_stream[bit_index:bit_index+b+1]

                        if current_embed_bits in codes:
                            code_index = codes.index(current_embed_bits) # 当前bits在哈夫曼编码中的索引
                            gen = int(indices[code_index]) #候选词索引
                            embed_bit += current_embed_bits
                            bit_index = bit_index + b + 1
                            break
                    stega_sentence += [vocabulary.i2w[gen]]
                    for b in range(2**int(num_bits)):
                        current_embed_bits = bit_stream[bit_index_pos:bit_index_pos+b+1]

                        if current_embed_bits in codes_pos:
                            code_index_pos = codes_pos.index(current_embed_bits)
                            gen_pos = int(indices_pos[code_index_pos])
                            # embed_bit_pos += current_embed_bits
                            bit_index_pos = bit_index_pos + b + 1
                            break
                    stega_sentence_pos += [vocabulary.i2w[gen_pos]]

                    if vocabulary.i2w[gen] == '_EOS' or vocabulary.i2w[gen_pos] == '_EOS':
                        break
                    x = torch.cat([x, torch.LongTensor([[gen]]).to(device)], dim=1)
                    x_pos = torch.cat([x_pos, torch.LongTensor([[gen_pos]]).to(device)], dim=1)
                
                # check
                if '_EOS' in stega_sentence:
                    stega_sentence.remove('_EOS')
                if '_EOS' in stega_sentence_pos:
                    stega_sentence_pos.remove('_EOS')
                if (len(stega_sentence) <= MAX_LEN) and (len(stega_sentence) > MIN_LEN)\
                        and (len(stega_sentence_pos) <= MAX_LEN) and (len(stega_sentence_pos) > MIN_LEN):
                    stega_text.append(stega_sentence)
                    stega_bits.append(embed_bit)
                    stega_text_pos.append(stega_sentence_pos)
                    # stega_bits_pos.append(embed_bit_pos)

            # print("time_cost: ", time.time() - start)

            #write file
            words_count = 0
            bits_count = 0
            with open('stego/' + DATASET + '/HC_' + str(2**num_bits) + 'CW_cal_bits.txt', 'a', encoding = 'utf8') as f:
                for sentence in stega_text:
                    print(sentence, "length: ", len(sentence))
                    words_count += len(sentence)
                    f.write(' '.join(sentence) + '\n')
            # with open('stego/' + DATASET + '/HC_' + str(2**num_bits) + 'CW_pos.txt', 'a', encoding = 'utf8') as f:
            #     for sentence_pos in stega_text_pos:
            #         f.write(' '.join(sentence_pos) + '\n')
            with open('stego/' + DATASET + '/HC_' + str(2**num_bits) + 'CW_cal_bits.bit', 'a', encoding = 'utf8') as f:
                for bits in stega_bits:
                    print(bits, "length: ", len(bits))
                    bits_count += len(bits)
                    f.write(bits + '\n')

            print(num_bits, "bpw: ", bits_count / words_count)

            # with open('stego/' + DATASET + '/HC_' + str(2**num_bits) + 'CW_pos.bit', 'a', encoding = 'utf8') as f:
            #     for bits in stega_bits_pos:
            #         f.write(bits + '\n')


if __name__ == '__main__':
    main(args)
