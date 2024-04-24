import sys
import torch
import utils
import lm
import os
import argparse


parser = argparse.ArgumentParser(description='ADG')
parser.add_argument('-dataset', type=str, default='Twitter',  required=True,\
                    help='The training corpus [default:None]')
parser.add_argument('-generate-num', type=int, default=10, required=True,\
                    help='The number of generated stego text [default:None]')
parser.add_argument('-idx-gpu', type=str, default=0, required=True,\
                    help='The index of the gpu for runing [default:None]')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.idx_gpu


def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit*(2**i)
    return res


def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}'%num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]


def near(alist, anum):
    up = len(alist) - 1
    if up == 0:
        return 0
    bottom = 0
    while up - bottom > 1:
        index = int((up + bottom) / 2)
        if alist[index] < anum:
            up = index
        elif alist[index] > anum:
            bottom = index
        else:
            return index

    if up - bottom == 1:
        if alist[bottom] - anum < anum - up:
            index = bottom
        else:
            index = up
    return index


def main():
    # ===================
    # hyper-parameters
    #====================
    DATASET = args.dataset
    WORD_DROP = 10
    MIN_LEN = 5
    MAX_LEN = 200
    EMBED_SIZE = 800
    HIDDEN_DIM = 800
    NUM_LAYERS = 3
    DROPOUT_RATE = 0.0
    MAX_GENERATE_LENGTH = 200
    GENERATE_NUM = args.generate_num

    if DATASET == 'IMDB':
        LOAD_EPOCH = 29
    if DATASET == 'News':
        LOAD_EPOCH = 30
    if DATASET == 'Twitter':
        LOAD_EPOCH = 28


    all_var = locals()
    print()
    for var in all_var:
        if var != "var_name":
            print("{0:15} ".format(var), all_var[var])
    print()

    # ========
    # data
    # ========
    data_path = './train_corpora/original/' + DATASET + '2022.txt'
    vocabulary = utils.Vocabulary(
                    data_path,
                    max_len = MAX_LEN,
                    min_len = MIN_LEN,
                    word_drop = WORD_DROP
                    )


    # ===============
    # building model
    # ===============

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    model = lm.LM(
                    vocab_size = vocabulary.vocab_size,
                    embed_size = EMBED_SIZE,
                    hidden_dim = HIDDEN_DIM,
                    num_layers = NUM_LAYERS,
                    dropout_rate = DROPOUT_RATE
                    )
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total params: {:d}".format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable_params: {:d}".format(total_trainable_params))
    model.load_state_dict(torch.load('models/' + DATASET + '-' +\
                            str(LOAD_EPOCH) + '.pkl', map_location=device))
    print('checkpoint loaded...')
    print()

    # ========================
    # starting steganography
    # ========================
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
        stega_bits_pos = []

        while len(stega_text) < GENERATE_NUM or len(stega_text_pos) < GENERATE_NUM:
            if len(stega_text) % 10 == 0:
                print('the length of stega_text: ', len(stega_text))
            stega_sentence = []
            stega_sentence_pos = []
            stega_bit = ''
            stega_bit_pos = ''
            x = torch.LongTensor([[vocabulary.w2i['_BOS']]]).to(device)
            samp = model.sample(x)
            stega_sentence.append(vocabulary.i2w[samp.reshape(-1).cpu().numpy()[0]])
            stega_sentence_pos.append(vocabulary.i2w[samp.reshape(-1).cpu().numpy()[0]])

            x = torch.cat([x, samp], dim = 1)
            x_pos = x

            for i in range(MAX_GENERATE_LENGTH - 1):
                if '_EOS' in stega_sentence or '_EOS' in stega_sentence_pos:
                    break
                # conditional probability distribution
                log_prob = model(x)
                log_prob_pos = model(x_pos)
                prob = torch.exp(log_prob)[:, -1, :].reshape(-1)
                prob_pos = torch.exp(log_prob_pos)[:, -1, :].reshape(-1)

                prob[1] = 0
                prob = prob / prob.sum()
                prob, indices = prob.sort(descending=True)

                prob_pos[1] = 0
                prob_pos = prob_pos / prob_pos.sum()
                prob_pos, indices_pos = prob_pos.sort(descending=True)

                # start recursion
                bit_tmp = 0
                bit_tmp_pos = 0
                while prob[0] <= 0.5:
                    # embedding bit
                    bit = 1
                    while (1 / 2 ** (bit + 1)) > prob[0]:
                        bit += 1
                    mean = 1 / 2 ** bit
                    # dp
                    prob = prob.tolist()
                    indices = indices.tolist()
                    result = []

                    for i in range(2 ** bit):
                        result.append([[], []])

                    for i in range(2 ** bit -1):
                        result[i][0].append(prob[0])
                        result[i][1].append(indices[0])
                        del (prob[0])
                        del (indices[0])

                        while sum(result[i][0]) < mean:
                            delta = mean - sum(result[i][0])
                            index = near(prob, delta)

                            if prob[index] - delta < delta:
                                result[i][0].append(prob[index])
                                result[i][1].append(indices[index])
                                del (prob[index])
                                del (indices[index])
                            else:
                                break
                        result[2 ** bit - 1][0].extend(prob)
                        result[2 ** bit - 1][1].extend(indices)

                        mean = sum(prob) / (2 ** bit - i - 1)

                    # read secret message
                    bit_embed = [int(_) for _ in bit_stream[bit_index + \
                                    bit_tmp:bit_index + bit_tmp + bit]]
                    int_embed = bits2int(bit_embed)

                    # updating
                    prob = torch.FloatTensor(result[int_embed][0]).to(device)
                    indices = torch.LongTensor(result[int_embed][1]).to(device)

                    prob = prob / prob.sum()
                    prob, _ = prob.sort(descending=True)
                    indices = indices[_]
                    bit_tmp =+ bit


                while prob_pos[0] <= 0.5:
                    # embedding bit
                    bit_pos = 1
                    while (1 / 2 ** (bit_pos + 1)) > prob_pos[0]:
                        bit_pos += 1
                    mean_pos = 1 / 2 ** bit_pos
                    # dp
                    prob_pos = prob_pos.tolist()
                    indices_pos = indices_pos.tolist()
                    result_pos = []

                    for i in range(2 ** bit_pos):
                        result_pos.append([[], []])

                    for i in range(2 ** bit_pos -1):
                        result_pos[i][0].append(prob_pos[0])
                        result_pos[i][1].append(indices_pos[0])
                        del (prob_pos[0])
                        del (indices_pos[0])

                        while sum(result_pos[i][0]) < mean_pos:
                            delta_pos = mean_pos - sum(result_pos[i][0])
                            index_pos = near(prob_pos, delta_pos)

                            if prob_pos[index_pos] - delta_pos < delta_pos:
                                result_pos[i][0].append(prob_pos[index_pos])
                                result_pos[i][1].append(indices_pos[index_pos])
                                del (prob_pos[index_pos])
                                del (indices_pos[index_pos])
                            else:
                                break
                        result_pos[2 ** bit_pos - 1][0].extend(prob_pos)
                        result_pos[2 ** bit_pos - 1][1].extend(indices_pos)

                        mean_pos = sum(prob_pos) / (2 ** bit_pos - i - 1)

                    # read secret message
                    bit_embed_pos = [int(_) for _ in bit_stream[bit_index_pos + \
                                    bit_tmp_pos:bit_index_pos + bit_tmp_pos + bit_pos]]
                    # for bit in bit_embed:
                    #     bit_embed_pos.append(0 if bit == 1 else 1)
                    # print("bit_embed_pos: ", bit_embed_pos)
                    int_embed_pos = bits2int(bit_embed_pos)

                    # updating
                    prob_pos = torch.FloatTensor(result_pos[int_embed_pos][0]).to(device)
                    indices_pos = torch.LongTensor(result_pos[int_embed_pos][1]).to(device)

                    prob_pos = prob_pos / prob_pos.sum()
                    prob_pos, _ = prob_pos.sort(descending=True)
                    indices_pos = indices_pos[_]
                    bit_tmp_pos =+ bit_pos

                # terminate
                gen = int(indices[int(torch.multinomial(prob, 1))])
                stega_sentence += [vocabulary.i2w[gen]]
                gen_pos = int(indices_pos[int(torch.multinomial(prob_pos, 1))])
                stega_sentence_pos += [vocabulary.i2w[gen_pos]]

                if vocabulary.i2w[gen] == '_EOS' or vocabulary.i2w[gen_pos] == '_EOS':
                    break
                x = torch.cat([x, torch.LongTensor([[gen]]).to(device)], dim = 1).to(device)
                x_pos = torch.cat([x_pos, torch.LongTensor([[gen_pos]]).to(device)], dim = 1).to(device)
                stega_bit += bit_stream[bit_index:bit_index + bit_tmp]
                stega_bit_pos += bit_stream[bit_index_pos:bit_index_pos + bit_tmp_pos]
                bit_index += bit_tmp
                bit_index_pos += bit_tmp_pos


            # check
            if '_EOS' in stega_sentence:
                stega_sentence.remove('_EOS')
            if '_EOS' in stega_sentence_pos:
                stega_sentence_pos.remove('_EOS')
            if ((len(stega_sentence) <= MAX_LEN) and (len(stega_sentence) >= MIN_LEN))\
                    and ((len(stega_sentence_pos) <= MAX_LEN) and (len(stega_sentence_pos) >= MIN_LEN)):
                print("stega_sentence: ", stega_sentence)
                print("stega_sentence_pos: ", stega_sentence_pos)
                with open('stego/' + DATASET + '/adg.txt', 'a', encoding='utf8') as f:
                    f.write(' '.join(stega_sentence) + '\n')
                with open('stego/' + DATASET + '/adg_pos.txt', 'a', encoding='utf8') as f:
                    f.write(' '.join(stega_sentence_pos) + '\n')
                with open('stego/' + DATASET + '/adg.bit', 'a', encoding='utf8') as f:
                    f.write(stega_bit + '\n')
                stega_text.append(stega_sentence)
                stega_bits.append(stega_bit)
                stega_text_pos.append(stega_sentence_pos)

        # write files
        # with open('stego/' + DATASET + '/adg.txt', 'a', encoding='utf8') as f:
        #     for sentence in stega_text:
        #         f.write(' '.join(sentence) + '\n')
        # with open('stego/' + DATASET + '/adg_pos.txt', 'a', encoding='utf8') as f:
        #     for sentence_pos in stega_text_pos:
        #         f.write(' '.join(sentence_pos) + '\n')
        # with open('stego/' + DATASET + '/adg.bit', 'w', encoding='utf8') as f:
        #     for bits in stega_bits:
        #         f.write(bits + '\n')


if __name__ == '__main__':
    main()
