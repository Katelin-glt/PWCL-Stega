import utils
import lm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch

def main():
    # ===================
    # hyper-parameters
    #====================
    DATASET = 'Twitter'
    WORD_DROP = 10 # 丢弃出现次数小于10次的词
    MIN_LEN = 5
    MAX_LEN = 200
    EMBED_SIZE = 800
    HIDDEN_DIM = 800
    NUM_LAYERS = 3
    DROPOUT_RATE = 0.5
    MAX_GENERATE_LENGTH = 200
    GENERATE_NUM = 90000

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
                    ) # 定义一个词典实例


    # ===============
    # building model
    # ===============

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    print()
    model = lm.LM(
                    vocab_size = vocabulary.vocab_size,
                    embed_size = EMBED_SIZE,
                    hidden_dim = HIDDEN_DIM,
                    num_layers = NUM_LAYERS,
                    dropout_rate = DROPOUT_RATE
                    ) # 定义语言模型
    model.to(device) # 将模型放入GPU
    total_params = sum(p.numel() for p in model.parameters())
    print("Total params: {:d}".format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() \
                                 if p.requires_grad) # 所有可训练的参数
    print("Trainable_params: {:d}".format(total_trainable_params))
    model.load_state_dict(torch.load('models/' + DATASET + '-' +\
                            str(LOAD_EPOCH) + '.pkl', map_location=device)) # 加载模型
    print('checkpoint loaded...')
    print()

    print('start generating normal texts....')
    os.makedirs('stego/' + DATASET, exist_ok=True)

    model.eval() # 将模型设置为评估模式
    with torch.no_grad(): # 禁用梯度下降
        norm_text = []
        norm_text_pos = []
        while len(norm_text) < GENERATE_NUM:
            if len(norm_text) % 1000 == 0:
                print('the length of norm_text: ', len(norm_text))
            norm_sentence = []
            norm_sentence_pos = []
            x = torch.LongTensor([[vocabulary.w2i['_BOS']]]).to(device)
            x_pos = x
            samp = model.sample(x)
            samp_pos = samp
            norm_sentence.append(vocabulary.i2w[samp.reshape(-1).cpu().numpy()[0]])
            norm_sentence_pos.append(vocabulary.i2w[samp.reshape(-1).cpu().numpy()[0]])
            for i in range(MAX_GENERATE_LENGTH - 1):
                if '_EOS' in norm_sentence:
                    break
                if '_EOS' in norm_sentence_pos:
                    break
                x = torch.cat([x, samp], dim = 1)
                x_pos = torch.cat([x_pos, samp_pos], dim = 1)
                samp = model.sample(x)
                samp_pos = model.sample(x_pos)

                norm_sentence.append(vocabulary.i2w[samp.reshape(-1).cpu().numpy()[0]])
                norm_sentence_pos.append(vocabulary.i2w[samp_pos.reshape(-1).cpu().numpy()[0]])

            # check
            if '_EOS' in norm_sentence:
                norm_sentence.remove('_EOS')
            if '_EOS' in norm_sentence_pos:
                norm_sentence_pos.remove('_EOS')
            if (len(norm_sentence) <= MAX_LEN) and (len(norm_sentence) >= MIN_LEN):
                norm_text.append(norm_sentence)
                norm_text_pos.append(norm_sentence_pos)

        # write files
        with open('stego/' + DATASET + '/No_embed.txt', 'a', encoding='utf8') as f:
            for sentence in norm_text:
                f.write(' '.join(sentence) + '\n')
            
        with open('stego/' + DATASET + '/No_embed_pos.txt', 'a', encoding='utf8') as f:
            for sentence_pos in norm_text_pos:
                f.write(' '.join(sentence_pos) + '\n')


if __name__ == '__main__':
    main()
