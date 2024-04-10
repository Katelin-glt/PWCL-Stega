corpus = "Twitter"
stego_method = "VLC"
dataset = ["5bpw"]

# 可能的可调参数，可以调
tuning_param = ["lambda_loss", "main_learning_rate", "batch_size", "nepoch", "temperature", "SEED", "dataset"]  # list of possible paramters to be tuned can be increased
lambda_loss = [0.5]  # 原参数0.5
temperature = [0.3]  # 原参数0.3
batch_size = [16]
decay = 1e-02
main_learning_rate = [2e-05]  # 原参数2e-05

hidden_size = 768
nepoch = [10]
loss_type = "scl"  # ['ce', 'scl', 'lcl', 'lcl_drop']
model_type = "electra"  # ['electra', 'bert']
is_waug = True
label_list = [None]
SEED = [0]


param = {"temperature": temperature, "corpus": corpus, "stego_method": stego_method, "dataset": dataset,
         "main_learning_rate": main_learning_rate, "batch_size": batch_size, "hidden_size": hidden_size,
         "nepoch": nepoch, "dataset": dataset, "lambda_loss": lambda_loss, "loss_type": loss_type,
         "decay": decay, "SEED": SEED, "model_type": model_type, "is_waug": is_waug}