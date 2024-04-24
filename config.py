corpus = "IMDB"
stego_method = "AC"
dataset = "1bpw"

# 可能的可调参数，可以调
tuning_param = ["lambda_loss", "kl_weight", "temperature", "batch_size", "main_learning_rate", "nepoch",  "SEED"]  # list of possible paramters to be tuned can be increased

lambda_loss = [0.9]
kl_weight = [3.0]
temperature = [0.3]
batch_size = [8]
main_learning_rate = [1e-05]
decay = 1e-02
nepoch = [10]
hidden_size = 768

loss_type = "lcl_drop"  # ['ce', 'scl', 'lcl', 'lcl_drop']
model_type = "electra"  # ['electra', 'bert']
is_waug = True

SEED = [0] # for repeated results


param = {"corpus": corpus, "stego_method": stego_method, "dataset": dataset, "temperature": temperature,
         "batch_size": batch_size, "main_learning_rate": main_learning_rate, "decay": decay, "nepoch": nepoch,
         "hidden_size": hidden_size, "lambda_loss": lambda_loss, "loss_type": loss_type, "model_type": model_type,
         "is_waug": is_waug, "kl_weight": kl_weight, "SEED": SEED}