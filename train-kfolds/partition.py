import pandas as pd
from sklearn import model_selection

df = pd.read_csv("../data/train.csv")
df = df.dropna().reset_index(drop = True)    # 过滤缺失数据并重置索引
df["kfold"] = -1    # 数据对折

df = df.sample(frac = 1).reset_index(drop = True)    # 随机采样

# 按sentiment分层采样
kf = model_selection.StratifiedKFold(n_splits = 5)
for fold, (train_set, validation_set) in enumerate(kf.split(X = df, y = df.sentiment.values)):
    print(len(train_set), len(validation_set))
    df.loc[validation_set, 'kfold'] = fold

df.to_csv("./train_5folds.csv", index = False)    # 存入指定路径