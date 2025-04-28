import pandas as pd
import torch
import argparse
import h5py
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# METABRIC
parser = argparse.ArgumentParser(description='Data Gen')

parser.add_argument('--dataset', type=str, default='support', choices=['support', 'mimic', 'sequence'])
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

def load_datasets(dataset_file):
    datasets = defaultdict(dict)
    
    with h5py.File(dataset_file, 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]
                
    return datasets

if args.dataset == 'support':
    support = load_datasets("./data/support/support_train_test.h5")
    
    support_train = support['train']
    support_test = support['test']

    support_train_t = support_train['t']
    support_train_e = support_train['e']
    support_train_x = support_train['x']

    support_test_t = support_test['t']
    support_test_e = support_test['e']
    support_test_x = support_test['x']

    support_t = torch.concat([torch.tensor(support_train_t), torch.tensor(support_test_t)])
    support_e = torch.concat([torch.tensor(support_train_e), torch.tensor(support_test_e)])
    support_x = torch.concat([torch.tensor(support_train_x), torch.tensor(support_test_x)])

    support_t_np = support_t.numpy()
    support_e_np = support_e.numpy()
    support_x_np = support_x.numpy()

    t_train, t_temp, e_train, e_temp, x_train, x_temp = train_test_split(
        support_t_np, support_e_np, support_x_np, test_size=0.4, stratify=support_e_np, random_state=args.seed
    )

    t_valid, t_test, e_valid, e_test, x_valid, x_test = train_test_split(
        t_temp, e_temp, x_temp, test_size=0.5, stratify=e_temp, random_state=args.seed
    )

    scaler1 = StandardScaler()
    support_train_x = scaler1.fit_transform(x_train)
    support_valid_x = scaler1.transform(x_valid)
    support_test_x = scaler1.transform(x_test)

    scaler2 = MinMaxScaler()
    support_train_t = scaler2.fit_transform(t_train.reshape(-1, 1))
    support_train_t = support_train_t.squeeze()
    support_valid_t = scaler2.transform(t_valid.reshape(-1, 1))
    support_valid_t = support_valid_t.squeeze()
    support_test_t = scaler2.transform(t_test.reshape(-1, 1))
    support_test_t = support_test_t.squeeze()

    support_train_t = torch.tensor(support_train_t)
    support_train_e = torch.tensor(e_train)
    support_train_x = torch.tensor(support_train_x)

    support_valid_t = torch.tensor(support_valid_t)
    support_valid_e = torch.tensor(e_valid)
    support_valid_x = torch.tensor(support_valid_x)

    support_test_t = torch.tensor(support_test_t)
    support_test_e = torch.tensor(e_test)
    support_test_x = torch.tensor(support_test_x)

    print("dataset:", args.seed)
    print("mean time:", torch.mean(support_train_t), torch.mean(support_valid_t), torch.mean(support_test_t))
    print("train:", support_train_x.shape, "validation:", support_valid_x.shape, "test:", support_test_x.shape)
    print(str(args.dataset) + " split by", support_train_t.shape[0], support_valid_t.shape[0], support_test_t.shape[0])
    print("training censoring rate:", 1 - (support_train_e.sum()/support_train_e.shape[0]).item())
    print("validation censoring rate:", 1 - (support_valid_e.sum()/support_valid_e.shape[0]).item())
    print("test censoring rate:", 1 - (support_test_e.sum()/support_test_e.shape[0]).item())
    
    torch.save(support_train_t, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_train_t.pt')
    torch.save(support_train_e, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_train_e.pt')
    torch.save(support_train_x, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_train_x.pt')
    
    torch.save(support_valid_t, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_valid_t.pt')
    torch.save(support_valid_e, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_valid_e.pt')
    torch.save(support_valid_x, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_valid_x.pt')
    
    torch.save(support_test_t, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_test_t.pt')
    torch.save(support_test_e, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_test_e.pt')
    torch.save(support_test_x, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_test_x.pt')

elif args.dataset == 'sequence':
    # split by 3/1/1
    sequence_train = pd.read_csv("./data/sequence/TR_SEQ.csv")
    sequence_test = pd.read_csv("./data/sequence/TS_SEQ.csv")
    
    sequence = pd.concat([sequence_train, sequence_test], axis=0).reset_index(drop=True)

    sequence_t = torch.tensor(sequence['futime'].values)

    X = sequence.loc[:, 'x1':'x24'].values
    t = sequence['futime'].values
    e = sequence['status'].values

    X_train, X_temp, t_train, t_temp, e_train, e_temp = train_test_split(
        X, t, e, test_size=0.4, stratify=e, random_state=args.seed
    )

    X_valid, X_test, t_valid, t_test, e_valid, e_test = train_test_split(
        X_temp, t_temp, e_temp, test_size=0.5, stratify=e_temp, random_state=args.seed
    )

    scaler1 = StandardScaler()
    sequence_train_x = scaler1.fit_transform(X_train)
    sequence_valid_x = scaler1.transform(X_valid)
    sequence_test_x = scaler1.transform(X_test)

    scaler2 = MinMaxScaler()
    sequence_train_t = scaler2.fit_transform(t_train.reshape(-1, 1))
    sequence_train_t = sequence_train_t.squeeze()
    sequence_valid_t = scaler2.transform(t_valid.reshape(-1, 1))
    sequence_valid_t = sequence_valid_t.squeeze()
    sequence_test_t = scaler2.transform(t_test.reshape(-1, 1))
    sequence_test_t = sequence_test_t.squeeze()

    sequence_train_t = torch.tensor(sequence_train_t)
    sequence_train_e = torch.tensor(e_train)
    sequence_train_x = torch.tensor(sequence_train_x)

    sequence_valid_t = torch.tensor(sequence_valid_t)
    sequence_valid_e = torch.tensor(e_valid)
    sequence_valid_x = torch.tensor(sequence_valid_x)

    sequence_test_t = torch.tensor(sequence_test_t)
    sequence_test_e = torch.tensor(e_test)
    sequence_test_x = torch.tensor(sequence_test_x)

    print("dataset:", args.seed)
    print("mean time:", torch.mean(sequence_train_t), torch.mean(sequence_valid_t), torch.mean(sequence_test_t))
    print("train:", sequence_train_x.shape, "validation:", sequence_valid_x.shape, "test:", sequence_test_x.shape)
    print("NB-SEQ split by", sequence_train_t.shape[0], sequence_valid_t.shape[0], sequence_test_t.shape[0])
    print("training censoring rate:", 1 - (sequence_train_e.sum()/sequence_train_e.shape[0]).item())
    print("validation censoring rate:", 1 - (sequence_valid_e.sum()/sequence_valid_e.shape[0]).item())
    print("test censoring rate:", 1 - (sequence_test_e.sum()/sequence_test_e.shape[0]).item())
    
    torch.save(sequence_train_t, './data/sequence/' + str(args.seed) + '/sequence_train_t.pt')
    torch.save(sequence_valid_t, './data/sequence/' + str(args.seed) + '/sequence_valid_t.pt')
    torch.save(sequence_test_t, './data/sequence/' + str(args.seed) + '/sequence_test_t.pt')

    torch.save(sequence_train_e, './data/sequence/' + str(args.seed) + '/sequence_train_e.pt')
    torch.save(sequence_valid_e, './data/sequence/' + str(args.seed) + '/sequence_valid_e.pt')
    torch.save(sequence_test_e, './data/sequence/' + str(args.seed) + '/sequence_test_e.pt')

    torch.save(sequence_train_x, './data/sequence/' + str(args.seed) + '/sequence_train_x.pt')
    torch.save(sequence_valid_x, './data/sequence/' + str(args.seed) + '/sequence_valid_x.pt')
    torch.save(sequence_test_x, './data/sequence/' + str(args.seed) + '/sequence_test_x.pt')

elif args.dataset == 'mimic':
    import os
    import pandas as pd
    # after pre-processing steps provided in https://github.com/YerevaNN/mimic3-benchmarks
    # folder_path = "./MIMIC-III/mimic3-benchmarks-v1.0.0-alpha/YerevaNN-mimic3-benchmarks-847eadc/data/length-of-stay/train"
    folder_path = "./MIMIC-III/mimic3-benchmarks-v1.0.0-alpha/YerevaNN-mimic3-benchmarks-847eadc/data/length-of-stay/test"

    target_string = "_episode1_"
    file_list = os.listdir(folder_path)
    for filename in file_list:
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and target_string in filename:
            continue

        elif os.path.isfile(file_path):
            os.remove(file_path)

    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    data_frames = []
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        obs = df.mean()
        if df['Glascow coma scale eye opening'].mode().empty:
            pass
        else:
            obs = obs.append(pd.Series({'Glascow coma scale eye opening':df['Glascow coma scale eye opening'].mode()[0]}))
            if df['Glascow coma scale motor response'].mode().empty:
                pass
            else:
                obs = obs.append(pd.Series({'Glascow coma scale motor response':df['Glascow coma scale motor response'].mode()[0]}))
                if df['Glascow coma scale verbal response'].mode().empty:
                    pass
                else:
                    obs = obs.append(pd.Series({'Glascow coma scale verbal response':df['Glascow coma scale verbal response'].mode()[0]}))
        obs = pd.DataFrame(obs).T
        data_frames.append(obs)

    data = pd.concat(data_frames, ignore_index=True)
    check = pd.concat([pd.DataFrame(csv_files), data], ignore_index=True, axis=1)
    # check.to_csv("./data/mimic/train.csv", index=False)
    check.to_csv("./data/mimic/test.csv", index=False)
    # And then, manually set the data 

    mimic_train_t = pd.read_csv("./data/mimic/train_t.csv")
    mimic_train_x = pd.read_csv("./data/mimic/train_x.csv")
    mimic_train = pd.merge(mimic_train_t, mimic_train_x).dropna()

    mimic_test_t = pd.read_csv("./data/mimic/test_t.csv")
    mimic_test_x = pd.read_csv("./data/mimic/test_x.csv")
    mimic_test = pd.merge(mimic_test_t, mimic_test_x).dropna()

    mimic = pd.concat([mimic_train, mimic_test], axis=0).reset_index(drop=True)

    mimic = pd.get_dummies(mimic, columns=['13', '14', '15'])

    X = mimic.iloc[:, 3:].values
    t = mimic['futime'].values
    e = mimic['event'].values

    # stratified split (train 60%, valid 20%, test 20%)
    X_train, X_temp, t_train, t_temp, e_train, e_temp = train_test_split(
        X, t, e, test_size=0.4, stratify=e, random_state=args.seed
    )

    X_valid, X_test, t_valid, t_test, e_valid, e_test = train_test_split(
        X_temp, t_temp, e_temp, test_size=0.5, stratify=e_temp, random_state=args.seed
    )

    scaler1 = StandardScaler()
    mimic_train_x = scaler1.fit_transform(X_train)
    mimic_valid_x = scaler1.transform(X_valid)
    mimic_test_x = scaler1.transform(X_test)

    scaler2 = MinMaxScaler()
    mimic_train_t = scaler2.fit_transform(t_train.reshape(-1, 1))
    mimic_train_t = mimic_train_t.squeeze()
    mimic_valid_t = scaler2.transform(t_valid.reshape(-1, 1))
    mimic_valid_t = mimic_valid_t.squeeze()
    mimic_test_t = scaler2.transform(t_test.reshape(-1, 1))
    mimic_test_t = mimic_test_t.squeeze()

    mimic_train_t = torch.tensor(mimic_train_t)
    mimic_train_e = torch.tensor(e_train)
    mimic_train_x = torch.tensor(mimic_train_x)

    mimic_valid_t = torch.tensor(mimic_valid_t)
    mimic_valid_e = torch.tensor(e_valid)
    mimic_valid_x = torch.tensor(mimic_valid_x)

    mimic_test_t = torch.tensor(mimic_test_t)
    mimic_test_e = torch.tensor(e_test)
    mimic_test_x = torch.tensor(mimic_test_x)

    print("dataset:", args.seed)
    print("mean time:", torch.mean(mimic_train_t), torch.mean(mimic_valid_t), torch.mean(mimic_test_t))
    print("train:", mimic_train_x.shape, "validation:", mimic_valid_x.shape, "test:", mimic_test_x.shape)
    print("MIMIC-III split by", mimic_train_t.shape[0], mimic_valid_t.shape[0], mimic_test_t.shape[0])
    print("training censoring rate:", 1 - (mimic_train_e.sum()/mimic_train_e.shape[0]).item())
    print("validation censoring rate:", 1 - (mimic_valid_e.sum()/mimic_valid_e.shape[0]).item())
    print("test censoring rate:", 1 - (mimic_test_e.sum()/mimic_test_e.shape[0]).item())
    
    torch.save(mimic_train_t, './data/mimic/' + str(args.seed) + '/mimic_train_t.pt')
    torch.save(mimic_valid_t, './data/mimic/' + str(args.seed) + '/mimic_valid_t.pt')
    torch.save(mimic_test_t, './data/mimic/' + str(args.seed) + '/mimic_test_t.pt')

    torch.save(mimic_train_e, './data/mimic/' + str(args.seed) + '/mimic_train_e.pt')
    torch.save(mimic_valid_e, './data/mimic/' + str(args.seed) + '/mimic_valid_e.pt')
    torch.save(mimic_test_e, './data/mimic/' + str(args.seed) + '/mimic_test_e.pt')

    torch.save(mimic_train_x, './data/mimic/' + str(args.seed) + '/mimic_train_x.pt')
    torch.save(mimic_valid_x, './data/mimic/' + str(args.seed) + '/mimic_valid_x.pt')
    torch.save(mimic_test_x, './data/mimic/' + str(args.seed) + '/mimic_test_x.pt')

else:
    assert False