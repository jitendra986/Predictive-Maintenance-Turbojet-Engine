import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch

class DataPreprocessor:
    def __init__(self, train_file_path):
        self.train_file_path = train_file_path
        self.train_df = None
        self.validation_df = None
        self.test_df = None
        self.seq_array = None
        self.seq_array_validation = None
        self.seq_array_test = None
        self.dummy_label_array = None
        self.dummy_label_array_validation = None
        self.dummy_label_array_test = None
        self.cols_normalize_train = None
        self.cols_normalize_validation = None
        self.cols_normalize_test = None
        self.sequence_cols = None

    def preprocess(self):
        self._load_and_prepare_data()
        self._calculate_rul()
        self._split_data()
        self._normalize_data()
        self._generate_sequences()
        self._generate_labels()
        return (self.seq_array, self.dummy_label_array, 
                self.seq_array_validation, self.dummy_label_array_validation,
               self.seq_array_test, self.dummy_label_array_test, self.sequence_cols)

    
    def test_data_pdmPolicy(self):
        self._load_and_prepare_data()
        self._calculate_rul()
        self._split_data()

        return (self.validation_df,self.test_df,self.train_df)
    
    def normalize_data_pdmPolicy(self):
        self._load_and_prepare_data()
        self._calculate_rul()
        self._split_data()
        self._normalize_data()
        return (self.cols_normalize_train,self.cols_normalize_validation,self.cols_normalize_test)
    
    def _load_and_prepare_data(self):
        self.train_df = pd.read_csv(self.train_file_path, sep=" ", header=None)
        self.train_df.drop(self.train_df.columns[[26, 27]], axis=1, inplace=True)
        self.train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                                 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                                 's15', 's16', 's17', 's18', 's19', 's20', 's21']
        self.train_df = self.train_df.sort_values(['id','cycle'])

    def _calculate_rul(self):
        rul = pd.DataFrame(self.train_df.groupby('id')['cycle'].max()).reset_index()
        rul.columns = ['id', 'max']
        self.train_df = self.train_df.merge(rul, on=['id'], how='left')
        self.train_df['RUL'] = self.train_df['max'] - self.train_df['cycle']
        self.train_df.drop('max', axis=1, inplace=True)
        
        w1 = 30
        w0 = 10
        self.train_df['label1'] = np.where(self.train_df['RUL'] <= w1, 1, 0)
        self.train_df['label2'] = self.train_df['label1']
        self.train_df.loc[self.train_df['RUL'] <= w0, 'label2'] = 2

    def _split_data(self):
        list_ID1 = np.arange(81,91,1)
        list_ID2 = np.arange(91,101,1)
        list_ID3 = np.arange(81,101,1)
        
        self.validation_df = self.train_df.loc[self.train_df['id'].isin(list_ID1)]
        self.test_df = self.train_df.loc[self.train_df['id'].isin(list_ID2)]
        self.train_df = self.train_df[~self.train_df.id.isin(list_ID3)]
        
        
    def _normalize_data(self):
        self.train_df['cycle_norm'] = self.train_df['cycle']
        self.validation_df['cycle_norm'] = self.validation_df['cycle']
        self.test_df['cycle_norm'] = self.test_df['cycle']

        self.cols_normalize_train = self.train_df.columns.difference(['id','cycle','RUL','label1','label2'])#removing irrelevant 5 columns
        self.cols_normalize_validation = self.validation_df.columns.difference(['id','cycle','RUL','label1','label2'])
        self.cols_normalize_test = self.test_df.columns.difference(['id','cycle','RUL','label1','label2'])
        
        min_max_scaler = preprocessing.MinMaxScaler()
        norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(self.train_df[self.cols_normalize_train]),
                                     columns=self.cols_normalize_train,
                                     index=self.train_df.index)
        norm_validation_df = pd.DataFrame(min_max_scaler.fit_transform(self.validation_df[self.cols_normalize_validation]),
                                          columns=self.cols_normalize_validation,
                                          index=self.validation_df.index)
        
        norm_test_df = pd.DataFrame(min_max_scaler.fit_transform(self.test_df[self.cols_normalize_test]),
                                          columns=self.cols_normalize_test,
                                          index=self.test_df.index)
        
        # MinMax normalization (from 0 to 1)
        join_df_train = self.train_df[self.train_df.columns.difference(self.cols_normalize_train)].join(norm_train_df)
        join_df_validation = self.validation_df[self.validation_df.columns.difference(self.cols_normalize_validation)].join(norm_validation_df)
        join_df_test = self.test_df[self.test_df.columns.difference(self.cols_normalize_test)].join(norm_test_df)

        self.train_df = join_df_train.reindex(columns = self.train_df.columns)
        self.validation_df = join_df_validation.reindex(columns = self.validation_df.columns)
        self.test_df = join_df_test.reindex(columns = self.test_df.columns)

    def _generate_sequences(self):
        sequence_length = 50
        sensor_cols = ['s' + str(i) for i in range(1,22)]
        self.sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm'] 
        self.sequence_cols.extend(sensor_cols)

        def gen_sequence(id_df, seq_length, seq_cols):
            data_matrix = id_df[seq_cols].values
            num_elements = data_matrix.shape[0]
            for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
                yield data_matrix[start:stop, :]

        seq_gen = (list(gen_sequence(self.train_df[self.train_df['id']==id], sequence_length, self.sequence_cols))
                   for id in self.train_df['id'].unique())
        seq_gen_validation = (list(gen_sequence(self.validation_df[self.validation_df['id']==id], sequence_length, self.sequence_cols))
                              for id in self.validation_df['id'].unique())
        
        seq_gen_test = (list(gen_sequence(self.test_df[self.test_df['id']==id], sequence_length, self.sequence_cols))
                              for id in self.test_df['id'].unique())

        self.seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
        self.seq_array_validation = np.concatenate(list(seq_gen_validation)).astype(np.float32)
        self.seq_array_test = np.concatenate(list(seq_gen_test)).astype(np.float32)

    def _generate_labels(self):
        sequence_length = 50

        def gen_labels(id_df, seq_length, label):
            data_matrix = id_df[label].values
            num_elements = data_matrix.shape[0]
            return data_matrix[seq_length:num_elements, :]

        label_gen = [gen_labels(self.train_df[self.train_df['id']==id], sequence_length, ['label2'])
                     for id in self.train_df['id'].unique()]
        label_array = np.concatenate(label_gen).astype(np.float32).flatten()

        label_gen_validation = [gen_labels(self.validation_df[self.validation_df['id']==id], sequence_length, ['label2'])
                                for id in self.validation_df['id'].unique()]
        label_array_validation = np.concatenate(label_gen_validation).astype(np.float32).flatten()
        
        label_gen_test = [gen_labels(self.test_df[self.test_df['id']==id], sequence_length, ['label2'])
                                for id in self.test_df['id'].unique()]
        label_array_test = np.concatenate(label_gen_test).astype(np.float32).flatten()

        # Assuming your array is a numpy array named `data`
        data = np.array([label_array]).reshape(-1)  # Ensure it is a 1D array of shape (12138,)
        data_validation = np.array([label_array_validation]).reshape(-1)  # Ensure it is a 1D array of shape (12138,)
        data_test = np.array([label_array_test]).reshape(-1)  # Ensure it is a 1D array of shape (12138,)

        # Convert the numpy array to a PyTorch tensor
        data_tensor = torch.tensor(data, dtype=torch.long)
        data_tensor_validation = torch.tensor(data_validation, dtype=torch.long)
        data_tensor_test = torch.tensor(data_test, dtype=torch.long)
        
        # Perform one-hot encoding
        dummy_label_array = torch.nn.functional.one_hot(data_tensor, num_classes=3)
        dummy_label_array_validation = torch.nn.functional.one_hot(data_tensor_validation, num_classes=3)
        dummy_label_array_test = torch.nn.functional.one_hot(data_tensor_test, num_classes=3)

        # Convert the PyTorch tensor to numpy array
        self.dummy_label_array = dummy_label_array.numpy().astype(np.int64)
        self.dummy_label_array_validation = dummy_label_array_validation.numpy().astype(np.int64)
        self.dummy_label_array_test = dummy_label_array_test.numpy().astype(np.int64)

        
        """
        # Perform one-hot encoding
        dummy_label_array = torch.nn.functional.one_hot(data_tensor, num_classes=3)
        dummy_label_array_validation = torch.nn.functional.one_hot(data_tensor_validation, num_classes=3)
        dummy_label_array_test = torch.nn.functional.one_hot(data_tensor_test, num_classes=3)

        # Convert the a PyTorch tensor to numpy array. 
        self.dummy_label_array = torch.nn.functional.numpy().astype(np.int64)
        self.dummy_label_array_validation = torch.nn.functional.numpy().astype(np.int64)
        self.dummy_label_array_test = torch.nn.functional.numpy().astype(np.int64)
        """
       