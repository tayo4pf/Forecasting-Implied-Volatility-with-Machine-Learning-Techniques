import math
import numpy as np
import pandas as pd
from distributions import get_realized_option_value, get_returns

class RNNDataLoader:
    def __init__(self, ticker: str, maturities: list[int], moneynesses: np.array, start_date: str, end_date: str, split_date: str):
        """
        Constructs a new OptionDataLoader object for options on the given ticker
        in the range of the maturities and moneynesses between start_date and end_date
        with the specified split_date to split the training and testing data
        @param ticker: The ticker of the underlying
        @param maturities: The maturities (DTE) of the options to load data for
        @param moneynesses: The moneynesses (S/K) of the options to load data for
        @param start_date: The start date of the data to be loaded
        @param end_date: The end date of the data to be loaded
        @param split_date: The split date of the data to be loaded
        """
        # Defining constant
        self.ticker = ticker
        self.maturities = maturities
        self.moneynesses = moneynesses
        self.start_date = start_date
        self.split_date = split_date
        self.end_date = end_date

        # Loading the training data (log returns)
        data = pd.DataFrame({"Log Returns": get_returns([self.ticker], start_date, end_date)})
        data = data.dropna(axis=0, how="all")
        data.index = pd.to_datetime(data.index)
        self.data = data

        self.train_y = np.empty((len(self.data["Log Returns"][:self.split_date]), len(self.maturities), len(self.moneynesses), 2)); self.train_y[:] = np.nan;
        self.train_S = np.empty((len(self.data["Log Returns"][:self.split_date]), len(self.maturities), len(self.moneynesses))); self.train_S[:] = np.nan;
        self.train_K = np.empty((len(self.data["Log Returns"][:self.split_date]), len(self.maturities), len(self.moneynesses))); self.train_K[:] = np.nan;
        self.train_t = np.empty((len(self.data["Log Returns"][:self.split_date]), len(self.maturities), len(self.moneynesses))); self.train_t[:] = np.nan;
        self.train_r = np.empty((len(self.data["Log Returns"][:self.split_date]), len(self.maturities), len(self.moneynesses))); self.train_r[:] = np.nan;
        self.train_q = np.empty((len(self.data["Log Returns"][:self.split_date]), len(self.maturities), len(self.moneynesses))); self.train_q[:] = np.nan;

        self.test_y = np.empty((len(self.data["Log Returns"][self.split_date:]), len(self.maturities), len(self.moneynesses), 2)); self.test_y[:] = np.nan;
        self.test_S = np.empty((len(self.data["Log Returns"][self.split_date:]), len(self.maturities), len(self.moneynesses))); self.test_S[:] = np.nan;
        self.test_K = np.empty((len(self.data["Log Returns"][self.split_date:]), len(self.maturities), len(self.moneynesses))); self.test_K[:] = np.nan;
        self.test_t = np.empty((len(self.data["Log Returns"][self.split_date:]), len(self.maturities), len(self.moneynesses))); self.test_t[:] = np.nan;
        self.test_r = np.empty((len(self.data["Log Returns"][self.split_date:]), len(self.maturities), len(self.moneynesses))); self.test_r[:] = np.nan;
        self.test_q = np.empty((len(self.data["Log Returns"][self.split_date:]), len(self.maturities), len(self.moneynesses))); self.test_q[:] = np.nan;


        # Constructing dataframe of the options data
        options = pd.DataFrame()
        for i, dte in enumerate(maturities):
            for j, moneyness in enumerate(moneynesses):
                ij_options = get_realized_option_value(ticker, start_date, end_date, moneyness, dte)
                ij_options.index = pd.to_datetime(ij_options.index)
                options = pd.concat((options, ij_options))

                self.train_y[:, i, j, 0] = ij_options.loc[self.data["Log Returns"][:self.split_date].index]["Realized Call Value"]
                self.train_y[:, i, j, 1] = ij_options.loc[self.data["Log Returns"][:self.split_date].index]["Realized Put Value"]
                self.train_S[:, i, j] = ij_options.loc[self.data["Log Returns"][:self.split_date].index]["Adj Close"]
                self.train_K[:, i, j] = ij_options.loc[self.data["Log Returns"][:self.split_date].index]["Adj Close"]/moneyness
                self.train_t[:, i, j] = ij_options.loc[self.data["Log Returns"][:self.split_date].index]["Tau"]
                self.train_r[:, i, j] = ij_options.loc[self.data["Log Returns"][:self.split_date].index]["Annualized Rate"] - 1
                self.train_q[:, i, j] = ij_options.loc[self.data["Log Returns"][:self.split_date].index]["Dividend Yield"]

                self.test_y[:, i, j, 0] = ij_options.loc[self.data["Log Returns"][self.split_date:].index]["Realized Call Value"]
                self.test_y[:, i, j, 1] = ij_options.loc[self.data["Log Returns"][self.split_date:].index]["Realized Put Value"]
                self.test_S[:, i, j] = ij_options.loc[self.data["Log Returns"][self.split_date:].index]["Adj Close"]
                self.test_K[:, i, j] = ij_options.loc[self.data["Log Returns"][self.split_date:].index]["Adj Close"]/moneyness
                self.test_t[:, i, j] = ij_options.loc[self.data["Log Returns"][self.split_date:].index]["Tau"]
                self.test_r[:, i, j] = ij_options.loc[self.data["Log Returns"][self.split_date:].index]["Annualized Rate"] - 1
                self.test_q[:, i, j] = ij_options.loc[self.data["Log Returns"][self.split_date:].index]["Dividend Yield"]

        self.options = options.dropna()

    def train_data(self, input_timesteps: int, output_timesteps: int) -> tuple[np.array, np.array, dict]:
        """
        Loads training data with specified amount of timesteps for the input and output data
        @param input_timesteps: number of timesteps of input data
        @param output_timesteps: number of timesteps of output data
        @return: tuple of training input, target (after black scholes), and dictionary of black scholes inputs for each output
        N.B. target final dimension is [target_call_value, target_put_value]
        These are present values because Black Scholes calculates **present value** of options 
        """
        # Defining the input data
        X_train = []

        # Iterating over timesteps before the split date, that have enough data to provide an input and target before the split date
        for i in range(len(self.data["Log Returns"][:self.split_date])-input_timesteps-output_timesteps):
            X_train.append(list(self.data["Log Returns"][:self.split_date][i:i+input_timesteps]))

        # Unsqueeze to shape (len, input_timesteps, features)
        X_train = np.array(X_train)
        X_train = np.expand_dims(X_train, axis=-1)

        # Defining the target data
        # black_scholes info is the input information required for black scholes to transform the model output IV
        # into a target price
        # Shape of each black_scholes info field is (len, output_timesteps, |maturities|, |moneynesses|)
        y_train = []
        black_scholes_info = {
            "S": [],
            "K": [],
            "t": [],
            "r": [],
            "q": []
        }

        # Iterating over timesteps before the split date, that have enough data to provide an input and target
        # Shape of target is (len, output_timesteps, |maturities|, |moneynesses|, 2)
        for i in range(len(self.data["Log Returns"][:self.split_date])-input_timesteps-output_timesteps):
            # y is the target data for the given input data [i, :, :]
            
            y = self.train_y[i+input_timesteps:i+input_timesteps+output_timesteps, :, :, :]
            y_S = self.train_S[i+input_timesteps:i+input_timesteps+output_timesteps, :, :]
            y_K = self.train_K[i+input_timesteps:i+input_timesteps+output_timesteps, :, :]
            y_t = self.train_t[i+input_timesteps:i+input_timesteps+output_timesteps, :, :]
            y_r = self.train_r[i+input_timesteps:i+input_timesteps+output_timesteps, :, :]
            y_q = self.train_q[i+input_timesteps:i+input_timesteps+output_timesteps, :, :]

            y_train.append(y)
            black_scholes_info["S"].append(y_S)
            black_scholes_info["K"].append(y_K)
            black_scholes_info["t"].append(y_t)
            black_scholes_info["r"].append(y_r)
            black_scholes_info["q"].append(y_q)

        # Converting lists to np.arrays
        y_train = np.array(y_train)
        for arg in black_scholes_info.keys():
            black_scholes_info[arg] = np.expand_dims(np.array(black_scholes_info[arg]), axis=-1)

        # Removing any samples with nan values
        mask = np.any(np.isnan(np.reshape(X_train, (X_train.shape[0], -1))), axis=1)
        mask |= np.any(np.isnan(np.reshape(y_train, (X_train.shape[0], -1))), axis=1)
        for arg in black_scholes_info.keys():
            mask |= np.any(np.isnan(np.reshape(black_scholes_info[arg], (X_train.shape[0], -1))), axis=1)
        
        X_train = X_train[~mask]
        y_train = y_train[~mask]
        for arg in black_scholes_info.keys():
            black_scholes_info[arg] = black_scholes_info[arg][~mask]

        return X_train, y_train, black_scholes_info
    
    def test_data(self, input_timesteps: int, output_timesteps: int, forecast: bool=False) -> tuple[np.array, np.array, dict]:
        """
        Loads testing data with specified amount of timesteps for the input and output data
        @param input_timesteps: number of timesteps of input data
        @param output_timesteps: number of timesteps of output data
        @return: tuple of training input, target (after black scholes), and dictionary of black scholes inputs for each output
        N.B. target final dimension is [target_call_value, target_put_value]
        These are present values because Black Scholes calculates **present value** of options 
        """
        # Defining the input data
        X_test = []

        # Iterating over timesteps before the split date, that have enough data to provide an input and target before the split date
        for i in range(len(self.data["Log Returns"][self.split_date:])-input_timesteps-output_timesteps):
            X_test.append(list(self.data["Log Returns"][self.split_date:][i:i+input_timesteps]))

        # Unsqueeze to shape (len, input_timesteps, features)
        X_test = np.array(X_test)
        X_test = np.expand_dims(X_test, axis=-1)

        # Defining the target data
        # black_scholes info is the input information required for black scholes to transform the model output IV
        # into a target price
        # Shape of each black_scholes info field is (len, output_timesteps, |maturities|, |moneynesses|)
        y_test = []
        black_scholes_info = {
            "S": [],
            "K": [],
            "t": [],
            "r": [],
            "q": []
        }

        if forecast:
            date = []

        # Iterating over timesteps before the split date, that have enough data to provide an input and target
        # Shape of target is (len, output_timesteps, |maturities|, |moneynesses|, 2)
        for i in range(len(self.data["Log Returns"][self.split_date:])-input_timesteps-output_timesteps):
            # y is the target data for the given input data [i, :, :]
            
            y = self.test_y[i+input_timesteps:i+input_timesteps+output_timesteps, :, :, :]
            y_S = self.test_S[i+input_timesteps:i+input_timesteps+output_timesteps, :, :]
            y_K = self.test_K[i+input_timesteps:i+input_timesteps+output_timesteps, :, :]
            y_t = self.test_t[i+input_timesteps:i+input_timesteps+output_timesteps, :, :]
            y_r = self.test_r[i+input_timesteps:i+input_timesteps+output_timesteps, :, :]
            y_q = self.test_q[i+input_timesteps:i+input_timesteps+output_timesteps, :, :]
            if forecast:
                dates = np.array(self.data["Log Returns"][self.split_date:][i+input_timesteps:i+input_timesteps+output_timesteps].index)
                date.append(dates)

            y_test.append(y)
            black_scholes_info["S"].append(y_S)
            black_scholes_info["K"].append(y_K)
            black_scholes_info["t"].append(y_t)
            black_scholes_info["r"].append(y_r)
            black_scholes_info["q"].append(y_q)
                

        # Converting lists to np.arrays
        y_test = np.array(y_test)
        for arg in black_scholes_info.keys():
            black_scholes_info[arg] = np.expand_dims(np.array(black_scholes_info[arg]), axis=-1)   

        # Removing any samples with nan values
        mask = np.any(np.isnan(np.reshape(X_test, (X_test.shape[0], -1))), axis=1)
        mask |= np.any(np.isnan(np.reshape(y_test, (X_test.shape[0], -1))), axis=1)
        for arg in black_scholes_info.keys():
            mask |= np.any(np.isnan(np.reshape(black_scholes_info[arg], (X_test.shape[0], -1))), axis=1)
        
        X_test = X_test[~mask]
        y_test = y_test[~mask]
        for arg in black_scholes_info.keys():
            black_scholes_info[arg] = black_scholes_info[arg][~mask]

        if forecast:
            date = np.array(date)
            date = date[~mask]
            black_scholes_info["Date"] = date

        return X_test, y_test, black_scholes_info
    
    def train_batch_gen(self, input_timesteps: int, output_timesteps: int, batch_size: int):
        """
        Returns a generator that produces batches of training data with the specified amount of input timesteps, and output timesteps
        @param input_timesteps: number of timesteps of input data
        @param output_timesteps: number of timesteps of output data
        @param batch_size: number of samples in each batch to be generated
        @return: tuple of training input, target (after black scholes), and dictionary of black scholes inputs for each output
        N.B. target final dimension is [target_call_value, target_put_value]
        These are present values because Black Scholes calculates **present value** of options at forecast date 
        """
        batch_count = 0
        discard_count = 0
        
        for i in range(len(self.data["Log Returns"][:self.split_date])-input_timesteps-output_timesteps):
            if batch_count == 0:
                X_train = []
                y_train = []
                black_scholes_info = {
                    "S": [], "K": [], "t": [], "r": [], "q": []
                }

            X = np.array(self.data["Log Returns"][:self.split_date][i:i+input_timesteps])

            y = self.train_y[i+input_timesteps:i+input_timesteps+output_timesteps, :, :, :]
            y_S = self.train_S[i+input_timesteps:i+input_timesteps+output_timesteps, :, :]
            y_K = self.train_K[i+input_timesteps:i+input_timesteps+output_timesteps, :, :]
            y_t = self.train_t[i+input_timesteps:i+input_timesteps+output_timesteps, :, :]
            y_r = self.train_r[i+input_timesteps:i+input_timesteps+output_timesteps, :, :]
            y_q = self.train_q[i+input_timesteps:i+input_timesteps+output_timesteps, :, :]

            # Check if sample data has any nan values, if so discard the sample
            if not any(np.any(np.isnan(arg.reshape(-1))) for arg in (X, y, y_S, y_K, y_t, y_r, y_q)):
                batch_count += 1
                X_train.append(X)
                y_train.append(y)
                black_scholes_info["S"].append(y_S); black_scholes_info["K"].append(y_K); black_scholes_info["t"].append(y_t); black_scholes_info["r"].append(y_r); black_scholes_info["q"].append(y_q);
            else:
                discard_count += 1
    
            if batch_count == batch_size or i == len(self.data["Log Returns"][:self.split_date])-input_timesteps-output_timesteps-1:
                batch_count = 0

                X_train = np.expand_dims(np.array(X_train), axis = -1)
                y_train = np.array(y_train)
                for arg in black_scholes_info.keys():
                    black_scholes_info[arg] = np.expand_dims(np.array(black_scholes_info[arg]), axis = -1)
                
                yield X_train, y_train, black_scholes_info
        print(f"{discard_count} samples discarded")

    def test_batch_gen(self, input_timesteps: int, output_timesteps: int, batch_size: int):
        """
        Returns a generator that produces batches of test data with the specified amount of input timesteps, and output timesteps
        @param input_timesteps: number of timesteps of input data
        @param output_timesteps: number of timesteps of output data
        @param batch_size: number of samples in each batch to be generated
        @return: tuple of training input, target (after black scholes), and dictionary of black scholes inputs for each output
        N.B. target final dimension is [target_call_value, target_put_value]
        These are present values because Black Scholes calculates **present value** of options at forecast date 
        """
        batch_count = 0

        for i in range(len(self.data["Log Returns"][self.split_date:])-input_timesteps-output_timesteps):
            if batch_count == 0:
                X_test = []
                y_test = []
                black_scholes_info = {
                    "S": [], "K": [], "t": [], "r": [], "q": []
                }

            X = np.array(self.data["Log Returns"][self.split_date:][i:i+input_timesteps])

            y = self.test_y[i+input_timesteps:i+input_timesteps+output_timesteps, :, :, :]
            y_S = self.test_S[i+input_timesteps:i+input_timesteps+output_timesteps, :, :]
            y_K = self.test_K[i+input_timesteps:i+input_timesteps+output_timesteps, :, :]
            y_t = self.test_t[i+input_timesteps:i+input_timesteps+output_timesteps, :, :]
            y_r = self.test_r[i+input_timesteps:i+input_timesteps+output_timesteps, :, :]
            y_q = self.test_q[i+input_timesteps:i+input_timesteps+output_timesteps, :, :]

            # Check if sample data has any nan values, if so discard the sample
            if not any(np.any(np.isnan(arg.reshape(-1))) for arg in (X, y, y_S, y_K, y_t, y_r, y_q)):
                batch_count += 1
                X_test.append(X)
                y_test.append(y)
                black_scholes_info["S"].append(y_S); black_scholes_info["K"].append(y_K); black_scholes_info["t"].append(y_t); black_scholes_info["r"].append(y_r); black_scholes_info["q"].append(y_q);
            else:
                print("Warning: Sample discarded")

            if batch_count == batch_size or i == len(self.data["Log Returns"][:self.split_date])-input_timesteps-output_timesteps-1:
                batch_count = 0

                X_test = np.expand_dims(np.array(X_test), axis = -1)
                y_test = np.array(y_test)
                for arg in black_scholes_info.keys():
                    black_scholes_info[arg] = np.expand_dims(np.array(black_scholes_info[arg]), axis = -1)
                
                yield X_test, y_test, black_scholes_info

    @staticmethod
    def batch_gen(X: np.array, y: np.array, bs_info: dict, batch_size: int, randomize=True):
        """
        Constructs a generator that produces batches of the given batch size
        @param X: Input data for the network
        @param y: The target data for the network
        @param bs_info: Dictionary containing the auxiliary Black-Scholes inputs for the network
        """
        # Check that X, y, and bs_info lengths match
        if len(X) != len(y):
            raise ValueError(f"Cannot generate training batches for training data with {len(X)} samples and {len(y)} targets")
        for key in bs_info.keys():
            if len(bs_info[key]) != len(X):
                raise ValueError(f"bs_info key: {key} with {len(bs_info[key])} samples does not match training data with {len(X)} samples")

        # Random batching order
        indices = np.arange(len(X))
        if randomize:
            np.random.shuffle(indices)

        # Yield batches
        for i in range(math.ceil(len(X)/batch_size)):
            x_ = X[indices[i*batch_size: min((i+1)*batch_size, len(X))]]
            y_ = y[indices[i*batch_size: min((i+1)*batch_size, len(X))]]
            bs_info_ = {
                key: bs_info[key][i*batch_size: min((i+1)*batch_size, len(X))] for key in bs_info.keys()
            }
            yield x_, y_, bs_info_