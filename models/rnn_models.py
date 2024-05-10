from typing import Type
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from torch.nn import ELU, Linear, RNN, LSTM, GRU, MSELoss
from torch.optim import Adam
from models.dataloaders.rnn_data import RNNDataLoader

class IVS_RNN(torch.nn.Module):
    def __init__(self, features, hidden_size, num_layers, num_maturities, num_moneynesses, batch_first=True, dropout=0.0):
        super(IVS_RNN, self).__init__()
        self.rnn = RNN(input_size=features, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout)
        self.lin = Linear(hidden_size, num_maturities*num_moneynesses*2)
        self.elu = ELU()

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        rnn_out, h_out = self.rnn(input)
        lin_out = self.lin(rnn_out)
        # Activation to ensure volatility output is greater than 0
        out = self.elu(lin_out) + 1
        return out, h_out
    
    @staticmethod
    # Defining black-scholes function to take model output IVs and output Option Prices to compare to target prices
    def bs_fn(call, S, K, r, v, t, q):
        cdf = torch.distributions.Normal(0, 1).cdf

        d1 = (torch.log(S/K) + (r - q + (v**2)/2)*t) / (v * (t**0.5))
        d2 = d1 - (v * (t**0.5))

        if call:
            ov = S*torch.exp(-q*t)*cdf(d1) - K*torch.exp(-r*t)*cdf(d2)
        else:
            ov = K*torch.exp(-r*t)*cdf(-d2) - S*torch.exp(-q*t)*cdf(-d1)
        
        return ov

class IVS_LSTM(IVS_RNN):
    def __init__(self, features, hidden_size, num_layers, num_maturities, num_moneynesses, batch_first=True, dropout=0.0):
        super(IVS_LSTM, self).__init__(features, hidden_size, num_layers, num_maturities, num_moneynesses, batch_first, dropout)
        self.rnn = LSTM(input_size=features, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout)
    
class IVS_GRU(IVS_RNN):
    def __init__(self, features, hidden_size, num_layers, num_maturities, num_moneynesses, batch_first=True, dropout=0.0):
        super(IVS_GRU, self).__init__(features, hidden_size, num_layers, num_maturities, num_moneynesses, batch_first, dropout)
        self.rnn = GRU(input_size=features, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout)
    
class RNNForecastModel:
    """
    Forecaster for options price and implied volatility based on implied volatility modelling with an RNN
    """
    def __init__(self, ticker: str, start: str, end: str, split_date: str, maturities: list[int], moneynesses: np.array, rnn_hidden_size: int,
                rnn_num_layers: int, ivs_rnn_type: Type[IVS_RNN], rnn_dropout: float=0.0, loss_fn: torch.nn.modules.loss._Loss=MSELoss()):
        """
        Initializes an RNN Forecast Model Object
        @param ticker: Ticker of the underlying security to be forecasted
        @param start: The start date for the training data for implied volatility to be modelled with
        @param end: The end date for the model to forecast implied volatility and price for
        @param split_date: The date for which training data ends and the forecast begins
        @param maturities: The list of days till expiration on contract for which implied volatility and prices should be calculated
        @param moneynesses: The list of moneynesses (S/K) on contract for which implied volatility and prices should be calculated
        @param output_timesteps: The number of timesteps for which an implied volatility surface should be calculated in the model
        @param rnn_hidden_size: The number of features of the RNN hidden layer
        @param rnn_num_layers: The number of RNN layers in the model
        @param ivs_rnn_type: The type of RNN the model should use (IVS_RNN, IVS_LSTM, IVS_GRU)
        @param rnn_dropout: (optional) The dropout probability between RNN layers
        """

        # Storing member variables
        self.ticker = ticker
        self.start = start
        self.split_date = split_date
        self.end = end
        self.maturities = maturities
        self.moneynesses = moneynesses

        # Constructing data loader
        self.model_data_loader = RNNDataLoader(ticker, maturities, moneynesses, start, end, split_date)

        # Selecting device for model computations to be performed on
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA for model computations")
        
        # Constructing model
        self.rnn_model = ivs_rnn_type(1, rnn_hidden_size, rnn_num_layers, len(maturities), len(moneynesses), dropout=rnn_dropout).to(self.device)

        self.loss_fn = loss_fn
        self.opt = Adam(self.rnn_model.parameters())

    def train(self, epochs: int, batch_size: int, input_timesteps: int, dynamic_load: bool=False, quiet: bool=False) -> list[float]:
        """
        Train the underlying RNN model using the data before the split date
        @param epochs: The number of epochs to train the model over
        @param batch_size: The size of batches the model should be trained with
        @param input_timesteps: The amount of timesteps of input data provided to the model in training
        @param dynamic_load: (optional) Whether all the data from the training data should be loaded to memory at once, or if it should be dynamically loaded into tensors for each batch
        @param quiet: (optional) Whether the latest loss of each epoch should be printed
        @return: list of losses 
        """
        if dynamic_load:
            batches = lambda x: self.model_data_loader.train_batch_gen(input_timesteps, batch_size)
        else:
            # Loading training data
            X_train, y_train, bs_info_train = self.model_data_loader.train_data(input_timesteps)
            batches = lambda x: self.model_data_loader.batch_gen(X_train, y_train, bs_info_train, batch_size)
        
        losses = []
        self.rnn_model.train()

        for i in range(epochs):
            for x_, y_, bs_info_ in batches(0):
                x_ = torch.tensor(x_).to(torch.float32).to(self.device)
                y_ = torch.tensor(y_).to(torch.float32).to(self.device)
                for key in bs_info_.keys():
                    bs_info_[key] = torch.tensor(bs_info_[key]).to(torch.float32).to(self.device)
                
                self.opt.zero_grad()

                output, _ = self.rnn_model(x_)
                output = output.reshape(-1, y_.shape[1], y_.shape[2], y_.shape[3], y_.shape[4])

                # Calculating Black-Scholes option prices based on model iv surface and the contemporary data that was loaded
                call_prices = IVS_RNN.bs_fn(call=True, v=torch.unsqueeze(output[:, :, :, :, 0], -1), **bs_info_)
                put_prices = IVS_RNN.bs_fn(call=False, v=torch.unsqueeze(output[:, :, :, :, 1], -1), **bs_info_)
                price_output = torch.cat((call_prices, put_prices), -1)

                loss = self.loss_fn(price_output, y_)
                loss.backward()

                losses.append(torch.detach(loss).cpu())

                self.opt.step()
            if not quiet:
                print(f"Epoch {i+1}/{epochs} - latest loss: {loss}")

        return losses
    
    def test(self, batch_size: int, input_timesteps: int, dynamic_load: bool=False) -> list[float]:
        """
        Tests the underlying RNN model using the data after the split date
        @param batch_size: The size of the batches the model will test
        @param input_timesteps: The amount of timesteps of input data provided to the model in training
        @param dynamic_load: (optional) Whether all the data 
        """
        if dynamic_load:
            batches = lambda x: self.model_data_loader.test_batch_gen(input_timesteps, self.output_timesteps, batch_size)
        else:
            X_test, y_test, bs_info_test = self.model_data_loader.test_data(input_timesteps, self.output_timesteps)
            batches = lambda x: self.model_data_loader.batch_gen(X_test, y_test, bs_info_test, batch_size)
        
        losses = []
        self.rnn_model.eval()

        for x_, y_, bs_info_ in batches(0):
            x_ = torch.tensor(x_).to(torch.float32).to(self.device)
            y_ = torch.tensor(y_).to(torch.float32).to(self.device)
            for key in bs_info_.keys():
                bs_info_[key] = torch.tensor(bs_info_[key]).to(torch.float32).to(self.device)
            
            self.opt.zero_grad()

            output, _ = self.rnn_model(x_)
            output = output.reshape(-1, y_.shape[1], y_.shape[2], y_.shape[3], y_.shape[4])

            # Calculating Black-Scholes pricing based on iv surface
            call_prices = IVS_RNN.bs_fn(call=True, v=torch.unsqueeze(output[:, :, :, :, 0], -1), **bs_info_)
            put_prices = IVS_RNN.bs_fn(call=False, v=torch.unsqueeze(output[:, :, :, :, 1], -1), **bs_info_)
            price_output = torch.cat((call_prices, put_prices), -1)

            loss = self.loss_fn(price_output, y_)

            losses.append(torch.detach(loss).cpu())

        return losses
    
    def options_pricing(self, input_timesteps: int):
        options = pd.DataFrame()
        X, y, bs_info = self.model_data_loader.test_data(input_timesteps, forecast=True)

        X = torch.tensor(X).to(torch.float32).to(self.device)
        y = torch.tensor(y).to(torch.float32).to(self.device)
        date = bs_info.pop("Date")
        for key in bs_info.keys():
            bs_info[key] = torch.tensor(bs_info[key]).to(torch.float32).to(self.device)

        output, _ = self.rnn_model(X)
        output = output.reshape(-1, y.shape[1], y.shape[2], y.shape[3], y.shape[4])

        call_prices = IVS_RNN.bs_fn(call=True, v=torch.unsqueeze(output[:, :, :, :, 0], -1), **bs_info)
        put_prices = IVS_RNN.bs_fn(call=False, v=torch.unsqueeze(output[:, :, :, :, 1], -1), **bs_info)

        for i, dte in enumerate(self.maturities):
            for j, moneyness in enumerate(self.moneynesses):
                iter_options = pd.DataFrame(
                    {
                        "Call Price": call_prices[:, -1, i, j, 0].cpu().detach().numpy(),
                        "Put Price": put_prices[:, -1, i, j, 0].cpu().detach().numpy(),
                        "Call IV": output[:, -1, i, j, 0].cpu().detach().numpy(),
                        "Put IV": output[:, -1, i, j, 1].cpu().detach().numpy(),
                        "Dividend Yield": bs_info["q"][:, -1, i, j, 0].cpu().detach().numpy(),
                        "Annualized Rate": bs_info["r"][:, -1, i, j, 0].cpu().detach().numpy()+1,
                        "Underlying Price": bs_info["S"][:, -1, i, j, 0].cpu().detach().numpy(),
                        "Strike Price": bs_info["K"][:, -1, i, j, 0].cpu().detach().numpy(),
                        "Tau": bs_info["t"][:, -1, i, j, 0].cpu().detach().numpy(),
                        "DTE": dte,
                        "Realized Call Value": y[:, -1, i, j, 0].cpu().detach().numpy(),
                        "Realized Put Value": y[:, -1, i, j, 1].cpu().detach().numpy(),
                        "Moneyness": moneyness,
                        "Date": date[:, -1]
                    }
                )
                options = pd.concat((options, iter_options), ignore_index=True)
        
        return options

    def plot_ivs(self, output: torch.Tensor, forecast_timestep: int, output_timestep: int):
        """
        Plots the call and put IV surface based on the non-reshaped output of the RNN model
        @param output: the output from the RNN model
        @param forecast_timestep: the timestep of the forecast
        @param output_timestep: the timestep from the forecast the iv surface is plotted
        """
        call_fig = go.Figure(data=[
            go.Surface(
                x=self.moneynesses,
                y=self.maturities,
                z=output[forecast_timestep, output_timestep, :, :, 0].cpu().detach().numpy()
                )
        ])
        call_fig.update_layout(title=f"Forecasted Call IV Surface")
        call_fig.update_scenes(
            xaxis_title_text="Moneyness",
            yaxis_title_text="Maturity",
            zaxis_title_text="Implied Volatility"
        )
        call_fig.show()

        put_fig = go.Figure(data=[
            go.Surface(
                x=self.moneynesses,
                y=self.maturities,
                z=output[forecast_timestep, output_timestep, :, :, 1].cpu().detach().numpy()
                )
        ])
        put_fig.update_layout(title=f"Forecasted Put IV Surface")
        put_fig.update_scenes(
            xaxis_title_text="Moneyness",
            yaxis_title_text="Maturity",
            zaxis_title_text="Implied Volatility"
        )
        put_fig.show()
