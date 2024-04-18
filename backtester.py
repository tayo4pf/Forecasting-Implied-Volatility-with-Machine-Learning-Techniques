import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime as dt
import dateutil as du
from models.econ_models import ARCHForecastModel, ARIMAForecastModel, ARIMAARCHForecastModel, HistoricVolModel
import time

def cartesian_product_simple_transpose(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
    return arr.reshape(la, -1).T

class Backtest:
    def __init__(self, tickers, start_dates, end_dates, split_years_ago, maturities, moneynesses):
        self.tickers = tickers
        self.start_dates = start_dates
        self.end_dates = end_dates
        self.split_years_ago = split_years_ago
        self.maturities = maturities
        self.moneynesses = moneynesses

        self.perms = cartesian_product_simple_transpose(
            (tickers,
            start_dates,
            end_dates,
            split_years_ago,
            maturities,
            moneynesses)
        )
        self.performances = {}

    def backtest(model, ticker, start, end, split, maturation, moneyness, params, repetitions):
        performances = {
            "Repetitions": repetitions,
            "Call Residuals": [],
            "Put Residuals": [],
            "Training Time": [],
            "Pricing Time": []
        }
        for _ in range(repetitions):
            if model is ARCHForecastModel:
                pack = (ticker, start, end, split, maturation, *params)
            elif model is ARIMAForecastModel:
                pack = (ticker, start, end, split, maturation, *params)
            elif model is ARIMAARCHForecastModel:
                pack = (ticker, start, end, split, maturation, *params)
            elif model is HistoricVolModel:
                pack = (ticker, start, end, split, maturation, *params)
            else:
                raise ValueError("Unknown model framework passed")
            train_start = time.process_time()
            forecaster = model(*pack)
            train_time = time.process_time() - train_start
            pricing_start = time.process_time()
            forecaster.forecaster(maturation)
            simulator = forecaster.simulation(maturation)
            pricing = forecaster.options_pricing(simulator, moneyness)
            pricing_time = time.process_time() - pricing_start
            call_residuals = pricing["Call Price ATM"] - pricing["Realized Call Value ATM"]
            put_residuals = pricing["Put Price ATM"] - pricing["Realized Put Value ATM"]
            performances["Call Residuals"].append(call_residuals)
            performances["Put Residuals"].append(put_residuals)  
            performances["Training Time"].append(train_time)
            performances["Pricing Time"].append(pricing_time)
        return performances

    def present_performance(model_explanation:str, performances:dict):
        full_call_residuals = np.concatenate(performances["Call Residuals"])
        full_put_residuals = np.concatenate(performances["Put Residuals"])
        mean_training_time = sum(performances["Training Time"])/performances["Repetitions"]
        mean_pricing_time = sum(performances["Pricing Time"])/performances["Repetitions"]
        mean_call_residual = np.mean(full_call_residuals)
        mean_put_residual = np.mean(full_put_residuals)
        std_call_residuals = np.std(full_call_residuals)
        std_put_residuals = np.std(full_put_residuals)
        pd.DataFrame({"Call Residuals":full_call_residuals}).plot(kind="kde", title="Call Residuals")
        pd.DataFrame({"Put Residuals":full_put_residuals}).plot(kind="kde", title="Put Residuals")

        print(f"""For {model_explanation},\nmean training time was {mean_training_time} seconds,
              and mean pricing time was {mean_pricing_time} seconds,
              mean call residual was {mean_call_residual} with variance {std_call_residuals**2} and,
              mean put residual was {mean_put_residual} with variance {std_put_residuals**2}
              """)

    def compare_performance(model_explanations:list[str], performances:list[dict]):
        #Plotting Residuals
        fig, (call_ax, put_ax) = plt.subplots(2, 1)
        for model, perf in zip(model_explanations, performances):
            full_call_residuals = np.concatenate(perf["Call Residuals"])
            full_put_residuals = np.concatenate(perf["Put Residuals"])
            pd.DataFrame({f"{model}": full_call_residuals}).plot(kind="kde", ax=call_ax)
            pd.DataFrame({f"{model}":full_put_residuals}).plot(kind="kde", ax=put_ax)
        call_ax.set_title("Call Residuals")
        put_ax.set_title("Put Residuals")
        fig.tight_layout()

        #Plotting Mean of Residuals
        fig, (mean_call_ax, mean_put_ax) = plt.subplots(2, 1)
        mean_call_ax.bar(model_explanations, [np.mean(perf["Call Residuals"]) for perf in performances])
        mean_put_ax.bar(model_explanations, [np.mean(perf["Put Residuals"]) for perf in performances])
        mean_call_ax.set_title("Mean Call Residuals")
        mean_put_ax.set_title("Mean Put Residuals")
        fig.tight_layout()

        #Plotting Variance of Residuals
        fig, (var_call_ax, var_put_ax) = plt.subplots(2, 1)
        var_call_ax.bar(model_explanations, [np.var(perf["Call Residuals"]) for perf in performances])
        var_put_ax.bar(model_explanations, [np.var(perf["Put Residuals"]) for perf in performances])
        var_call_ax.set_title("Variance of Call Residuals")
        var_put_ax.set_title("Variance of Put Residuals")
        fig.tight_layout()

        #Plotting Training and Pricing Times
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.bar(model_explanations, [sum(perf["Training Time"])/(perf["Repetitions"]) for perf in performances])
        ax2.bar(model_explanations, [(sum(perf["Pricing Time"])/perf["Repetitions"]) for perf in performances])
        ax1.set_title("Training Time")
        ax1.set_ylabel("Seconds")
        ax2.set_title("Pricing Time")
        ax2.set_ylabel("Seconds")
        fig.tight_layout()
