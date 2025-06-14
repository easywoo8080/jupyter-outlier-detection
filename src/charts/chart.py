import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_ses_results(
    series: pd.Series, fitted: pd.Series, forecast: pd.Series, title: str
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(series, label="Original")
    plt.plot(fitted, label="Fitted")
    plt.plot(forecast, label="Forecast", linestyle="--")
    plt.legend()
    plt.title(title)
    plt.grid(True)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
