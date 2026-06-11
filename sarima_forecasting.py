import pandas as pd
import matplotlib.pyplot as plt

from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA_PATH = "data/clean_invbanker.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully")
print("Total Records:", len(df))

df["Day Posted"] = pd.to_datetime(
    df["Day Posted"],
    errors="coerce"
)

df = df.dropna(subset=["Day Posted"])

monthly_jobs = (
    df.groupby(
        pd.Grouper(
            key="Day Posted",
            freq="M"
        )
    )
    .size()
)

monthly_jobs = monthly_jobs.asfreq("M", fill_value=0)

print("\nMonthly Time Series Created")
print(monthly_jobs.head())

print("\nSearching Best SARIMA Parameters...")

auto_model = auto_arima(
    monthly_jobs,
    seasonal=True,
    m=12,
    trace=True,
    suppress_warnings=True
)

print("\nBest Order:", auto_model.order)
print("Best Seasonal Order:", auto_model.seasonal_order)

model = SARIMAX(
    monthly_jobs,
    order=auto_model.order,
    seasonal_order=auto_model.seasonal_order
)

results = model.fit()

print("\nModel Trained Successfully")

forecast_steps = 36

forecast = results.get_forecast(
    steps=forecast_steps
)

forecast_values = forecast.predicted_mean
confidence = forecast.conf_int()

future_dates = pd.date_range(
    start=monthly_jobs.index[-1] +
    pd.offsets.MonthEnd(1),
    periods=36,
    freq="M"
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Jobs": forecast_values.values
})

forecast_df.to_csv(
    "forecast_2025_2027.csv",
    index=False
)

print("\nForecast Saved Successfully")

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_jobs.index,
    monthly_jobs.values,
    label="Historical Jobs"
)

plt.plot(
    future_dates,
    forecast_values,
    color="red",
    label="Forecast"
)

plt.fill_between(
    future_dates,
    confidence.iloc[:, 0],
    confidence.iloc[:, 1],
    alpha=0.3
)

plt.title("Jobify Employment Trend Forecast (2025-2027)")
plt.xlabel("Date")
plt.ylabel("Number of Jobs")

plt.legend()
plt.tight_layout()

plt.savefig(
    "forecast_dashboard.png",
    dpi=300
)

plt.show()

print("\nForecast Dashboard Saved")
