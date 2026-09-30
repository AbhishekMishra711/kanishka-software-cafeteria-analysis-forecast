import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def build_features(df, target_col='total_orders'):
    df_feat = df.copy()
    
    # Lag features
    for lag in [1, 2, 3, 7, 14, 21]:
        df_feat[f'lag_{lag}'] = df_feat[target_col].shift(lag)
    
    # Rolling features
    df_feat['roll_mean_7'] = df_feat[target_col].shift(1).rolling(7).mean()
    df_feat['roll_std_7'] = df_feat[target_col].shift(1).rolling(7).std()
    df_feat['roll_mean_14'] = df_feat[target_col].shift(1).rolling(14).mean()
    
    # Calendar features
    df_feat['day_of_week'] = df_feat['date'].dt.dayofweek
    df_feat['is_weekend'] = df_feat['day_of_week'].isin([5, 6]).astype(int)
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['day_of_month'] = df_feat['date'].dt.day
    
    return df_feat

def main():
    print("=" * 70)
    print("      CAFETERIA ORDER FORECASTING - 7-DAY PREDICTION CHALLENGE")
    print("=" * 70)

    # Load daily data
    daily_df = pd.read_csv('data/processed/daily_branch_orders.csv')
    daily_df['date'] = pd.to_datetime(daily_df['date'])

    # Find branch with highest volume
    branch_totals = daily_df.groupby(['branch_id', 'branch_name'])['total_orders'].sum().reset_index()
    top_branch = branch_totals.sort_values('total_orders', ascending=False).iloc[0]
    selected_branch_id = int(top_branch['branch_id'])
    selected_branch_name = top_branch['branch_name']
    
    print(f"\nSelected Branch for Forecasting: Branch {selected_branch_id} - '{selected_branch_name}'")
    print(f"Total Annual Orders in Branch:   {top_branch['total_orders']:,}")

    # Filter and reindex continuous daily series
    b_df = daily_df[daily_df['branch_id'] == selected_branch_id].copy()
    b_df = b_df.sort_values('date').set_index('date')
    
    full_idx = pd.date_range(start=b_df.index.min(), end=b_df.index.max(), freq='D')
    b_df = b_df.reindex(full_idx)
    b_df['total_orders'] = b_df['total_orders'].fillna(0)
    b_df['revenue'] = b_df['revenue'].fillna(0)
    b_df = b_df.reset_index().rename(columns={'index': 'date'})

    print(f"Time Series Span: {b_df['date'].min().strftime('%Y-%m-%d')} to {b_df['date'].max().strftime('%Y-%m-%d')} ({len(b_df)} continuous days)")

    # Define Holdout Validation Split (last 14 days for robust test evaluation)
    test_days = 14
    train_df = b_df.iloc[:-test_days].copy()
    test_df = b_df.iloc[-test_days:].copy()
    
    print(f"Training Set:  {train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')} ({len(train_df)} days)")
    print(f"Test Set:      {test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')} ({len(test_df)} days)")

    # -------------------------------------------------------------
    # MODEL 1: Seasonal Naive Baseline (Lag 7)
    # -------------------------------------------------------------
    test_pred_snaive = []
    for d in test_df['date']:
        lag_date = d - timedelta(days=7)
        lag_val = b_df.loc[b_df['date'] == lag_date, 'total_orders'].values
        test_pred_snaive.append(lag_val[0] if len(lag_val) > 0 else train_df['total_orders'].mean())
    test_pred_snaive = np.array(test_pred_snaive)

    # -------------------------------------------------------------
    # MODEL 2: Holt-Winters Exponential Smoothing (Additive Seasonality)
    # -------------------------------------------------------------
    hw_model = ExponentialSmoothing(
        train_df['total_orders'].values,
        seasonal_periods=7,
        trend='add',
        seasonal='add',
        damped_trend=True
    ).fit()
    test_pred_hw = hw_model.forecast(test_days)

    # -------------------------------------------------------------
    # MODEL 3: SARIMAX (Seasonal ARIMA with Weekly Periodicity)
    # -------------------------------------------------------------
    sarimax_model = SARIMAX(
        train_df['total_orders'].values,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    test_pred_sarimax = sarimax_model.forecast(steps=test_days)

    # -------------------------------------------------------------
    # MODEL 4: Random Forest with Lag & Calendar Features
    # -------------------------------------------------------------
    feat_df = build_features(b_df)
    feature_cols = [c for c in feat_df.columns if c.startswith('lag_') or c.startswith('roll_') or c in ['day_of_week', 'is_weekend', 'month', 'day_of_month']]
    
    # Drop rows with NaN from lags
    clean_feat = feat_df.dropna(subset=feature_cols).copy()
    
    train_feat = clean_feat[clean_feat['date'] <= train_df['date'].max()]
    test_feat = clean_feat[clean_feat['date'] > train_df['date'].max()]
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(train_feat[feature_cols], train_feat['total_orders'])
    test_pred_rf = rf.predict(test_feat[feature_cols])

    # -------------------------------------------------------------
    # MODEL 5: Gradient Boosting Regressor
    # -------------------------------------------------------------
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    gb.fit(train_feat[feature_cols], train_feat['total_orders'])
    test_pred_gb = gb.predict(test_feat[feature_cols])

    # -------------------------------------------------------------
    # EVALUATION ON HOLDOUT TEST SET
    # -------------------------------------------------------------
    models = {
        'Seasonal Naive (7-day)': test_pred_snaive,
        'Holt-Winters Exp Smoothing': test_pred_hw,
        'SARIMAX(1,1,1)(1,1,1,7)': test_pred_sarimax,
        'Random Forest Regressor': test_pred_rf,
        'Gradient Boosting Regressor': test_pred_gb
    }

    metrics = []
    y_true = test_df['total_orders'].values

    print("\n" + "=" * 70)
    print("           MODEL PERFORMANCE COMPARISON (TEST SET EVALUATION)")
    print("=" * 70)
    print(f"{'Model Name':<28} | {'MAE':<10} | {'RMSE':<10} | {'MAPE (%)':<10}")
    print("-" * 70)

    best_model_name = None
    best_mae = float('inf')

    for name, preds in models.items():
        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        # Mask zero actuals to avoid division by zero in MAPE
        mask = y_true > 0
        mape = np.mean(np.abs((y_true[mask] - preds[mask]) / y_true[mask])) * 100
        
        metrics.append({
            'Model': name,
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE (%)': round(mape, 2)
        })
        print(f"{name:<28} | {mae:<10.2f} | {rmse:<10.2f} | {mape:<10.2f}%")
        
        if mae < best_mae:
            best_mae = mae
            best_model_name = name

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('data/processed/model_evaluation_metrics.csv', index=False)
    print(f"\nBest Performing Model: {best_model_name} (Lowest MAE: {best_mae:.2f})")

    # -------------------------------------------------------------
    # 7-DAY OUT-OF-SAMPLE FORECAST (NEXT 7 DAYS)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("          GENERATING OUT-OF-SAMPLE 7-DAY FORWARD FORECAST")
    print("=" * 70)

    # Retrain best statistical model on 100% of data for the forward forecast
    full_sarimax = SARIMAX(
        b_df['total_orders'].values,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    forecast_res = full_sarimax.get_forecast(steps=7)
    point_forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.05)

    last_date = b_df['date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]

    # Compute expected revenue based on recent Average Order Value (AOV)
    recent_aov = b_df['revenue'].tail(14).sum() / b_df['total_orders'].tail(14).sum()

    forecast_records = []
    print(f"\nNext 7-Day Forecast for {selected_branch_name} (Branch ID: {selected_branch_id}):")
    print(f"{'Date':<12} | {'Day':<10} | {'Forecast Orders':<16} | {'95% CI Lower':<14} | {'95% CI Upper':<14} | {'Est. Revenue (INR)':<18}")
    print("-" * 92)

    for i in range(7):
        f_date = future_dates[i]
        dow = f_date.strftime('%A')
        orders_pred = max(0, round(point_forecast[i]))
        lower_b = max(0, round(conf_int[i, 0]))
        upper_b = max(0, round(conf_int[i, 1]))
        est_rev = round(orders_pred * recent_aov, 2)
        
        forecast_records.append({
            'date': f_date.strftime('%Y-%m-%d'),
            'day_of_week': dow,
            'forecast_orders': orders_pred,
            'lower_ci_95': lower_b,
            'upper_ci_95': upper_b,
            'estimated_revenue_inr': est_rev
        })
        print(f"{f_date.strftime('%Y-%m-%d'):<12} | {dow:<10} | {orders_pred:<16} | {lower_b:<14} | {upper_b:<14} | INR {est_rev:<14,.2f}")

    forecast_df = pd.DataFrame(forecast_records)
    forecast_df.to_csv('data/processed/forecast_results.csv', index=False)
    print(f"\nSaved 7-Day Forecast to: data/processed/forecast_results.csv")
    print(f"Total Expected Orders for Next 7 Days: {forecast_df['forecast_orders'].sum():,}")
    print(f"Total Projected Revenue for Next 7 Days: INR {forecast_df['estimated_revenue_inr'].sum():,.2f}")

    # Also save test predictions for visualization plotting
    test_eval_df = test_df[['date', 'total_orders']].copy().rename(columns={'total_orders': 'actual_orders'})
    for m_name, preds in models.items():
        test_eval_df[m_name] = np.round(preds, 1)
    test_eval_df.to_csv('data/processed/test_predictions.csv', index=False)
    print(f"Saved Test Evaluation Predictions to: data/processed/test_predictions.csv")

if __name__ == '__main__':
    main()
