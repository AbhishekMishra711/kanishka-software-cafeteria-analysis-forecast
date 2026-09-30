import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def generate_all_charts():
    print("=" * 70)
    print("      GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("=" * 70)

    # Load datasets
    daily_df = pd.read_csv('data/processed/daily_branch_orders.csv')
    hourly_df = pd.read_csv('data/processed/hourly_sales_pattern.csv')
    pay_df = pd.read_csv('data/processed/payment_methods.csv')
    chan_df = pd.read_csv('data/processed/order_channels.csv')
    dishes_df = pd.read_csv('data/processed/top_dishes.csv')
    cust_df = pd.read_csv('data/processed/customer_frequency.csv')
    forecast_df = pd.read_csv('data/processed/forecast_results.csv')
    test_eval_df = pd.read_csv('data/processed/test_predictions.csv')

    daily_df['date'] = pd.to_datetime(daily_df['date'])

    # -------------------------------------------------------------
    # CHART 1: Branch Order Volume & Revenue Comparison
    # -------------------------------------------------------------
    print("Generating Chart 1: Branch Volume & Revenue...")
    branch_agg = daily_df.groupby('branch_name').agg(
        orders=('total_orders', 'sum'),
        revenue=('revenue', 'sum')
    ).reset_index().sort_values('orders', ascending=True)

    fig, ax1 = plt.subplots(figsize=(12, 7))
    y_pos = np.arange(len(branch_agg))
    
    # Horizontal bars for orders
    colors = ['#1f77b4' if o < branch_agg['orders'].max() * 0.5 else '#0d47a1' for o in branch_agg['orders']]
    bars = ax1.barh(y_pos, branch_agg['orders'] / 1000, color=colors, alpha=0.85, height=0.6, label='Orders (Thousands)')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(branch_agg['branch_name'], fontsize=11, fontweight='medium')
    ax1.set_xlabel('Total Orders (in Thousands)', fontsize=12, fontweight='bold', color='#1a237e')
    ax1.set_title('Annual Order Volume & Revenue by Branch (FY 2024-25)', fontsize=14, fontweight='bold', pad=15)
    
    # Add data labels
    for bar, rev in zip(bars, branch_agg['revenue']):
        width = bar.get_width()
        ax1.text(width + 5, bar.get_y() + bar.get_height()/2, f'{width:.1f}k (₹{rev/1e6:.1f}M)', 
                 va='center', ha='left', fontsize=9.5, color='#263238', fontweight='semibold')
    
    ax1.set_xlim(0, max(branch_agg['orders'] / 1000) * 1.25)
    plt.tight_layout()
    fig.savefig('visualizations/01_branch_order_volume_comparison.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # CHART 2: Monthly Sales Trend by Top Branches
    # -------------------------------------------------------------
    print("Generating Chart 2: Monthly Sales Trends...")
    daily_df['month_year'] = daily_df['date'].dt.to_period('M')
    top5_branches = daily_df.groupby('branch_name')['revenue'].sum().nlargest(5).index
    
    monthly_trend = daily_df[daily_df['branch_name'].isin(top5_branches)].groupby(['month_year', 'branch_name'])['revenue'].sum().unstack()
    monthly_trend.index = monthly_trend.index.astype(str)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, col in enumerate(monthly_trend.columns):
        ax.plot(monthly_trend.index, monthly_trend[col] / 1e6, marker='o', linewidth=2.5, label=col, color=palette[i % len(palette)])

    ax.set_title('Monthly Revenue Trend for Top 5 Branches (FY 2024-25)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gross Revenue (INR Millions)', fontsize=12, fontweight='bold')
    ax.legend(title='Branch', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig('visualizations/02_monthly_sales_trend_by_branch.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # CHART 3: Hourly Operating Peaks (Meal Time Rushes)
    # -------------------------------------------------------------
    print("Generating Chart 3: Hourly Operating Peaks...")
    hour_summary = hourly_df.groupby('hour')['order_count'].sum().reset_index()
    hour_summary['order_k'] = hour_summary['order_count'] / 1000

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(hour_summary['hour'], hour_summary['order_k'], color='#0288d1', alpha=0.75, width=0.7, edgecolor='#01579b')
    ax.plot(hour_summary['hour'], hour_summary['order_k'], color='#d81b60', marker='s', linewidth=2, label='Order Volume Curve')

    # Highlight Meal Rush Windows
    ax.axvspan(8.5, 10.5, color='#ffe082', alpha=0.35, label='Breakfast Rush (9:00 - 10:30 AM)')
    ax.axvspan(12.0, 14.5, color='#ffab91', alpha=0.35, label='Lunch Peak (12:00 - 2:30 PM)')
    ax.axvspan(16.5, 18.5, color='#c5cae9', alpha=0.35, label='Evening Snacks / Tea (4:30 - 6:30 PM)')

    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right', fontsize=9.5)
    ax.set_title('Hourly Cafeteria Traffic Pattern: Distinct Breakfast, Lunch & Tea Peaks', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Annual Orders (Thousands)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=10.5)
    plt.tight_layout()
    fig.savefig('visualizations/03_hourly_peak_order_distribution.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # CHART 4: Top 15 Best-Selling Menu Items
    # -------------------------------------------------------------
    print("Generating Chart 4: Top 15 Selling Dishes...")
    top15_dishes = dishes_df.head(15).sort_values('total_quantity_sold', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7.5))
    y_pos = np.arange(len(top15_dishes))
    bars = ax.barh(y_pos, top15_dishes['total_quantity_sold'] / 1000, color='#388e3c', alpha=0.85, height=0.65)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top15_dishes['dish_name'], fontsize=11, fontweight='medium')
    ax.set_xlabel('Total Units Sold (Thousands)', fontsize=12, fontweight='bold', color='#1b5e20')
    ax.set_title('Top 15 Most Ordered Cafeteria Menu Items', fontsize=14, fontweight='bold', pad=15)

    for bar, rev, prc in zip(bars, top15_dishes['total_revenue'], top15_dishes['avg_dish_price']):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}k units | ₹{rev/1e6:.2f}M (@ ₹{prc:.0f})', 
                va='center', ha='left', fontsize=9, color='#1b5e20', fontweight='semibold')

    ax.set_xlim(0, max(top15_dishes['total_quantity_sold'] / 1000) * 1.35)
    plt.tight_layout()
    fig.savefig('visualizations/04_top_15_selling_dishes.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # CHART 5: Channel and Payment Split (Donut Charts)
    # -------------------------------------------------------------
    print("Generating Chart 5: Channels & Payments...")
    ch_agg = chan_df.groupby('order_through')['order_count'].sum().sort_values(ascending=False)
    # Clean channel names
    ch_labels = ['POS Counter' if 'pos' in str(k).lower() else 'Mobile App' if 'mobile' in str(k).lower() else 'QR / Web' if 'qr' in str(k).lower() else str(k).title() for k in ch_agg.index]
    
    pay_agg = pay_df.groupby('mode_of_transaction')['order_count'].sum().sort_values(ascending=False).head(5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    
    # Donut 1: Channels
    colors_ch = ['#1976d2', '#ff9800', '#4caf50', '#9c27b0']
    wedges1, texts1, autotexts1 = ax1.pie(ch_agg.values, labels=ch_labels, autopct='%1.1f%%',
                                           startangle=140, colors=colors_ch[:len(ch_agg)],
                                           pctdistance=0.75, wedgeprops=dict(width=0.45, edgecolor='w'))
    for autotext in autotexts1:
        autotext.set_color('white')
        autotext.set_weight('bold')
    ax1.set_title('Ordering Channel Share\n(POS vs Mobile App vs QR)', fontsize=13, fontweight='bold', pad=15)

    # Donut 2: Payment Modes
    colors_pay = ['#00897b', '#f4511e', '#3949ab', '#7cb342', '#8e24aa']
    wedges2, texts2, autotexts2 = ax2.pie(pay_agg.values, labels=pay_agg.index, autopct='%1.1f%%',
                                           startangle=140, colors=colors_pay[:len(pay_agg)],
                                           pctdistance=0.75, wedgeprops=dict(width=0.45, edgecolor='w'))
    for autotext in autotexts2:
        autotext.set_color('white')
        autotext.set_weight('bold')
    ax2.set_title('Payment Method Adoption\n(Top Transaction Modes)', fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    fig.savefig('visualizations/05_order_channel_and_payment_breakdown.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # CHART 6: Daily Time-Series for Selected Branch
    # -------------------------------------------------------------
    print("Generating Chart 6: Selected Branch Daily Series...")
    # Find branch with highest volume
    top_b_id = daily_df.groupby('branch_id')['total_orders'].sum().idxmax()
    top_b_name = daily_df[daily_df['branch_id'] == top_b_id]['branch_name'].iloc[0]
    
    b_df = daily_df[daily_df['branch_id'] == top_b_id].copy().sort_values('date')
    b_df['rolling_7d'] = b_df['total_orders'].rolling(7).mean()

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(b_df['date'], b_df['total_orders'], color='#90caf9', alpha=0.7, linewidth=1, label='Daily Actual Orders')
    ax.plot(b_df['date'], b_df['rolling_7d'], color='#0d47a1', linewidth=2.5, label='7-Day Moving Average')

    ax.set_title(f'Daily Order Trajectory: {top_b_name} (Branch ID: {top_b_id})', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Daily Orders', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig('visualizations/06_daily_orders_time_series_selected_branch.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # CHART 7: Forecast Comparison & Next 7 Days
    # -------------------------------------------------------------
    print("Generating Chart 7: Forecast & 7-Day Forward Projection...")
    test_eval_df['date'] = pd.to_datetime(test_eval_df['date'])
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])

    fig, ax = plt.subplots(figsize=(13, 6.5))
    
    # Plot test period actuals vs predictions
    ax.plot(test_eval_df['date'], test_eval_df['actual_orders'], color='black', marker='o', linewidth=2.5, label='Actual Orders (Holdout Test)')
    ax.plot(test_eval_df['date'], test_eval_df['SARIMAX(1,1,1)(1,1,1,7)'], color='#e53935', linestyle='--', marker='x', linewidth=2, label='SARIMAX Model (Test Backtest)')
    if 'Random Forest Regressor' in test_eval_df.columns:
        ax.plot(test_eval_df['date'], test_eval_df['Random Forest Regressor'], color='#3949ab', linestyle=':', marker='^', linewidth=1.8, label='Random Forest (Test Backtest)')

    # Plot forward 7-day forecast
    ax.plot(forecast_df['date'], forecast_df['forecast_orders'], color='#2e7d32', marker='D', linewidth=3, label='Next 7-Day Forecast (Forward Projection)')
    ax.fill_between(forecast_df['date'], forecast_df['lower_ci_95'], forecast_df['upper_ci_95'], color='#81c784', alpha=0.35, label='95% Prediction Interval')

    # Add vertical divider line
    split_date = test_eval_df['date'].max()
    ax.axvline(split_date, color='#546e7a', linestyle='-', linewidth=1.5, alpha=0.8)
    ax.text(split_date - timedelta(days=2), ax.get_ylim()[1] * 0.92, 'Holdout Test Period ◄', ha='right', fontsize=10.5, fontweight='bold', color='#37474f')
    ax.text(split_date + timedelta(days=0.5), ax.get_ylim()[1] * 0.92, '► 7-Day Out-of-Sample Forecast', ha='left', fontsize=10.5, fontweight='bold', color='#2e7d32')

    # Add labels on forecasted points
    for _, row in forecast_df.iterrows():
        ax.annotate(f"{int(row['forecast_orders']):,}\n({row['day_of_week'][:3]})", 
                    (row['date'], row['forecast_orders']),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8.5, fontweight='bold', color='#1b5e20')

    ax.set_title(f'7-Day Forward Forecast with 95% Confidence Bounds: {top_b_name}', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Daily Orders', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig('visualizations/07_forecast_comparison_next_7_days.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # CHART 8: Customer Ordering Frequency Cohorts
    # -------------------------------------------------------------
    print("Generating Chart 8: Customer Retention & Cohorts...")
    cohorts = pd.cut(cust_df['order_count'], bins=[0, 1, 5, 20, 50, 100, 10000], 
                     labels=['1 Order\n(One-off)', '2-5 Orders\n(Occasional)', '6-20 Orders\n(Regular)', '21-50 Orders\n(Loyal)', '51-100 Orders\n(Super Loyal)', '100+ Orders\n(Power Users)'])
    cohort_counts = cohorts.value_counts(sort=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(cohort_counts.index.astype(str), cohort_counts.values, color='#5c6bc0', alpha=0.85, width=0.6, edgecolor='#283593')
    
    total_c = len(cust_df)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 500, f'{h:,}\n({h/total_c*100:.1f}%)', 
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1a237e')

    ax.set_title('Customer Order Frequency Distribution & Retention Cohorts', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Customer Engagement Cohort', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Unique Customers', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(cohort_counts.values) * 1.18)
    plt.tight_layout()
    fig.savefig('visualizations/08_customer_retention_and_frequency.png', dpi=300)
    plt.close()

    print("\nAll 8 publication-quality charts successfully created in 'visualizations/' folder!")

if __name__ == '__main__':
    generate_all_charts()
