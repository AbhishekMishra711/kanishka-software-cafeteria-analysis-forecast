import pandas as pd
import numpy as np

def run_eda():
    print("=" * 70)
    print("       CAFETERIA ORDER DATA - EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    # Load datasets
    daily_df = pd.read_csv('data/processed/daily_branch_orders.csv')
    hourly_df = pd.read_csv('data/processed/hourly_sales_pattern.csv')
    pay_df = pd.read_csv('data/processed/payment_methods.csv')
    chan_df = pd.read_csv('data/processed/order_channels.csv')
    dishes_df = pd.read_csv('data/processed/top_dishes.csv')
    cntr_df = pd.read_csv('data/processed/counter_sales.csv')
    cust_df = pd.read_csv('data/processed/customer_frequency.csv')
    branches_df = pd.read_csv('data/processed/branches.csv')
    users_df = pd.read_csv('data/processed/users.csv')

    daily_df['date'] = pd.to_datetime(daily_df['date'])

    print("\n--- 1. OVERALL PORTFOLIO PERFORMANCE ---")
    total_orders = daily_df['total_orders'].sum()
    total_rev = daily_df['revenue'].sum()
    total_subtotal = daily_df['subtotal'].sum()
    total_discount = daily_df['discount'].sum()
    total_tax = daily_df['tax'].sum()
    overall_aov = total_rev / total_orders if total_orders > 0 else 0
    date_min = daily_df['date'].min().strftime('%Y-%m-%d')
    date_max = daily_df['date'].max().strftime('%Y-%m-%d')
    num_days = (daily_df['date'].max() - daily_df['date'].min()).days + 1

    print(f"Date Range:              {date_min} to {date_max} ({num_days} days)")
    print(f"Total Completed Orders:  {total_orders:,}")
    print(f"Total Gross Revenue:     INR {total_rev:,.2f}")
    print(f"Total Subtotal:          INR {total_subtotal:,.2f}")
    print(f"Total Discounts Given:   INR {total_discount:,.2f} ({total_discount/total_subtotal*100:.2f}%)")
    print(f"Total Tax Collected:     INR {total_tax:,.2f}")
    print(f"Overall Average Order:   INR {overall_aov:.2f}")

    print("\n--- 2. BRANCH-WISE PERFORMANCE RANKING ---")
    branch_summary = daily_df.groupby(['branch_id', 'branch_name']).agg(
        total_orders=('total_orders', 'sum'),
        total_revenue=('revenue', 'sum'),
        avg_daily_orders=('total_orders', 'mean'),
        avg_daily_rev=('revenue', 'mean'),
        avg_aov=('avg_order_value', 'mean')
    ).reset_index().sort_values('total_orders', ascending=False)
    branch_summary['rev_share_%'] = (branch_summary['total_revenue'] / total_rev) * 100
    branch_summary['order_share_%'] = (branch_summary['total_orders'] / total_orders) * 100

    print(branch_summary[['branch_id', 'branch_name', 'total_orders', 'order_share_%', 'total_revenue', 'rev_share_%', 'avg_daily_orders']].to_string(index=False))

    print("\n--- 3. ORDERING CHANNELS DISTRIBUTION ---")
    ch_agg = chan_df.groupby('order_through').agg(
        orders=('order_count', 'sum'),
        revenue=('revenue', 'sum')
    ).reset_index().sort_values('orders', ascending=False)
    ch_agg['order_pct'] = (ch_agg['orders'] / ch_agg['orders'].sum()) * 100
    ch_agg['rev_pct'] = (ch_agg['revenue'] / ch_agg['revenue'].sum()) * 100
    print(ch_agg.to_string(index=False))

    print("\n--- 4. PAYMENT MODES BREAKDOWN ---")
    pay_agg = pay_df.groupby('mode_of_transaction').agg(
        orders=('order_count', 'sum'),
        revenue=('revenue', 'sum')
    ).reset_index().sort_values('orders', ascending=False)
    pay_agg['order_pct'] = (pay_agg['orders'] / pay_agg['orders'].sum()) * 100
    pay_agg['rev_pct'] = (pay_agg['revenue'] / pay_agg['revenue'].sum()) * 100
    print(pay_agg.head(10).to_string(index=False))

    print("\n--- 5. HOURLY PEAK OPERATING TRAFFIC ---")
    hour_agg = hourly_df.groupby('hour').agg(
        total_orders=('order_count', 'sum'),
        total_revenue=('revenue', 'sum')
    ).reset_index().sort_values('hour')
    hour_agg['order_share_%'] = (hour_agg['total_orders'] / hour_agg['total_orders'].sum()) * 100
    print(hour_agg.to_string(index=False))

    # Top peak hours
    top_hours = hour_agg.sort_values('total_orders', ascending=False).head(5)
    print("\nTop 5 Peak Hours:")
    for _, r in top_hours.iterrows():
        print(f"  Hour {int(r['hour']):02d}:00 - {int(r['hour'])+1:02d}:00: {int(r['total_orders']):,} orders ({r['order_share_%']:.1f}% of day)")

    print("\n--- 6. TOP 15 BEST-SELLING DISHES (VOLUME & REVENUE) ---")
    print(dishes_df[['dish_name', 'counter_name', 'total_quantity_sold', 'total_revenue', 'avg_dish_price']].head(15).to_string(index=False))

    print("\n--- 7. TOP COUNTERS / VENDORS ---")
    print(cntr_df.head(10).to_string(index=False))

    print("\n--- 8. CUSTOMER BEHAVIOR & COHORTS ---")
    total_customers = len(cust_df)
    orders_p_cust = cust_df['order_count']
    print(f"Total Registered/Transacting Customers: {total_customers:,}")
    print(f"Median Orders per Customer:            {orders_p_cust.median():.0f}")
    print(f"Average Orders per Customer:           {orders_p_cust.mean():.1f}")
    print(f"Top 10% Customers Order Threshold:     {orders_p_cust.quantile(0.90):.0f} orders")
    print(f"Max Orders by a Single Customer:       {orders_p_cust.max():,}")

    # Frequency cohorts
    cohorts = pd.cut(orders_p_cust, bins=[0, 1, 5, 20, 50, 100, 10000], labels=['1 Order', '2-5 Orders', '6-20 Orders', '21-50 Orders', '51-100 Orders', '100+ Orders'])
    cohort_summary = cohorts.value_counts(sort=False).reset_index()
    cohort_summary.columns = ['Cohort', 'User_Count']
    cohort_summary['Percentage'] = (cohort_summary['User_Count'] / total_customers) * 100
    print(cohort_summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("EDA Execution Complete! Key statistics successfully computed.")
    print("=" * 70)

if __name__ == '__main__':
    run_eda()
