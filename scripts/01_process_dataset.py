import csv
import time
import os
from collections import defaultdict
import pandas as pd
from datetime import datetime

def main():
    print("=" * 60)
    print("  CAFETERIA ORDER DATA - FULL STREAMING ETL PIPELINE")
    print("=" * 60)
    
    # Load reference mappings
    branches_df = pd.read_csv('data/processed/branches.csv')
    branch_map = dict(zip(branches_df['id'].astype(int), branches_df['name']))
    print(f"Loaded {len(branch_map)} branches.")

    date_cache = {}

    # Storage structures for aggregated analysis
    # Key: (date_str, branch_id)
    # Value: [total_orders, completed, cancelled, subtotal, tax, discount, grand_total, pos, mobile, qr, users_set]
    daily_stats = defaultdict(lambda: [0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, set()])

    # Key: (branch_id, dow_name, dow_num, hour)
    # Value: [order_count, revenue]
    hourly_stats = defaultdict(lambda: [0, 0.0])

    # Key: (branch_id, mode_of_transaction)
    # Value: [count, revenue]
    payment_stats = defaultdict(lambda: [0, 0.0])

    # Key: (branch_id, order_through)
    # Value: [count, revenue]
    channel_stats = defaultdict(lambda: [0, 0.0])

    # Key: user_id -> [order_count, total_spend, first_date, last_date]
    user_stats = defaultdict(lambda: [0, 0.0, None, None])

    # Key: dish_name -> [dish_id, counter_id, total_qty, total_rev, order_count]
    dish_stats = defaultdict(lambda: [0, 0, 0, 0.0, 0])

    # Key: counter_id -> [total_qty, total_rev, order_count]
    counter_stats = defaultdict(lambda: [0, 0.0, 0])

    # Stream through the main SQL dump
    sql_file = 'Cafeteria Order Data/Cafeteria Order Data.sql'
    
    t0 = time.time()
    order_rows = 0
    detail_rows = 0
    in_orders = False
    in_details = False

    print(f"\nStarting stream processing of: {sql_file} (~11 GB)...")
    
    with open(sql_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            # Check transitions
            if line.startswith('INSERT INTO `orders` '):
                if not in_orders:
                    print(f"[{time.strftime('%H:%M:%S')}] Started parsing 'orders' table at line {line_num}...")
                in_orders = True
                in_details = False
                continue
            elif line.startswith('INSERT INTO `order_details` '):
                if not in_details:
                    print(f"[{time.strftime('%H:%M:%S')}] Started parsing 'order_details' table at line {line_num}...")
                in_orders = False
                in_details = True
                continue
            elif line.startswith('CREATE TABLE `order_extra_addons`'):
                print(f"[{time.strftime('%H:%M:%S')}] Reached end of order_details at line {line_num}. Finishing stream.")
                break

            # Parse order rows
            if in_orders:
                sline = line.strip().rstrip(';,')
                if sline.startswith('(') and sline.endswith(')'):
                    try:
                        row = next(csv.reader([sline[1:-1]], quotechar="'", skipinitialspace=True))
                        order_rows += 1
                        
                        # Col 2: user_id, Col 3: order_status, Col 6: order_date
                        # Col 8: branch_id, Col 12: order_through, Col 13: sub_total
                        # Col 14: tax_amount, Col 18: discount_amount, Col 19: mode_of_transaction
                        # Col 28: grand_total
                        user_id = int(row[2]) if row[2] and row[2] != 'NULL' else 0
                        order_status = int(row[3]) if row[3] and row[3] != 'NULL' else 0
                        order_date_str = row[6]
                        
                        if not order_date_str or order_date_str == 'NULL':
                            continue
                        
                        date_str = order_date_str[:10]
                        time_part = order_date_str[11:19]
                        hour = int(order_date_str[11:13]) if len(order_date_str) >= 13 else 0
                        
                        branch_id = int(row[8]) if row[8] and row[8] != 'NULL' else 0
                        channel = (row[12] or 'unknown').strip().lower()
                        
                        subtotal = float(row[13]) if row[13] and row[13] != 'NULL' else 0.0
                        tax = float(row[14]) if row[14] and row[14] != 'NULL' else 0.0
                        discount = float(row[18]) if row[18] and row[18] != 'NULL' else 0.0
                        mode = (row[19] or 'unknown').strip()
                        grand_total = float(row[28]) if row[28] and row[28] != 'NULL' else 0.0

                        # Fast date cache to avoid strptime overhead
                        if date_str not in date_cache:
                            try:
                                dt_obj = datetime.strptime(date_str, '%Y-%m-%d')
                                date_cache[date_str] = (dt_obj.weekday(), dt_obj.strftime('%A'))
                            except Exception:
                                date_cache[date_str] = (0, 'Monday')
                        dow_num, dow_name = date_cache[date_str]

                        # Update daily stats
                        d = daily_stats[(date_str, branch_id)]
                        d[0] += 1
                        if order_status == 3:
                            d[1] += 1
                        elif order_status == 4:
                            d[2] += 1
                        d[3] += subtotal
                        d[4] += tax
                        d[5] += discount
                        d[6] += grand_total
                        
                        if 'pos' in channel:
                            d[7] += 1
                        elif 'mobile' in channel:
                            d[8] += 1
                        elif 'qr' in channel:
                            d[9] += 1
                        
                        if user_id > 0:
                            d[10].add(user_id)

                        # Update hourly stats
                        h = hourly_stats[(branch_id, dow_name, dow_num, hour)]
                        h[0] += 1
                        h[1] += grand_total

                        # Payment stats
                        p = payment_stats[(branch_id, mode)]
                        p[0] += 1
                        p[1] += grand_total

                        # Channel stats
                        c = channel_stats[(branch_id, channel)]
                        c[0] += 1
                        c[1] += grand_total

                        # User stats
                        if user_id > 0:
                            u = user_stats[user_id]
                            u[0] += 1
                            u[1] += grand_total
                            if u[2] is None or date_str < u[2]:
                                u[2] = date_str
                            if u[3] is None or date_str > u[3]:
                                u[3] = date_str

                        if order_rows % 500000 == 0:
                            elapsed = time.time() - t0
                            print(f"Processed {order_rows:,} orders ({order_rows/elapsed:,.0f} rows/s)...")

                    except Exception as e:
                        continue

            # Parse order detail rows
            elif in_details:
                sline = line.strip().rstrip(';,')
                if sline.startswith('(') and sline.endswith(')'):
                    try:
                        row = next(csv.reader([sline[1:-1]], quotechar="'", skipinitialspace=True))
                        detail_rows += 1
                        
                        # Col 2: dish_id, Col 9: dish_name, Col 10: order_quantity
                        # Col 15: dish_price, Col 17: dish_cal_price, Col 18: counter_id
                        dish_id = int(row[2]) if row[2] and row[2] != 'NULL' else 0
                        dish_name = row[9].strip() if row[9] and row[9] != 'NULL' else 'Unknown Dish'
                        qty = int(row[10]) if row[10] and row[10] != 'NULL' else 1
                        dish_cal_price = float(row[17]) if row[17] and row[17] != 'NULL' else 0.0
                        counter_id = int(row[18]) if row[18] and row[18] != 'NULL' else 0

                        # Dish stats
                        ds = dish_stats[dish_name]
                        ds[0] = dish_id
                        ds[1] = counter_id
                        ds[2] += qty
                        ds[3] += dish_cal_price
                        ds[4] += 1

                        # Counter stats
                        cs = counter_stats[counter_id]
                        cs[0] += qty
                        cs[1] += dish_cal_price
                        cs[2] += 1

                        if detail_rows % 500000 == 0:
                            elapsed = time.time() - t0
                            print(f"Processed {detail_rows:,} order details ({detail_rows/elapsed:,.0f} rows/s)...")

                    except Exception as e:
                        continue

    total_time = time.time() - t0
    print(f"\nCompleted streaming extraction in {total_time:.1f}s!")
    print(f"Total Orders Processed: {order_rows:,}")
    print(f"Total Order Details Processed: {detail_rows:,}")

    # ==========================================
    # SAVE PROCESSED AGGREGATED DATASETS
    # ==========================================
    print("\nSaving processed CSV datasets...")

    # 1. Daily Branch Orders
    daily_records = []
    for (d_str, b_id), vals in daily_stats.items():
        b_name = branch_map.get(b_id, f"Branch {b_id}")
        daily_records.append({
            'date': d_str,
            'branch_id': b_id,
            'branch_name': b_name,
            'total_orders': vals[0],
            'completed_orders': vals[1],
            'cancelled_orders': vals[2],
            'subtotal': round(vals[3], 2),
            'tax': round(vals[4], 2),
            'discount': round(vals[5], 2),
            'revenue': round(vals[6], 2),
            'pos_orders': vals[7],
            'mobile_orders': vals[8],
            'qr_orders': vals[9],
            'unique_customers': len(vals[10]),
            'avg_order_value': round(vals[6] / vals[0], 2) if vals[0] > 0 else 0.0
        })
    df_daily = pd.DataFrame(daily_records).sort_values(['date', 'branch_id'])
    df_daily.to_csv('data/processed/daily_branch_orders.csv', index=False)
    print(f"Saved: data/processed/daily_branch_orders.csv ({len(df_daily):,} rows)")

    # 2. Hourly Sales Pattern
    hourly_records = []
    for (b_id, dow_name, dow_num, hour), vals in hourly_stats.items():
        b_name = branch_map.get(b_id, f"Branch {b_id}")
        hourly_records.append({
            'branch_id': b_id,
            'branch_name': b_name,
            'day_of_week': dow_name,
            'day_of_week_num': dow_num,
            'hour': hour,
            'order_count': vals[0],
            'revenue': round(vals[1], 2)
        })
    df_hourly = pd.DataFrame(hourly_records).sort_values(['branch_id', 'day_of_week_num', 'hour'])
    df_hourly.to_csv('data/processed/hourly_sales_pattern.csv', index=False)
    print(f"Saved: data/processed/hourly_sales_pattern.csv ({len(df_hourly):,} rows)")

    # 3. Payment Methods
    payment_records = []
    for (b_id, mode), vals in payment_stats.items():
        b_name = branch_map.get(b_id, f"Branch {b_id}")
        payment_records.append({
            'branch_id': b_id,
            'branch_name': b_name,
            'mode_of_transaction': mode,
            'order_count': vals[0],
            'revenue': round(vals[1], 2)
        })
    df_pay = pd.DataFrame(payment_records).sort_values(['branch_id', 'order_count'], ascending=[True, False])
    df_pay.to_csv('data/processed/payment_methods.csv', index=False)
    print(f"Saved: data/processed/payment_methods.csv ({len(df_pay):,} rows)")

    # 4. Channels
    channel_records = []
    for (b_id, ch), vals in channel_stats.items():
        b_name = branch_map.get(b_id, f"Branch {b_id}")
        channel_records.append({
            'branch_id': b_id,
            'branch_name': b_name,
            'order_through': ch,
            'order_count': vals[0],
            'revenue': round(vals[1], 2)
        })
    df_chan = pd.DataFrame(channel_records).sort_values(['branch_id', 'order_count'], ascending=[True, False])
    df_chan.to_csv('data/processed/order_channels.csv', index=False)
    print(f"Saved: data/processed/order_channels.csv ({len(df_chan):,} rows)")

    # 5. Top Dishes
    counters_df = pd.read_csv('data/processed/counters.csv')
    counter_map = dict(zip(counters_df['id'].astype(int), counters_df['counter_name']))
    
    dish_records = []
    for dish_name, vals in dish_stats.items():
        cntr_name = counter_map.get(vals[1], f"Counter {vals[1]}")
        dish_records.append({
            'dish_name': dish_name,
            'dish_id': vals[0],
            'counter_id': vals[1],
            'counter_name': cntr_name,
            'total_quantity_sold': vals[2],
            'total_revenue': round(vals[3], 2),
            'total_orders': vals[4],
            'avg_dish_price': round(vals[3] / vals[2], 2) if vals[2] > 0 else 0.0
        })
    df_dishes = pd.DataFrame(dish_records).sort_values('total_quantity_sold', ascending=False)
    df_dishes.to_csv('data/processed/top_dishes.csv', index=False)
    print(f"Saved: data/processed/top_dishes.csv ({len(df_dishes):,} dishes)")

    # 6. Counter Sales
    counter_records = []
    for cntr_id, vals in counter_stats.items():
        cntr_name = counter_map.get(cntr_id, f"Counter {cntr_id}")
        counter_records.append({
            'counter_id': cntr_id,
            'counter_name': cntr_name,
            'total_items_sold': vals[0],
            'total_revenue': round(vals[1], 2),
            'total_order_lines': vals[2]
        })
    df_cntr = pd.DataFrame(counter_records).sort_values('total_revenue', ascending=False)
    df_cntr.to_csv('data/processed/counter_sales.csv', index=False)
    print(f"Saved: data/processed/counter_sales.csv ({len(df_cntr):,} counters)")

    # 7. Customer Frequency Summary (Histogram / Bins)
    user_records = []
    for u_id, vals in user_stats.items():
        user_records.append({
            'user_id': u_id,
            'order_count': vals[0],
            'total_spend': round(vals[1], 2),
            'first_order_date': vals[2],
            'last_order_date': vals[3]
        })
    df_users = pd.DataFrame(user_records)
    # Save frequency summary
    df_users.to_csv('data/processed/customer_frequency.csv', index=False)
    print(f"Saved: data/processed/customer_frequency.csv ({len(df_users):,} customers)")

    print("\nETL Pipeline Execution Complete! All summary datasets ready for Analysis and Forecasting.")

if __name__ == '__main__':
    main()
