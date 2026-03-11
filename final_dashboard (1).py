
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error

# 1. PAGE SETUP & THEMING
st.set_page_config(page_title="Milma Strategic Intelligence", layout="wide", page_icon="")

st.markdown("""
    <style>
    /* Typography: Academic Standard */
    html, body, [class*="css"], .stApp, h1, h2, h3, p, span, div, button, select {
        font-family: 'Times New Roman', Times, serif !important;
    }

    /* Background: Deep Executive Gradient */
    .stApp {
        background: radial-gradient(circle at top right, #1F4959, #011425);
        color: #FFFFFF;
    }

    /* Sidebar: Professional Slate */
    [data-testid="stSidebar"] {
        background-color: rgba(1, 20, 37, 0.95) !important;
        border-right: 1px solid #5C7C89;
    }

    /* Glassmorphism Cards: Clean Containers for Data */
    div[data-testid="stMetric"], .stPlotlyChart, .stDataFrame {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(92, 124, 137, 0.3) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        backdrop-filter: blur(15px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    /* Metric Styling: Emerald Green for Growth */
    [data-testid="stMetricValue"] {
        color: #2ECC71 !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# GLOBAL UTILITY FUNCTIONS
# ---------------------------------------------------------

def get_final_report(data):
    report_results = []
    for prod in data['product'].unique():
        p_data = data[data['product'] == prod].copy()
        p_agg = p_data.groupby('date').agg({'qty': 'sum', 'price': 'mean', 'month': 'first'}).reset_index()
        p_agg = p_agg[(p_agg['qty'] > 0) & (p_agg['price'] > 0)]
        if p_agg['price'].nunique() <= 1:
            category = "Price Stable"
            elasticity = 0
            p_val = 1.0
        else:
            try:
                X = p_agg[['price', 'month']].copy()
                X['price'] = np.log(X['price'])
                X = sm.add_constant(X)
                y = np.log(p_agg['qty'])
                model = sm.OLS(y, X).fit()
                elasticity = model.params['price']
                p_val = model.pvalues['price']
                month_p = model.pvalues['month']
                if month_p < 0.05:
                    if elasticity < -1.0:
                        category = "Seasonal - High Sensitivity"
                    else:
                        category = "Seasonal - Low Sensitivity"
                elif elasticity > 0.1:
                    category = "Veblen/Giffen Behavior"
                elif -1.0 <= elasticity <= 0.1:
                    category = "Low Sensitivity (Anchor)"
                elif elasticity < -1.0:
                    category = "High Sensitivity (Leaker)"
                else:
                    category = "Neutral"
            except:
                category = "Insufficient Data"
                elasticity = 0
                p_val = 1.0
        report_results.append({
            'Product Name': prod,
            'Category': category,
            'Elasticity': round(elasticity, 3),
            'P-Value': round(p_val, 4)
        })
    return pd.DataFrame(report_results)


# 2. DATA ENGINE
@st.cache_data
def load_and_clean_data(uploaded_files):
    all_data = []
    m_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for file in uploaded_files:
        df = pd.read_csv(file)
        fname = file.name.lower()
        df.columns = [str(c).strip().lower() for c in df.columns]
        p_col = next((c for c in df.columns if any(x in c for x in ['product', 'item', 'particulars'])), df.columns[0])
        q_col = next((c for c in df.columns if any(x in c for x in ['qty', 'quantity']) and 'group' not in c), None)
        s_col = next((c for c in df.columns if any(x in c for x in ['sales', 'tot-amt', 'total amt', 'amt', 'value'])), None)
        r_col = next((c for c in df.columns if any(x in c for x in ['rate', 'price', 'unit price'])), None)
        if not q_col or not s_col:
            continue
        dept = "Ice-Cream" if 'ice' in fname else "Chocolate" if 'choco' in fname else "Sip-up" if 'sip' in fname else "General"
        date_col = next((c for c in df.columns if 'date' in c), None)
        if date_col:
            df['date_dt'] = pd.to_datetime(df[date_col], dayfirst=True, format='mixed', errors='coerce')
        else:
            df['date_dt'] = df[p_col].str.extract(r'(\d{2}-\d{2}-\d{4})').pipe(pd.to_datetime, dayfirst=True, errors='coerce').ffill()
        df = df.dropna(subset=['date_dt']).copy()
        df['qty_val'] = pd.to_numeric(df[q_col], errors='coerce').fillna(0)
        df['sales_val'] = pd.to_numeric(df[s_col], errors='coerce').fillna(0)
        df['price_val'] = pd.to_numeric(df[r_col], errors='coerce').fillna(0) if r_col else (df['sales_val'] / df['qty_val'].replace(0, 1))
        clean = pd.DataFrame({
            'product': df[p_col].astype(str).str.strip(),
            'qty': df['qty_val'],
            'sales': df['sales_val'],
            'price': df['price_val'],
            'date': df['date_dt'],
            'year': df['date_dt'].dt.year,
            'month_name': df['date_dt'].dt.strftime('%b'),
            'month': df['date_dt'].dt.month,
            'dept': dept,
            'order_type': df['qty_val'].apply(lambda x: 'Bulk' if x >= 10 else 'Retail')
        })
        clean = clean[clean['product'].str.contains(r'^\d', na=False)]
        all_data.append(clean)
    if not all_data:
        return pd.DataFrame()
    full_df = pd.concat(all_data, ignore_index=True)
    full_df['month_name'] = pd.Categorical(full_df['month_name'], categories=m_order, ordered=True)
    return full_df


def create_toggle_chart(data_df, title_text, label_prefix=""):
    trend_df = data_df.groupby(['year', 'month_name'], observed=False).agg({'sales': 'sum', 'qty': 'sum'}).reset_index().sort_values(['year', 'month_name'])
    years = sorted(trend_df['year'].unique())
    n_years = len(years)
    fig = go.Figure()
    for yr in years:
        yr_d = trend_df[trend_df['year'] == yr]
        fig.add_trace(go.Scatter(x=yr_d['month_name'], y=yr_d['sales'], name=f"{label_prefix}Rev {yr}", mode='lines+markers', visible=True))
    for yr in years:
        yr_d = trend_df[trend_df['year'] == yr]
        fig.add_trace(go.Scatter(x=yr_d['month_name'], y=yr_d['qty'], name=f"{label_prefix}Qty {yr}", mode='lines+markers', visible=False))
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="left",
            buttons=[
                dict(label="Revenue", method="update", args=[{"visible": [True]*n_years + [False]*n_years}, {"yaxis": {"title": "Total Revenue (₹)"}}]),
                dict(label="Quantity", method="update", args=[{"visible": [False]*n_years + [True]*n_years}, {"yaxis": {"title": "Total Quantity (Units)"}}])
            ],
            pad={"r": 10, "t": 10}, showactive=True, x=0, xanchor="left", y=1.2, yanchor="top"
        )],
        xaxis=dict(title="Month"),
        yaxis=dict(title="Total Revenue (₹)"),
        template="plotly_white",
        title=f"<b>{title_text}</b>",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    return fig


# ---------------------------------------------------------
# HEALTH METRICS FUNCTION
# ---------------------------------------------------------
def get_health_metrics(data):
    if data.empty:
        return 0, 1, "NO DATA", "gray"
    m_rev = data.groupby(data['date'].dt.to_period('M')).agg({'sales': 'sum'}).reset_index()
    m_rev['idx'] = range(len(m_rev))
    if len(m_rev) < 2:
        return 0, 0.5, "INSUFFICIENT DATA", "white"
    z = np.polyfit(m_rev['idx'].astype(float), m_rev['sales'].astype(float), 1)
    slope, mean_sales = z[0], m_rev['sales'].mean()
    cv = m_rev['sales'].std() / mean_sales if mean_sales > 0 else 1
    if slope > 0 and cv < 0.5:
        v, c = "EXCELLENT: Stable Growth", "#2ecc71"
    elif slope > 0:
        v, c = "GOOD BUT VOLATILE", "#f1c40f"
    elif cv < 0.4:
        v, c = "STAGNANT: High Competition", "#e67e22"
    else:
        v, c = "CRITICAL: Declining/Unstable", "#e74c3c"
    return slope, cv, v, c


# 3. INTERFACE
with st.sidebar:
    st.title(" Strategic Control")
    st.markdown("---")
    files = st.file_uploader("📂 Upload Sales Data (2020-2025)", type="csv", accept_multiple_files=True)
    st.markdown("---")
    st.caption("Post-Graduate Research Project")
    st.caption("Subject: Malabar Milma Analytics")

if files:
    df = load_and_clean_data(files)
    if not df.empty:
        menu = st.sidebar.radio("Navigation", [
            "Executive Summary",
            "Product-Wise Trend",
            "Product Rankings",
            "Elasticity Engine",
            "Price Optimization",
            "Strategic Clustering",
            "Strategic Recommendations",
            "Risk & Forecast"
        ])
        sel_dept = st.sidebar.selectbox("Department", sorted(df['dept'].unique()))
        v = df[df['dept'] == sel_dept].copy()

        # ── EXECUTIVE SUMMARY ──────────────────────────────────────
        if menu == "Executive Summary":
            st.title(f" {sel_dept} Performance Overview")
            c1, c2, c3 = st.columns(3)
            active_list = sorted(v['product'].unique())
            c1.metric("Active Products", len(active_list))
            c2.metric("Total Revenue", f"₹{v['sales'].sum():,.0f}")
            c3.metric("Total Qty", f"{int(v['qty'].sum()):,}")
            st.markdown("---")
            st.subheader("Category Performance (Revenue vs Quantity)")
            st.plotly_chart(create_toggle_chart(v, f"Overall {sel_dept} Monthly Trend"), use_container_width=True)

        # ── PRODUCT-WISE TREND ─────────────────────────────────────
        elif menu == "Product-Wise Trend":
            st.title(" Product Monthly Analysis")
            active_list = sorted(v['product'].unique())
            if st.toggle(" Show Full Master Product List"):
                num_cols = 4
                list_cols = st.columns(num_cols)
                for i, item in enumerate(active_list):
                    list_cols[i % num_cols].write(f"• {item}")
            p_sel = st.selectbox("Select Product to Analyze", active_list)
            p_df = v[v['product'] == p_sel]
            st.plotly_chart(create_toggle_chart(p_df, f"Trend for {p_sel}"), use_container_width=True)

        # ── PRODUCT RANKINGS ───────────────────────────────────────
        elif menu == "Product Rankings":
            st.title(" Rankings & Consistency")
            t1, t2, t3, t4 = st.tabs(["Performance", "Yearly Lifecycle", "Consistency (Old vs New)", "Revenue Share: Bulk vs Retail"])

            with t1:
                st.plotly_chart(
                    px.bar(
                        v.groupby('product')['sales'].sum().nlargest(10).reset_index(),
                        x='sales', y='product', orientation='h', title="Top 10 Products"
                    ), use_container_width=True
                )

            with t2:
                yearly_sets = v.groupby('year')['product'].apply(lambda x: set(x.str.strip().unique())).to_dict()
                all_yrs = sorted(list(yearly_sets.keys()))
                target_yr = st.selectbox("Lifecycle Year", all_yrs, index=len(all_yrs)-1)
                st.metric(f"Total Products Sold in {target_yr}", len(yearly_sets[target_yr]))
                curr = yearly_sets[target_yr]
                past_yrs = [y for y in all_yrs if y < target_yr]
                seen_before = set().union(*(yearly_sets[y] for y in past_yrs)) if past_yrs else set()
                prev_yr = max(past_yrs) if past_yrs else None
                prev_s = yearly_sets[prev_yr] if prev_yr else set()
                b_new = sorted(list(curr - seen_before))
                drop = sorted(list(prev_s - curr))
                ret = sorted(list((curr & seen_before) - prev_s))

                # FIX: proper for loops inside with blocks to remove [...] artifacts
                l1, l2, l3 = st.columns(3)
                with l1:
                    st.info(f"NEW ({len(b_new)})")
                    for p in b_new:
                        st.write(f"+ {p}")
                with l2:
                    st.error(f"DROPPED ({len(drop)})")
                    for p in drop:
                        st.write(f"- {p}")
                with l3:
                    st.warning(f"RETURNING ({len(ret)})")
                    for p in ret:
                        st.write(f"↺ {p}")

            with t3:
                st.subheader(" The Consistency Dashboard")
                all_yrs = sorted(v['year'].unique())
                consistent_prods = v.groupby('product')['year'].nunique()
                old_faithfuls = consistent_prods[consistent_prods == len(all_yrs)].index.tolist()
                recent_yrs = all_yrs[-2:]
                first_seen = v.groupby('product')['year'].min()
                newbies = first_seen[first_seen.isin(recent_yrs)].index.tolist()
                metric = st.radio("Compare by:", ["Revenue", "Quantity"], horizontal=True)
                val_col = 'sales' if metric == "Revenue" else 'qty'
                old_avg = v[v['product'].isin(old_faithfuls)].groupby('year')[val_col].mean()
                new_avg = v[v['product'].isin(newbies)].groupby('year')[val_col].mean()
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(x=old_avg.index, y=old_avg.values, name="Old Faithfuls (Avg)", line=dict(color='gold', width=4)))
                fig_c.add_trace(go.Scatter(x=new_avg.index, y=new_avg.values, name="Newbies (Avg)", line=dict(color='cyan', dash='dot')))
                fig_c.update_layout(title=f"Consistency: {metric} Comparison", template="plotly_white")
                st.plotly_chart(fig_c, use_container_width=True)

            with t4:
                st.markdown("### 📊 Sales Channel Distribution")
                split = v.groupby('order_type')['sales'].sum().reset_index()
                fig_pie = go.Figure(data=[go.Pie(
                    labels=split['order_type'],
                    values=split['sales'],
                    hole=0.5,
                    marker=dict(colors=['#1F4959', '#5C7C89']),
                    textinfo='label+percent',
                    hovertemplate='<b>%{label}</b><br>Revenue: ₹%{value:,.0f}<br>Share: %{percent}<extra></extra>'
                )])
                fig_pie.update_layout(
                    title_text="Revenue Share: Bulk vs Retail",
                    template="plotly_dark",
                    font_family="Times New Roman",
                    font=dict(color="white"),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5,
                        font=dict(color="white")
                    )
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # ── ELASTICITY ENGINE ──────────────────────────────────────
        elif menu == "Elasticity Engine":
            st.title(" Advanced Product Categorization")
            tab_inventory, tab_seasonality, tab_cannibal, tab_fin = st.tabs([
                "Inventory Sensitivity Map",
                "The 'Heat Effect' Trend",
                "Cannibalization Audit",
                "Financial Impact Audit"
            ])

            inventory_table = get_final_report(v)

            with tab_inventory:
                st.subheader("Technical Accuracy Guide")
                g1, g2 = st.columns(2)
                with g1:
                    st.success("**Perfect P-Value: < 0.05**")
                    st.write("This means there is a 95% confidence that the price change actually caused the sales change.")
                with g2:
                    st.info("**Perfect Elasticity Score: < -1.0**")
                    st.write("A score deeper than -1.0 (e.g., -2.5) means the product is 'Sensitive'—customers react strongly to price.")
                st.markdown("---")
                st.write("**Full Product Inventory Report (Seasonal & Sensitivity Breakdown):**")
                st.dataframe(inventory_table, use_container_width=True)
                st.markdown("---")
                target_p = st.selectbox("Select Product to View Detailed Hover Logic:", inventory_table['Product Name'].unique())
                p_plot = v[v['product'] == target_p].copy()
                p_plot['date_str'] = p_plot['date'].dt.strftime('%Y-%m-%d')
                fig_sens = px.scatter(
                    p_plot, x='price', y='qty', trendline="ols",
                    hover_data={'date_str': True, 'price': True, 'qty': True},
                    labels={'date_str': 'Date'},
                    title=f"<b>Price-Demand Logic: {target_p}</b>"
                )
                st.plotly_chart(fig_sens, use_container_width=True)
                st.markdown("### ▶ How to Read This Chart")
                st.markdown("""
                **What are the Dots?**
                * Each **blue dot** is a real historical sales record from your data.
                * **Horizontal (Price):** Moving right means the price was higher.
                * **Vertical (Qty):** Moving up means more units were sold.
                * *Insight:* Stacks of dots at one price point show where demand varied at the same price.

                **What is the Light Blue Line?**
                * This is the **Trend Line** (Linear Regression). It calculates the 'average' behavior of your customers.
                * **The Downward Slope:** This confirms that as prices raise, the quantity sold usually drops (Law of Demand).
                * **The Gap:** Dots above the line are "Sales Wins" (high volume), while dots below indicate stock-outs or low demand periods.
                """)

            with tab_seasonality:
                st.subheader(f"Kerala's Climate & Volume Correlation: {sel_dept}")
                seasonal_trend = v.groupby(['month_name'], observed=False).agg({'qty': 'sum', 'price': 'mean'}).reset_index()
                fig_heat = px.line(
                    seasonal_trend, x='month_name', y='qty',
                    title=f"The 'Heat Effect': Seasonal Volume Trend ({sel_dept})",
                    markers=True, template='plotly_white'
                )
                fig_heat.add_vrect(x0="Mar", x1="May", fillcolor="orange", opacity=0.2, annotation_text="Peak Summer")
                fig_heat.add_vrect(x0="Jun", x1="Aug", fillcolor="blue", opacity=0.1, annotation_text="Monsoon")
                fig_heat.update_layout(hovermode="x unified")
                st.plotly_chart(fig_heat, use_container_width=True)

            with tab_cannibal:
                st.subheader("Product Cannibalization Audit")
                pivot_qty = v.pivot_table(index='date', columns='product', values='qty', aggfunc='sum').fillna(0)
                corr_matrix = pivot_qty.corr()

                def check_cannibalization(target_product, data_df, p_qty, c_mat):
                    try:
                        correlations = c_mat[target_product].drop(target_product).sort_values(ascending=False)
                        potential_competitor = correlations.index[0]

                        # FIX: Rename columns before joining to avoid column count mismatch
                        target_col = p_qty[[target_product]].copy()
                        target_col.columns = ['target_qty']
                        comp_col = p_qty[[potential_competitor]].copy()
                        comp_col.columns = ['comp_qty']
                        target_prices = data_df[data_df['product'] == target_product].groupby('date')['price'].mean()
                        target_prices.name = 'target_price'
                        merged_data = target_col.join(comp_col, how='inner').join(target_prices, how='inner').dropna()

                        if len(merged_data) < 3:
                            return target_product, potential_competitor, 0, "N/A", "Insufficient Data"

                        X_cross = sm.add_constant(np.log(merged_data['target_price']))
                        y_cross = np.log(merged_data['comp_qty'] + 1)
                        cross_model = sm.OLS(y_cross, X_cross).fit()
                        cross_elast = cross_model.params.iloc[1]
                        p_value = cross_model.pvalues.iloc[1]

                        if p_value >= 0.05:
                            status = "⚪ Not Significant"
                        elif cross_elast > 0:
                            status = "🚨 Cannibalization"
                        else:
                            status = "✅ Complementary"

                        return target_product, potential_competitor, round(cross_elast, 3), round(p_value, 4), status

                    except Exception as e:
                        return target_product, "N/A", 0, "N/A", f"Error: {str(e)}"

                can_res = []
                for p in v['product'].unique():
                    can_res.append(check_cannibalization(p, v, pivot_qty, corr_matrix))
                st.dataframe(
                    pd.DataFrame(can_res, columns=['Target Product', 'Strongest Competitor', 'Cross-Elasticity', 'P-Value', 'Market Status']),
                    use_container_width=True
                )

            with tab_fin:
                st.subheader("Financial Impact Audit")

                def calculate_rupee_impact(target_prod, data_df, inv_table):
                    p_row = inv_table[inv_table['Product Name'] == target_prod]
                    e = p_row['Elasticity'].values[0]
                    p_data = data_df[data_df['product'] == target_prod]
                    cur_qty, avg_p = p_data['qty'].sum(), p_data['price'].mean()
                    cur_rev = cur_qty * avg_p
                    results = []
                    for hike in [0.05, 0.10, 0.15, 0.20]:
                        new_p = avg_p * (1 + hike)
                        new_qty = max(0, cur_qty * ((1 + hike) ** e))
                        new_rev = new_qty * new_p
                        results.append({
                            'Price Hike %': f"{hike*100:.0f}%",
                            'New Price': f"₹{new_p:.2f}",
                            'New Revenue': f"₹{new_rev:,.2f}",
                            'Rupee Gain/Loss': f"₹{new_rev - cur_rev:,.2f}"
                        })
                    return pd.DataFrame(results)

                f_target = st.selectbox("Select Product to Audit Impact", sorted(inventory_table['Product Name'].unique()))
                st.table(calculate_rupee_impact(f_target, v, inventory_table))

        # ── PRICE OPTIMIZATION ─────────────────────────────────────
        elif menu == "Price Optimization":
            st.title("Strategic Price Engine")
            tab_chart, tab_audit = st.tabs(["Optimization Curve", " Revenue Gap Audit"])

            with tab_chart:
                product_list = v['product'].unique()
                fig = go.Figure()
                for prod in product_list:
                    df_p = v[v['product'] == prod]
                    if len(df_p) > 5:
                        X, y = df_p[['price']].values, df_p['qty'].values
                        poly = PolynomialFeatures(degree=2, include_bias=False)
                        X_poly = poly.fit_transform(X)
                        model = LinearRegression().fit(X_poly, y)
                        y_pred_all = model.predict(X_poly)
                        r2 = r2_score(y, y_pred_all)
                        p_range = np.linspace(X.min() * 0.7, X.max() * 1.4, 100).reshape(-1, 1)
                        pred_qty = model.predict(poly.transform(p_range))
                        pred_rev = p_range.flatten() * pred_qty
                        opt_idx = np.argmax(pred_rev)
                        opt_p, opt_q = p_range[opt_idx][0], pred_qty[opt_idx]
                        fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name=f'Actual {prod}', visible=False, marker=dict(color='black', opacity=0.6)))
                        fig.add_trace(go.Scatter(x=p_range.flatten(), y=pred_qty, mode='lines', name=f'Curve (R²: {r2:.2f})', visible=False, line=dict(color='red', width=3)))
                        fig.add_trace(go.Scatter(x=[opt_p], y=[opt_q], mode='markers+text', text=[f"OPTIMAL: ₹{opt_p:.2f}"], textposition="top center", name=f"Peak {prod}", visible=False, marker=dict(color='blue', size=14, symbol='diamond')))
                buttons = []
                for i, prod in enumerate(product_list):
                    visibility = [False] * len(fig.data)
                    visibility[i*3 : i*3 + 3] = [True, True, True]
                    buttons.append(dict(label=prod, method="update", args=[{"visible": visibility}, {"title": f"<b>{prod} Strategy</b>"}]))
                if len(fig.data) > 0:
                    for j in range(3):
                        fig.data[j].visible = True
                fig.update_layout(
                    updatemenus=[dict(active=0, buttons=buttons, x=0, y=1.2, xanchor="left", yanchor="top")],
                    title="<b>MILMA PRICE OPTIMIZATION ENGINE</b>",
                    xaxis_title="Price per Unit (₹)",
                    yaxis_title="Demand (Units)",
                    template="plotly_white",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab_audit:
                audit_data = []
                prod_list_audit = v['product'].unique()
                for prod in prod_list_audit:
                    df_p = v[v['product'] == prod]
                    if len(df_p) > 5:
                        X, y = df_p[['price']].values, df_p['qty'].values
                        poly = PolynomialFeatures(degree=2, include_bias=False)
                        model = LinearRegression().fit(poly.fit_transform(X), y)
                        p_range = np.linspace(X.min() * 0.5, X.max() * 1.5, 200).reshape(-1, 1)
                        pred_qty = model.predict(poly.transform(p_range))
                        current_avg_price, current_rev = X.mean(), X.mean() * y.mean()
                        rev_curve = p_range.flatten() * pred_qty
                        opt_idx = np.argmax(rev_curve)
                        opt_price, opt_rev = p_range[opt_idx][0], rev_curve[opt_idx]
                        gap = opt_rev - current_rev
                        audit_data.append({
                            'Product': prod,
                            'Current Price': round(current_avg_price, 2),
                            'Optimal Price': round(opt_price, 2),
                            'Current Revenue': round(current_rev, 2),
                            'Potential Revenue': round(opt_rev, 2),
                            'Revenue Gap (₹)': round(gap, 2),
                            'Improvement (%)': round((gap / current_rev) * 100, 2)
                        })
                if audit_data:
                    audit_df = pd.DataFrame(audit_data).sort_values(by='Revenue Gap (₹)', ascending=False)
                    st.write("**REVENUE GAP AUDIT COMPLETED**")
                    st.dataframe(audit_df.head(129), use_container_width=True)

        # ── STRATEGIC CLUSTERING ───────────────────────────────────
        elif menu == "Strategic Clustering":
            st.title("Strategic Portfolio K-Means")
            inv_rep = get_final_report(v)
            agg_stats = v.groupby('product').agg({'qty': 'sum', 'sales': 'sum', 'price': 'mean'}).reset_index()
            master_table = pd.merge(agg_stats, inv_rep[['Product Name', 'Elasticity']], left_on='product', right_on='Product Name').drop('Product Name', axis=1)
            master_table.columns = ['product', 'qty', 'revenue', 'price', 'Elasticity']
            features = ['qty', 'revenue', 'price', 'Elasticity']
            X_clus = master_table[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clus)
            kmeans = KMeans(n_clusters=min(len(X_clus), 4), random_state=42, n_init=20)
            master_table['cluster_id'] = kmeans.fit_predict(X_scaled)
            cluster_stats = master_table.groupby('cluster_id')[features].mean()
            star_id = cluster_stats['revenue'].idxmax()
            out_id = cluster_stats['revenue'].idxmin()
            remaining_ids = cluster_stats.drop([star_id, out_id]).index
            premium_id = cluster_stats.loc[remaining_ids, 'price'].idxmax()
            push_id = remaining_ids.difference([premium_id])[0]
            strategic_map = {
                star_id: "The Core Engines (Star)",
                premium_id: "The High-Value Staples (Premium)",
                push_id: "The High-Volume Risks (Push)",
                out_id: "The Low-Yield Candidates (Out)"
            }
            master_table['Strategic Cluster'] = master_table['cluster_id'].map(strategic_map)
            st.write("Cluster count")
            st.write(master_table['Strategic Cluster'].value_counts())
            st.plotly_chart(px.scatter_3d(master_table, x='revenue', y='qty', z='price', color='Strategic Cluster', hover_name='product', title="Strategic Portfolio Segments"))
            st.dataframe(master_table.sort_values('Strategic Cluster'), use_container_width=True)

        # ── STRATEGIC RECOMMENDATIONS ──────────────────────────────
        elif menu == "Strategic Recommendations":
            st.title(" Strategic Product Audit")
            target_p = st.selectbox("Select Product for Triple-Model Audit", sorted(v['product'].unique()))
            p_df = v[v['product'] == target_p].sort_values('date')

            if len(p_df) >= 3:
                z = np.polyfit(range(len(p_df)), p_df['sales'], 1)
                slope = z[0]

                try:
                    p_agg = p_df.groupby('date').agg({'qty': 'sum', 'price': 'mean'}).reset_index()
                    p_agg = p_agg[(p_agg['qty'] > 0) & (p_agg['price'] > 0)]
                    X_e = sm.add_constant(np.log(p_agg['price']))
                    y_e = np.log(p_agg['qty'])
                    e_model = sm.OLS(y_e, X_e).fit()
                    elasticity = e_model.params.iloc[1]
                except:
                    elasticity = 0

                avg_cat_rev = v.groupby('product')['sales'].sum().mean()
                prod_total_rev = p_df['sales'].sum()
                market_rank = "High Volume" if prod_total_rev > avg_cat_rev else "Niche/Low Volume"

                if slope > 0 and elasticity > -1.0:
                    verdict, color = " STRATEGIC FOCUS: YES", "#2ecc71"
                    reason = "Product is growing organically and is INELASTIC. Customers accept current pricing. Focus on expansion."
                elif slope > 0 and elasticity <= -1.0:
                    verdict, color = " CONDITIONAL FOCUS: MONITOR", "#f1c40f"
                    reason = "Growth is present but highly ELASTIC. Any price increase will drastically drop volume. Focus on cost control."
                else:
                    verdict, color = " STRATEGIC FOCUS: NO", "#e74c3c"
                    reason = "Declining trend or poor revenue performance. Recommend product refresh or SKU rationalization."

                st.markdown(f"""
                    <div style="background:{color}; padding:25px; border-radius:10px; margin-bottom:20px; border: 1px solid rgba(255,255,255,0.2);">
                        <h2 style="color:white; margin:0;">{verdict}</h2>
                        <p style="color:white; font-size:18px; margin-top:10px; font-family:'Times New Roman';">{reason}</p>
                    </div>
                """, unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Growth Slope (Reg)", f"{slope:+.2f}")
                m2.metric("Elasticity Score", f"{elasticity:.2f}")
                m3.metric("Portfolio Position", market_rank)

                st.markdown("### Evidence Visualization")
                fig_col1, fig_col2 = st.columns(2)

                with fig_col1:
                    fig_trend = px.scatter(p_df, x='date', y='sales', trendline="ols",
                                          title=f"Regression Trendline: {target_p}", template="plotly_dark")
                    fig_trend.update_traces(marker=dict(color=color))
                    st.plotly_chart(fig_trend, use_container_width=True)

                with fig_col2:
                    fig_price = px.scatter(p_df, x='price', y='qty',
                                          title=f"Price vs Demand (Sensitivity): {target_p}", template="plotly_dark")
                    fig_price.update_traces(marker=dict(color="#5C7C89"))
                    st.plotly_chart(fig_price, use_container_width=True)
            else:
                st.warning("Insufficient historical data for this product (Need 3+ periods).")

        # ── RISK & FORECAST ────────────────────────────────────────
        elif menu == "Risk & Forecast":
            st.title(" Strategic Health & 2026 Projections")
            tab_total, tab_product = st.tabs([" Category Health", " Product-Specific Audit"])

            with tab_total:
                slope, cv, verdict, color = get_health_metrics(v)
                st.markdown(f"### Overall Category Status: <span style='color:{color}'>{verdict}</span>", unsafe_allow_html=True)
                fig_meter = go.Figure(go.Indicator(
                    mode="gauge+number", value=max(0, (1 - cv) * 100),
                    title={'text': "Stability Score", 'font': {'size': 18}},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}}
                ))
                fig_meter.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=250)
                st.plotly_chart(fig_meter, use_container_width=True)
                annual = v.groupby('year')['sales'].sum().reset_index()
                fig_cat = px.line(annual, x='year', y='sales', title="Whole Category Revenue Trend", markers=True)
                fig_cat.update_traces(line_color=color)
                st.plotly_chart(fig_cat, use_container_width=True)

            with tab_product:
                target_p = st.selectbox("Select Product for Risk Audit", sorted(v['product'].unique()))
                p_data = v[v['product'] == target_p]
                slope_p, cv_p, verdict_p, color_p = get_health_metrics(p_data)
                p_annual = p_data.groupby('year')['sales'].sum().reset_index()

                if len(p_annual) >= 2:
                    X = p_annual['year'].values.reshape(-1, 1)
                    y = p_annual['sales'].values
                    if len(p_annual) >= 5:
                        poly = PolynomialFeatures(degree=2)
                        X_fit = poly.fit_transform(X)
                        X_pred_2026 = poly.transform([[2026]])
                        xr_transformed = poly.transform(np.arange(p_annual['year'].min(), 2027).reshape(-1, 1))
                    else:
                        poly = None
                        X_fit = X
                        X_pred_2026 = [[2026]]
                        xr_transformed = np.arange(p_annual['year'].min(), 2027).reshape(-1, 1)
                    model = LinearRegression().fit(X_fit, y)
                    pred_2026 = max(0, model.predict(X_pred_2026)[0])
                    last_year_rev = y[-1] if y[-1] != 0 else 1
                    delta_pct = ((pred_2026 / last_year_rev) - 1) * 100
                    st.metric(
                        f"Predicted 2026 Revenue for {target_p}",
                        f"₹{pred_2026:,.2f}",
                        delta=f"{delta_pct:.1f}% vs Last Year"
                    )
                    xr = np.arange(p_annual['year'].min(), 2027).reshape(-1, 1)
                    fig_f = go.Figure()
                    fig_f.add_trace(go.Scatter(x=p_annual['year'], y=y, name='Actual Sales', mode='lines+markers'))
                    fig_f.add_trace(go.Scatter(
                        x=xr.flatten(),
                        y=model.predict(xr_transformed),
                        name='2026 Forecast Trend',
                        line=dict(dash='dash', color='red')
                    ))
                    model_label = "Polynomial" if poly else "Linear"
                    fig_f.update_layout(title=f"2026 Forecast ({model_label} Model)", template="plotly_dark")
                    st.plotly_chart(fig_f, use_container_width=True)
                else:
                    st.warning("Insufficient historical data for 2026 forecasting (Need 2+ years).")

                st.markdown(f"**Health Audit:** {verdict_p}")

else:
    st.info("Upload CSV files to begin.")
