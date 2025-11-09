# =============================================================================
# STREAMLIT WEB APPLICATION
# Interactive dashboard for recommendation system with table display
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add the current directory to path to import recommendation_engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommendation_engine import RecommendationEngine

# Page configuration
st.set_page_config(
    page_title="🛍️ Online Retail Recommender",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .recommendation-table {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .customer-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    """Load recommendation engine with caching"""
    try:
        # Try multiple file paths
        possible_paths = [
            'OnlineRetail.csv',
            'data/OnlineRetail.csv',
            '../OnlineRetail.csv',
            './data/OnlineRetail.csv'
        ]

        for path in possible_paths:
            try:
                if os.path.exists(path):
                    engine = RecommendationEngine(path)
                    st.success(f"✅ Successfully loaded data from: {path}")
                    return engine
            except Exception as e:
                continue

        st.error("❌ Could not load any data file. Please ensure your CSV file is available.")
        return None

    except Exception as e:
        st.error(f"❌ Error loading recommendation engine: {e}")
        return None


def format_recommendations_table(recommendations):
    """Format recommendations for table display"""
    if not recommendations:
        return pd.DataFrame()

    # Create DataFrame from recommendations
    df = pd.DataFrame(recommendations)

    # Format the DataFrame for better display
    df_display = df.copy()
    df_display['Rank'] = range(1, len(df_display) + 1)
    df_display['Score'] = df_display['Score'].apply(lambda x: f"{x:.3f}")

    # Reorder columns
    columns_order = ['Rank', 'Description', 'StockCode', 'Score', 'Reason']
    df_display = df_display[columns_order]

    # Rename columns for better readability
    df_display = df_display.rename(columns={
        'Rank': '🏆 Rank',
        'Description': '📦 Product Description',
        'StockCode': '🆔 Product ID',
        'Score': '⭐ Score',
        'Reason': '💡 Recommendation Reason'
    })

    return df_display


def display_customer_metrics(engine, customer_id):
    """Display customer metrics in a formatted way"""
    customer_metrics = engine.get_customer_metrics(customer_id)
    customer_data = engine.df[engine.df['CustomerID'] == customer_id]

    if customer_metrics:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "💰 Total Spent",
                f"${customer_metrics['total_spent']:,.2f}",
                help="Total amount spent by the customer"
            )

        with col2:
            st.metric(
                "📦 Total Orders",
                customer_metrics['total_orders'],
                help="Number of unique orders placed"
            )

        with col3:
            st.metric(
                "🛍️ Unique Products",
                customer_metrics['total_products'],
                help="Number of different products purchased"
            )

        with col4:
            if customer_metrics['avg_order_value'] > 0:
                st.metric(
                    "📈 Avg Order Value",
                    f"${customer_metrics['avg_order_value']:,.2f}",
                    help="Average spending per order"
                )
            else:
                st.metric("📈 Avg Order Value", "$0.00")
    else:
        st.warning("No data found for selected customer")


def main():
    # Header
    st.markdown('<h1 class="main-header">🛍️ Online Retail Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Load recommendation engine
    with st.spinner('🚀 Loading recommendation engine and data... This may take a minute.'):
        engine = load_engine()

    if engine is None or engine.df.empty:
        st.error("""
        ❌ Failed to load the recommendation engine. 

        Please ensure you have the dataset file available. Expected files:
        - `OnlineRetail.csv` (in current directory)
        - `data/OnlineRetail.csv` 

        The dataset should contain columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
        """)

        # Show file explorer to help user
        with st.expander("📁 File Explorer Help"):
            st.write("Current directory contents:")
            current_files = [f for f in os.listdir('.') if f.endswith('.csv')]
            if current_files:
                for file in current_files:
                    st.write(f"   - {file}")
            else:
                st.write("   No CSV files found in current directory")

            if os.path.exists('data'):
                st.write("Data directory contents:")
                data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
                for file in data_files:
                    st.write(f"   - {file}")
        return

    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration Panel")

    # Customer selection
    st.sidebar.subheader("👤 Customer Selection")
    available_customers = engine.df['CustomerID'].dropna().unique()

    if len(available_customers) == 0:
        st.error("No customers found in the data!")
        return

    selected_customer = st.sidebar.selectbox(
        "Choose Customer ID:",
        options=sorted(available_customers)[:100],  # Limit for performance
        index=0,
        help="Select a customer to generate personalized recommendations"
    )

    # Recommendation settings
    st.sidebar.subheader("🎯 Recommendation Settings")

    n_recommendations = st.sidebar.slider(
        "Number of recommendations:",
        min_value=5,
        max_value=25,
        value=10,
        help="Adjust how many recommendations to generate"
    )

    rec_type = st.sidebar.radio(
        "Recommendation Algorithm:",
        ["Hybrid (All Methods)", "Item-Based CF", "Association Rules", "Popular Products"],
        help="Choose the recommendation algorithm strategy"
    )

    # Display algorithm description
    algorithm_descriptions = {
        "Hybrid (All Methods)": "Combines collaborative filtering and association rules for comprehensive recommendations",
        "Item-Based CF": "Recommends products similar to what the customer has purchased before",
        "Association Rules": "Suggests products frequently bought together with customer's purchases",
        "Popular Products": "Shows overall best-selling products as fallback"
    }

    st.sidebar.info(f"**Algorithm**: {algorithm_descriptions[rec_type]}")

    # Main content area
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Customer Profile")

        # Customer header
        st.markdown(f"""
        <div class="customer-header">
            <h3>👤 Customer {selected_customer}</h3>
            <p>Personalized Recommendations</p>
        </div>
        """, unsafe_allow_html=True)

        # Customer metrics
        display_customer_metrics(engine, selected_customer)

        # Recent purchases
        st.subheader("🛒 Recent Purchase History")
        recent_purchases = engine.get_customer_history(selected_customer)

        if not recent_purchases.empty:
            # Format for display
            display_df = recent_purchases.copy()
            if 'InvoiceDate' in display_df.columns:
                display_df['InvoiceDate'] = display_df['InvoiceDate'].dt.strftime('%Y-%m-%d')
            display_df['UnitPrice'] = display_df['UnitPrice'].apply(lambda x: f"${x:.2f}")
            display_df['TotalPrice'] = display_df['TotalPrice'].apply(lambda x: f"${x:.2f}")

            st.dataframe(
                display_df[['InvoiceDate', 'Description', 'Quantity', 'UnitPrice', 'TotalPrice']],
                use_container_width=True,
                hide_index=True,
                height=300
            )
        else:
            st.info("📝 No purchase history found for this customer.")

    with col2:
        st.subheader("🎯 Smart Recommendations")

        # Generate recommendations button
        if st.button("🚀 Generate Recommendations", type="primary", use_container_width=True):
            with st.spinner('🔍 Analyzing purchase patterns and generating recommendations...'):
                try:
                    # Get recommendations based on selected type
                    if rec_type == "Item-Based CF":
                        recommendations = engine.item_based_recommendations(selected_customer, n_recommendations)
                        st.info(
                            "🔍 Using **Item-Based Collaborative Filtering** - Finding products similar to your purchases")
                    elif rec_type == "Association Rules":
                        recommendations = engine.association_rule_recommendations(selected_customer, n_recommendations)
                        st.info("🛒 Using **Market Basket Analysis** - Products frequently bought together")
                    elif rec_type == "Popular Products":
                        recommendations = engine.get_popular_products(n_recommendations)
                        st.info("🏆 Showing **Most Popular Products** - Overall best sellers")
                    else:  # Hybrid
                        recommendations = engine.get_recommendations(selected_customer, n_recommendations)
                        st.info("🎯 Using **Hybrid Approach** - Combined methods for best results")

                    # Display recommendations in table format
                    if recommendations:
                        st.success(f"✅ Generated {len(recommendations)} personalized recommendations!")

                        # Format recommendations as table
                        recommendations_table = format_recommendations_table(recommendations)

                        # Display styled table
                        st.markdown("### 📋 Recommendation Results")
                        st.dataframe(
                            recommendations_table,
                            use_container_width=True,
                            hide_index=True,
                            height=400
                        )

                        # Additional insights
                        with st.expander("📊 Recommendation Insights"):
                            st.write(f"**Total Recommendations**: {len(recommendations)}")
                            st.write(f"**Algorithm Used**: {rec_type}")
                            st.write(f"**Customer ID**: {selected_customer}")

                            # Show score distribution
                            if recommendations:
                                scores = [rec['Score'] for rec in recommendations]
                                avg_score = sum(scores) / len(scores)
                                st.write(f"**Average Confidence Score**: {avg_score:.3f}")

                    else:
                        st.warning("⚠️ No specific recommendations found. Showing popular products instead.")
                        popular_recs = engine.get_popular_products(n_recommendations)
                        popular_table = format_recommendations_table(popular_recs)

                        st.dataframe(
                            popular_table,
                            use_container_width=True,
                            hide_index=True,
                            height=400
                        )

                except Exception as e:
                    st.error(f"❌ Error generating recommendations: {str(e)}")
                    st.info("💡 Try selecting a different customer or algorithm")
        else:
            st.info("👆 Click the **'Generate Recommendations'** button above to get personalized product suggestions!")

        # Business Analytics Section
        st.markdown("---")
        st.subheader("📈 Business Overview")

        col3, col4, col5 = st.columns(3)

        with col3:
            total_customers = engine.df['CustomerID'].nunique()
            st.metric("👥 Total Customers", f"{total_customers:,}")

        with col4:
            total_products = engine.df['StockCode'].nunique()
            st.metric("📚 Total Products", f"{total_products:,}")

        with col5:
            total_transactions = engine.df['InvoiceNo'].nunique()
            st.metric("💳 Total Transactions", f"{total_transactions:,}")

        # Top products chart
        st.subheader("🏆 Top Selling Products")
        top_products = engine.df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

        if not top_products.empty:
            fig = px.bar(
                x=top_products.values,
                y=top_products.index,
                orientation='h',
                title="Top 10 Products by Quantity Sold",
                labels={'x': 'Quantity Sold', 'y': 'Product'},
                color=top_products.values,
                color_continuous_scale='viridis'
            )
            fig.update_layout(
                showlegend=False,
                height=400,
                xaxis_title="Quantity Sold",
                yaxis_title="",
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for top products chart.")

    # Data exploration section
    with st.expander("🔍 Dataset Overview & Technical Details"):
        st.write(f"**Dataset Shape**: {engine.df.shape[0]:,} rows × {engine.df.shape[1]} columns")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**First 10 rows:**")
            st.dataframe(engine.df.head(10), use_container_width=True)

        with col2:
            st.write("**Basic Statistics:**")
            st.dataframe(engine.df[['Quantity', 'UnitPrice', 'TotalPrice']].describe(), use_container_width=True)

        # Engine information
        st.write("**Recommendation Engine Info:**")
        st.write(
            f"- Collaborative Filtering Matrix: {engine.user_item_matrix.shape[0]:,} customers × {engine.user_item_matrix.shape[1]:,} products")
        if engine.rules is not None:
            st.write(f"- Association Rules: {len(engine.rules):,} rules generated")
        else:
            st.write("- Association Rules: Not available")


if __name__ == "__main__":
    main()