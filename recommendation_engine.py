# =============================================================================
# RECOMMENDATION ENGINE
# Advanced ML-powered recommendation system with enhanced algorithms
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

warnings.filterwarnings('ignore')


class RecommendationEngine:
    def __init__(self, data_path):
        """
        Initialize the recommendation engine with data
        """
        self.df = self.load_and_clean_data(data_path)
        self.user_item_matrix = None
        self.item_similarity = None
        self.rules = None
        self.prepare_data()
        print("✅ Recommendation Engine initialized successfully!")

    def load_and_clean_data(self, data_path):
        """
        Load and clean the retail dataset with enhanced preprocessing
        """
        try:
            df = pd.read_csv(data_path, encoding='unicode_escape')
            print(f"📥 Original dataset loaded: {df.shape}")

            # Enhanced cleaning pipeline
            initial_count = len(df)

            # Remove rows with missing critical fields
            df = df.dropna(subset=['CustomerID', 'Description'])
            print(f"✅ Removed rows with missing CustomerID/Description: {initial_count - len(df)}")

            # Remove duplicates
            initial_count = len(df)
            df = df.drop_duplicates()
            print(f"✅ Removed duplicate rows: {initial_count - len(df)}")

            # Filter out invalid quantities and prices
            df = df[df['Quantity'] > 0]
            df = df[df['UnitPrice'] > 0]
            df = df[df['UnitPrice'] < 1000]  # Remove extreme outliers
            print(f"✅ Filtered invalid quantities/prices")

            # Create TotalPrice column
            df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

            # Convert and extract date features
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            df['InvoiceYear'] = df['InvoiceDate'].dt.year
            df['InvoiceMonth'] = df['InvoiceDate'].dt.month
            df['InvoiceDay'] = df['InvoiceDate'].dt.day
            df['InvoiceHour'] = df['InvoiceDate'].dt.hour

            print(f"📊 Cleaned dataset shape: {df.shape}")
            return df

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return pd.DataFrame()

    def prepare_data(self):
        """
        Prepare data for different recommendation approaches
        """
        self.prepare_collaborative_filtering()
        self.prepare_association_rules()

    def prepare_collaborative_filtering(self):
        """
        Create user-item matrix for collaborative filtering with enhanced features
        """
        try:
            # Create user-item matrix with binary encoding (purchased/not purchased)
            user_item = self.df.groupby(['CustomerID', 'StockCode'])['Quantity'].sum().unstack(fill_value=0)
            self.user_item_matrix = user_item.applymap(lambda x: 1 if x > 0 else 0)

            # Calculate item-item similarity using cosine similarity
            self.item_similarity = cosine_similarity(self.user_item_matrix.T)
            self.item_similarity_df = pd.DataFrame(
                self.item_similarity,
                index=self.user_item_matrix.columns,
                columns=self.user_item_matrix.columns
            )

            print(
                f"✅ Collaborative filtering prepared: {self.user_item_matrix.shape[0]:,} customers × {self.user_item_matrix.shape[1]:,} products")

        except Exception as e:
            print(f"❌ Collaborative filtering preparation failed: {e}")
            self.user_item_matrix = pd.DataFrame()
            self.item_similarity_df = pd.DataFrame()

    def prepare_association_rules(self):
        """
        Prepare association rules for market basket analysis with optimized parameters
        """
        try:
            # Create basket data for association rules
            basket = self.df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack(fill_value=0)
            basket = basket.applymap(lambda x: 1 if x > 0 else 0)

            # Remove invoices with only one item (they don't generate rules)
            basket = basket[basket.sum(axis=1) > 1]

            if len(basket) > 0:
                # Generate frequent itemsets with optimized support
                min_support = max(0.005, 10 / len(basket))  # Dynamic support threshold
                frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)

                # Generate association rules with quality filtering
                if len(frequent_itemsets) > 0:
                    self.rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
                    self.rules = self.rules[self.rules['lift'] > 1.0]  # Only meaningful associations
                    self.rules = self.rules.sort_values('confidence', ascending=False)
                    print(f"✅ Association rules prepared: {len(self.rules):,} quality rules generated")
                else:
                    self.rules = None
                    print("⚠️ No frequent itemsets found for association rules")
            else:
                self.rules = None
                print("⚠️ Insufficient basket data for association rules")

        except Exception as e:
            print(f"❌ Association rules preparation failed: {e}")
            self.rules = None

    def get_customer_history(self, customer_id):
        """
        Get comprehensive purchase history for a customer
        """
        customer_data = self.df[self.df['CustomerID'] == customer_id]
        if customer_data.empty:
            return pd.DataFrame()

        recent_purchases = customer_data.sort_values('InvoiceDate', ascending=False).head(10)
        return recent_purchases[['InvoiceDate', 'StockCode', 'Description', 'Quantity', 'UnitPrice', 'TotalPrice']]

    def item_based_recommendations(self, customer_id, n_recommendations=10):
        """
        Generate item-based collaborative filtering recommendations with enhanced scoring
        """
        try:
            if self.user_item_matrix.empty or customer_id not in self.user_item_matrix.index:
                return []

            # Get customer's purchased items
            customer_items = self.user_item_matrix.loc[customer_id]
            purchased_items = customer_items[customer_items > 0].index.tolist()

            if not purchased_items:
                return []

            # Calculate similarity scores with weighted approach
            item_scores = {}
            for item in purchased_items:
                if item in self.item_similarity_df.columns:
                    similar_items = self.item_similarity_df[item].sort_values(ascending=False)

                    # Consider top 20 similar items for each purchased item
                    for similar_item, score in similar_items.iloc[1:21].items():  # Skip self-similarity
                        if similar_item not in purchased_items and score > 0.1:  # Minimum similarity threshold
                            # Weight by similarity score
                            item_scores[similar_item] = item_scores.get(similar_item, 0) + score

            # Get top recommendations
            recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]

            # Format results with enhanced information
            result = []
            for stock_code, score in recommendations:
                product_info = self.df[self.df['StockCode'] == stock_code].iloc[0]
                result.append({
                    'StockCode': stock_code,
                    'Description': product_info['Description'],
                    'Score': round(score, 3),
                    'Reason': f'Similar to items in your purchase history'
                })

            return result

        except Exception as e:
            print(f"❌ Error in item-based recommendations: {e}")
            return []

    def association_rule_recommendations(self, customer_id, n_recommendations=10):
        """
        Generate recommendations based on association rules with customer context
        """
        try:
            if self.rules is None or self.rules.empty:
                return []

            customer_history = self.get_customer_history(customer_id)
            if customer_history.empty:
                return []

            purchased_descriptions = set(customer_history['Description'].unique())

            recommendations = []
            seen_products = set()

            for _, rule in self.rules.iterrows():
                if len(recommendations) >= n_recommendations:
                    break

                antecedents = set(rule['antecedents'])
                consequents = set(rule['consequents'])

                # Check if customer purchased any of the antecedents
                if antecedents.intersection(purchased_descriptions):
                    for consequent in consequents:
                        if (consequent not in purchased_descriptions and
                                consequent not in seen_products and
                                len(recommendations) < n_recommendations):

                            # Find product info
                            product_match = self.df[self.df['Description'] == consequent]
                            if not product_match.empty:
                                product_info = product_match.iloc[0]
                                recommendations.append({
                                    'StockCode': product_info['StockCode'],
                                    'Description': consequent,
                                    'Score': round(rule['confidence'], 3),
                                    'Reason': f"Frequently bought with {', '.join(list(antecedents)[:2])}"
                                })
                                seen_products.add(consequent)

            return recommendations[:n_recommendations]

        except Exception as e:
            print(f"❌ Error in association rule recommendations: {e}")
            return []

    def get_popular_products(self, n_recommendations=10):
        """
        Get most popular products as fallback recommendations with enhanced metrics
        """
        try:
            # Calculate popularity based on both quantity and number of unique customers
            popular_products = self.df.groupby(['StockCode', 'Description']).agg({
                'Quantity': 'sum',
                'CustomerID': 'nunique',
                'TotalPrice': 'sum'
            }).reset_index()

            # Create a composite score (weighted by quantity and customer count)
            popular_products['PopularityScore'] = (
                    popular_products['Quantity'] * 0.7 +
                    popular_products['CustomerID'] * 0.3
            )

            popular_products = popular_products.sort_values('PopularityScore', ascending=False).head(n_recommendations)

            recommendations = []
            for _, product in popular_products.iterrows():
                recommendations.append({
                    'StockCode': product['StockCode'],
                    'Description': product['Description'],
                    'Score': int(product['Quantity']),
                    'Reason': f"Popular: {product['Quantity']:,} units sold to {product['CustomerID']} customers"
                })

            return recommendations

        except Exception as e:
            print(f"❌ Error in popular products: {e}")
            return []

    def get_recommendations(self, customer_id, n_recommendations=15):
        """
        Main method to get hybrid recommendations combining all approaches
        """
        try:
            recommendations = []
            seen_products = set()

            # Get item-based recommendations (40%)
            item_based = self.item_based_recommendations(customer_id, n_recommendations // 2)
            for rec in item_based:
                if rec['StockCode'] not in seen_products:
                    recommendations.append(rec)
                    seen_products.add(rec['StockCode'])

            # Get association rule recommendations (40%)
            assoc_based = self.association_rule_recommendations(customer_id, n_recommendations // 2)
            for rec in assoc_based:
                if rec['StockCode'] not in seen_products:
                    recommendations.append(rec)
                    seen_products.add(rec['StockCode'])

            # If not enough recommendations, add popular products (20%)
            if len(recommendations) < n_recommendations:
                needed = n_recommendations - len(recommendations)
                popular = self.get_popular_products(needed + 5)  # Get extra to account for duplicates
                for pop in popular:
                    if pop['StockCode'] not in seen_products and len(recommendations) < n_recommendations:
                        recommendations.append(pop)
                        seen_products.add(pop['StockCode'])

            return recommendations[:n_recommendations]

        except Exception as e:
            print(f"❌ Error in hybrid recommendations: {e}")
            return self.get_popular_products(n_recommendations)  # Fallback

    def get_customer_metrics(self, customer_id):
        """
        Get comprehensive customer metrics for dashboard display
        """
        customer_data = self.df[self.df['CustomerID'] == customer_id]
        if customer_data.empty:
            return {}

        # Calculate various customer metrics
        total_spent = customer_data['TotalPrice'].sum()
        total_orders = customer_data['InvoiceNo'].nunique()
        total_products = customer_data['StockCode'].nunique()

        # Average order value
        order_totals = customer_data.groupby('InvoiceNo')['TotalPrice'].sum()
        avg_order_value = order_totals.mean() if len(order_totals) > 0 else 0

        # Customer tenure (days between first and last purchase)
        if len(customer_data) > 1:
            tenure_days = (customer_data['InvoiceDate'].max() - customer_data['InvoiceDate'].min()).days
        else:
            tenure_days = 0

        return {
            'total_spent': total_spent,
            'total_orders': total_orders,
            'total_products': total_products,
            'avg_order_value': avg_order_value,
            'tenure_days': tenure_days,
            'first_purchase': customer_data['InvoiceDate'].min(),
            'last_purchase': customer_data['InvoiceDate'].max()
        }

    def get_engine_stats(self):
        """
        Get statistics about the recommendation engine
        """
        stats = {
            'total_customers': self.df['CustomerID'].nunique(),
            'total_products': self.df['StockCode'].nunique(),
            'total_transactions': self.df['InvoiceNo'].nunique(),
            'total_revenue': self.df['TotalPrice'].sum(),
            'data_period': f"{self.df['InvoiceDate'].min().strftime('%Y-%m-%d')} to {self.df['InvoiceDate'].max().strftime('%Y-%m-%d')}"
        }

        if self.user_item_matrix is not None:
            stats['collaborative_matrix'] = f"{self.user_item_matrix.shape[0]:,} × {self.user_item_matrix.shape[1]:,}"

        if self.rules is not None:
            stats['association_rules'] = len(self.rules)

        return stats