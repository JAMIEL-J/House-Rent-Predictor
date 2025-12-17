import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from predict import predict_rent

# Page config
st.set_page_config(
    page_title="House Rent Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        color: #28a745;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/House_Rent_Dataset.csv")
    df['Rent_per_sqft'] = df['Rent'] / df['Size']
    return df


data = load_data()

# Title
st.markdown('<h1 class="main-header">🏠 House Rent Predictor & Analysis Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🔮 Rent Predictor", "📊 Analysis Dashboard", "📈 Advanced Insights"])

# ============================================
# RENT PREDICTOR PAGE
# ============================================
if page == "🔮 Rent Predictor":
    st.header("🔮 Predict Your House Rent")
    st.write("Enter the property details below to get an estimated rent prediction.")
    
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    with col_center:
        with st.form("rent_form", clear_on_submit=False):
            st.subheader("📝 Property Details")
            
            col1, col2 = st.columns(2)
            with col1:
                bhk = st.number_input("🛏️ Number of BHK", min_value=1, max_value=10, value=2)
            with col2:
                size = st.number_input("📐 Size (sq ft)", min_value=100, max_value=10000, value=1000)
            
            col3, col4 = st.columns(2)
            with col3:
                bathroom = st.number_input("🚿 Number of Bathrooms", min_value=1, max_value=10, value=2)
            with col4:
                area_type = st.selectbox("📍 Area Type", options=["Super Area", "Carpet Area", "Built Area"])
            
            col5, col6 = st.columns(2)
            with col5:
                city = st.selectbox("🏙️ City", options=["Mumbai", "Chennai", "Bangalore", "Hyderabad", "Delhi", "Kolkata"])
            with col6:
                furnishing_status = st.selectbox("🛋️ Furnishing Status", options=["Unfurnished", "Semi-Furnished", "Furnished"])
            
            tenant_preferred = st.selectbox("👥 Tenant Preferred", options=["Bachelors", "Bachelors/Family", "Family"])
            
            submitted = st.form_submit_button("🎯 Predict Rent", use_container_width=True)
        
        if submitted:
            with st.spinner("Calculating..."):
                try:
                    predicted_rent = predict_rent(
                        bhk=bhk, size=size, bathroom=bathroom,
                        area_type=area_type, city=city,
                        furnishing_status=furnishing_status,
                        tenant_preferred=tenant_preferred
                    )
                    
                    st.markdown(f'<div class="prediction-result">💰 Predicted Monthly Rent: ₹{predicted_rent:,}</div>', unsafe_allow_html=True)
                    
                    st.subheader("📊 Quick Insights")
                    city_avg = data[data['City'] == city]['Rent'].mean()
                    city_median = data[data['City'] == city]['Rent'].median()
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric(f"City Average ({city})", f"₹{int(city_avg):,}", f"{((predicted_rent - city_avg) / city_avg) * 100:+.1f}%")
                    c2.metric(f"City Median ({city})", f"₹{int(city_median):,}")
                    c3.metric("Rent per Sq Ft", f"₹{predicted_rent / size:.2f}")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# ============================================
# ANALYSIS DASHBOARD PAGE
# ============================================
elif page == "📊 Analysis Dashboard":
    st.header("📊 Rent Analysis Dashboard")
    
    # Key Statistics
    st.subheader("📌 Key Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Mean Rent", f"₹{int(data['Rent'].mean()):,}")
    c2.metric("📈 Median Rent", f"₹{int(data['Rent'].median()):,}")
    c3.metric("🔺 Highest Rent", f"₹{int(data['Rent'].max()):,}")
    c4.metric("🔻 Lowest Rent", f"₹{int(data['Rent'].min()):,}")
    
    st.divider()
    
    # City Analysis
    st.subheader("🏙️ Rent Distribution by City")
    col1, col2 = st.columns(2)
    
    with col1:
        city_avg = data.groupby('City')['Rent'].mean().sort_values(ascending=False)
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=city_avg.index.tolist(),
            y=city_avg.values.tolist(),
            marker_color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'],
            text=[f"₹{int(v):,}" for v in city_avg.values],
            textposition='outside'
        ))
        fig1.update_layout(
            title="Average Rent by City",
            xaxis_title="City",
            yaxis_title="Average Rent (₹)",
            yaxis=dict(range=[0, city_avg.max() * 1.2]),
            height=450
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = go.Figure()
        cities = data['City'].unique()
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
        for i, city in enumerate(cities):
            city_data = data[data['City'] == city]['Rent']
            fig2.add_trace(go.Box(y=city_data.tolist(), name=city, marker_color=colors[i % len(colors)]))
        fig2.update_layout(
            title="Rent Distribution by City",
            yaxis_title="Rent (₹)",
            height=450,
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.divider()
    
    # BHK Analysis
    st.subheader("🛏️ Rent by BHK Configuration")
    col3, col4 = st.columns(2)
    
    with col3:
        bhk_avg = data.groupby('BHK')['Rent'].mean()
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=bhk_avg.index.tolist(),
            y=bhk_avg.values.tolist(),
            marker=dict(color=bhk_avg.values.tolist(), colorscale='Viridis'),
            text=[f"₹{int(v):,}" for v in bhk_avg.values],
            textposition='outside'
        ))
        fig3.update_layout(
            title="Average Rent by BHK",
            xaxis_title="BHK",
            yaxis_title="Average Rent (₹)",
            yaxis=dict(range=[0, bhk_avg.max() * 1.2]),
            height=450
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        sample = data.sample(min(500, len(data)), random_state=42)
        fig4 = go.Figure()
        for bhk_val in sorted(sample['BHK'].unique()):
            bhk_data = sample[sample['BHK'] == bhk_val]
            fig4.add_trace(go.Scatter(
                x=bhk_data['Size'].tolist(),
                y=bhk_data['Rent'].tolist(),
                mode='markers',
                name=f'{bhk_val} BHK',
                opacity=0.7
            ))
        fig4.update_layout(
            title="Size vs Rent (by BHK)",
            xaxis_title="Size (sq ft)",
            yaxis_title="Rent (₹)",
            height=450
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    st.divider()
    
    # Furnishing Status
    st.subheader("🛋️ Rent by Furnishing Status")
    furnish_data = data.groupby(['City', 'Furnishing Status'])['Rent'].mean().reset_index()
    
    fig5 = go.Figure()
    colors = {'Furnished': '#ff6b6b', 'Semi-Furnished': '#4ecdc4', 'Unfurnished': '#45b7d1'}
    for status in ['Furnished', 'Semi-Furnished', 'Unfurnished']:
        status_data = furnish_data[furnish_data['Furnishing Status'] == status]
        fig5.add_trace(go.Bar(
            x=status_data['City'].tolist(),
            y=status_data['Rent'].tolist(),
            name=status,
            marker_color=colors[status]
        ))
    fig5.update_layout(
        title="Average Rent by City and Furnishing Status",
        xaxis_title="City",
        yaxis_title="Average Rent (₹)",
        barmode='group',
        height=450
    )
    st.plotly_chart(fig5, use_container_width=True)

# ============================================
# ADVANCED INSIGHTS PAGE
# ============================================
elif page == "📈 Advanced Insights":
    st.header("📈 Advanced Market Insights")
    
    # Filters
    st.sidebar.subheader("🔧 Filters")
    selected_cities = st.sidebar.multiselect(
        "Select Cities",
        options=data['City'].unique().tolist(),
        default=data['City'].unique().tolist()
    )
    
    bhk_range = st.sidebar.slider(
        "BHK Range",
        min_value=int(data['BHK'].min()),
        max_value=int(data['BHK'].max()),
        value=(int(data['BHK'].min()), int(data['BHK'].max()))
    )
    
    filtered = data[(data['City'].isin(selected_cities)) & (data['BHK'].between(bhk_range[0], bhk_range[1]))].copy()
    
    st.info(f"**Showing {len(filtered):,} properties** out of {len(data):,} total")
    
    # Rent per sqft
    st.subheader("💰 Rent per Square Foot Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        sqft_avg = filtered.groupby('City')['Rent_per_sqft'].mean().sort_values(ascending=False)
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(
            x=sqft_avg.index.tolist(),
            y=sqft_avg.values.tolist(),
            marker=dict(color=sqft_avg.values.tolist(), colorscale='RdYlGn_r'),
            text=[f"₹{v:.1f}" for v in sqft_avg.values],
            textposition='outside'
        ))
        fig6.update_layout(
            title="Average Rent per Sq Ft by City",
            xaxis_title="City",
            yaxis_title="Rent per Sq Ft (₹)",
            yaxis=dict(range=[0, sqft_avg.max() * 1.2]),
            height=400
        )
        st.plotly_chart(fig6, use_container_width=True)
    
    with col2:
        fig7 = go.Figure()
        for city in filtered['City'].unique():
            city_sqft = filtered[filtered['City'] == city]['Rent_per_sqft']
            fig7.add_trace(go.Violin(y=city_sqft.tolist(), name=city, box_visible=True))
        fig7.update_layout(
            title="Rent per Sq Ft Distribution",
            yaxis_title="Rent per Sq Ft (₹)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig7, use_container_width=True)
    
    st.divider()
    
    # Tenant Analysis
    st.subheader("👥 Tenant Preference Analysis")
    col3, col4 = st.columns(2)
    
    with col3:
        tenant_counts = filtered.groupby('Tenant Preferred').size()
        fig8 = go.Figure()
        fig8.add_trace(go.Pie(
            labels=tenant_counts.index.tolist(),
            values=tenant_counts.values.tolist(),
            marker_colors=['#667eea', '#764ba2', '#f093fb'],
            textinfo='percent+label'
        ))
        fig8.update_layout(title="Properties by Tenant Type", height=400)
        st.plotly_chart(fig8, use_container_width=True)
    
    with col4:
        tenant_rent = filtered.groupby(['City', 'Tenant Preferred'])['Rent'].mean().reset_index()
        fig9 = go.Figure()
        tenant_colors = {'Bachelors': '#667eea', 'Bachelors/Family': '#764ba2', 'Family': '#f093fb'}
        for tenant in ['Bachelors', 'Bachelors/Family', 'Family']:
            t_data = tenant_rent[tenant_rent['Tenant Preferred'] == tenant]
            fig9.add_trace(go.Bar(
                x=t_data['City'].tolist(),
                y=t_data['Rent'].tolist(),
                name=tenant,
                marker_color=tenant_colors[tenant]
            ))
        fig9.update_layout(
            title="Average Rent by City and Tenant Type",
            xaxis_title="City",
            yaxis_title="Average Rent (₹)",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig9, use_container_width=True)
    
    st.divider()
    
    # Correlation Heatmap
    st.subheader("🔗 Feature Correlations")
    numeric_cols = ['BHK', 'Rent', 'Size', 'Bathroom', 'Rent_per_sqft']
    corr_matrix = filtered[numeric_cols].corr()
    
    fig10 = go.Figure()
    fig10.add_trace(go.Heatmap(
        z=corr_matrix.values.tolist(),
        x=numeric_cols,
        y=numeric_cols,
        colorscale='RdBu',
        text=[[f"{val:.2f}" for val in row] for row in corr_matrix.values],
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    fig10.update_layout(
        title="Correlation Matrix",
        height=450
    )
    st.plotly_chart(fig10, use_container_width=True)
    
    st.divider()
    
    # Data Preview
    st.subheader("📋 Data Preview")
    st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)

# Footer
st.sidebar.divider()
st.sidebar.markdown("**🏠 Smart Rent Predictor**")
st.sidebar.markdown("Built with ❤️ using Streamlit")
