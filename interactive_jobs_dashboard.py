import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Jobs Market Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_clean_data(file_path):
    """Load and clean the jobs dataset"""
    df = pd.read_csv(file_path)
    
    # Clean salary data - extract numeric values
    def extract_salary_range(salary_str):
        if pd.isna(salary_str):
            return None, None
        
        # Remove currency symbols and extract numbers
        numbers = re.findall(r'[\d,]+', str(salary_str))
        if len(numbers) >= 2:
            min_sal = float(numbers[0].replace(',', ''))
            max_sal = float(numbers[1].replace(',', ''))
            return min_sal, max_sal
        elif len(numbers) == 1:
            val = float(numbers[0].replace(',', ''))
            return val, val
        return None, None
    
    # Apply salary extraction
    salary_ranges = df['Salary_Range'].apply(extract_salary_range)
    df['Min_Salary'] = [x[0] for x in salary_ranges]
    df['Max_Salary'] = [x[1] for x in salary_ranges]
    df['Avg_Salary'] = df[['Min_Salary', 'Max_Salary']].mean(axis=1)
    
    # Clean skills data
    df['Skills_List'] = df['Skills_Required'].str.split(', ')
    
    # Convert posting date to datetime
    df['Posting_Date'] = pd.to_datetime(df['Posting_Date'])
    
    return df

def create_metrics_section(df):
    """Create key metrics section"""
    st.markdown('<h2 class="main-header">📊 Jobs Market Analysis Dashboard</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Job Postings",
            value=f"{len(df):,}",
            delta=f"{len(df)} jobs"
        )
    
    with col2:
        st.metric(
            label="Unique Companies",
            value=f"{df['Company'].nunique():,}",
            delta=f"{df['Company'].nunique()} companies"
        )
    
    with col3:
        st.metric(
            label="Industries",
            value=f"{df['Industry'].nunique():,}",
            delta=f"{df['Industry'].nunique()} sectors"
        )
    
    with col4:
        avg_salary = df['Avg_Salary'].mean()
        st.metric(
            label="Average Salary",
            value=f"{avg_salary:,.0f} EGP",
            delta=f"{avg_salary:,.0f} EGP"
        )

def create_industry_analysis(df):
    """Create industry analysis section"""
    st.markdown("## 🏭 Industry Analysis")
    
    # Top industries
    top_industries = df['Industry'].value_counts().head(10)
    
    # Create interactive industry analysis with Plotly
    # 1. Horizontal bar chart
    fig1 = px.bar(x=top_industries.values, y=top_industries.index, 
                  orientation='h',
                  title='Top 10 Industries by Job Count',
                  hover_data=[top_industries.values],
                  color=top_industries.values,
                  color_continuous_scale='Blues')
    
    fig1.update_layout(xaxis_title="Number of Job Postings", yaxis_title="Industry")
    fig1.update_traces(hovertemplate="<b>%{y}</b><br>Jobs: %{x}<extra></extra>")
    st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Pie chart
    fig2 = px.pie(values=top_industries.values, names=top_industries.index,
                   title='Industry Distribution',
                   hover_data=[top_industries.values])
    
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    fig2.update_layout(showlegend=True)
    st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Industry details table
    st.markdown("### 📊 Industry Details")
    industry_details = df.groupby('Industry').agg({
        'Avg_Salary': ['mean', 'median', 'count'],
        'Company': 'nunique',
        'Job_Type': lambda x: x.value_counts().index[0] if len(x) > 0 else 'N/A'
    }).round(2)
    
    industry_details.columns = ['Avg_Salary_Mean', 'Avg_Salary_Median', 'Job_Count', 'Company_Count', 'Most_Common_Job_Type']
    industry_details = industry_details.sort_values('Job_Count', ascending=False).head(10)
    
    # Format for display
    display_df = industry_details.reset_index()
    display_df['Avg_Salary_Mean'] = display_df['Avg_Salary_Mean'].apply(lambda x: f"{x:,.0f} EGP")
    display_df['Avg_Salary_Median'] = display_df['Avg_Salary_Median'].apply(lambda x: f"{x:,.0f} EGP")
    
    st.dataframe(display_df, use_container_width=True)

def create_salary_analysis(df):
    """Create salary analysis section"""
    st.markdown("## 💰 Salary Analysis")
    
    # Filter out extreme outliers
    salary_data = df[df['Avg_Salary'] < df['Avg_Salary'].quantile(0.95)]
    
    # Create interactive salary analysis with Plotly
    # 1. Salary by Experience Level
    exp_salary_data = salary_data.groupby('Experience_Level')['Avg_Salary'].agg(['mean', 'median', 'count']).reset_index()
    
    fig1 = px.box(salary_data, x='Experience_Level', y='Avg_Salary', 
                  title='Salary Distribution by Experience Level',
                  hover_data=['Company', 'Industry', 'Job_Type'],
                  color='Experience_Level',
                  color_discrete_sequence=px.colors.qualitative.Set3)
    fig1.update_layout(xaxis_title="Experience Level", yaxis_title="Average Salary (EGP)")
    fig1.update_xaxes(tickangle=45)
    st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Salary by Job Type
    fig2 = px.violin(salary_data, x='Job_Type', y='Avg_Salary',
                     title='Salary Distribution by Job Type',
                     hover_data=['Company', 'Industry', 'Experience_Level'],
                     color='Job_Type',
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig2.update_layout(xaxis_title="Job Type", yaxis_title="Average Salary (EGP)")
    fig2.update_xaxes(tickangle=45)
    st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Salary Distribution Histogram
    fig3 = px.histogram(salary_data, x='Avg_Salary', nbins=30,
                        title='Salary Distribution',
                        hover_data=['Company', 'Industry', 'Job_Type', 'Experience_Level'],
                        color_discrete_sequence=['steelblue'])
    fig3.add_vline(x=salary_data['Avg_Salary'].mean(), line_dash="dash", line_color="red",
                   annotation_text=f"Mean: {salary_data['Avg_Salary'].mean():.0f} EGP")
    fig3.add_vline(x=salary_data['Avg_Salary'].median(), line_dash="dash", line_color="green",
                   annotation_text=f"Median: {salary_data['Avg_Salary'].median():.0f} EGP")
    fig3.update_layout(xaxis_title="Average Salary (EGP)", yaxis_title="Number of Job Postings")
    st.plotly_chart(fig3, use_container_width=True)
    
    # 4. Company Size vs Salary (only if we have data)
    company_sizes = df['Company'].value_counts()
    df['Company_Size'] = df['Company'].map(company_sizes)
    plot_data = df[df['Avg_Salary'] < df['Avg_Salary'].quantile(0.95)]
    
    if len(plot_data) > 0:
        fig4 = px.scatter(plot_data, x='Company_Size', y='Avg_Salary',
                          title='Company Size vs Average Salary',
                          hover_data=['Company', 'Industry', 'Job_Type', 'Experience_Level'],
                          color='Industry',
                          size='Avg_Salary',
                          opacity=0.7)
        
        # Add trend line if we have enough data points
        if len(plot_data) > 1:
            z = np.polyfit(plot_data['Company_Size'], plot_data['Avg_Salary'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(plot_data['Company_Size'].min(), plot_data['Company_Size'].max(), 100)
            y_trend = p(x_trend)
            
            fig4.add_scatter(x=x_trend, y=y_trend, mode='lines', 
                           line=dict(color='red', dash='dash'),
                           name='Trend Line', showlegend=True)
        
        fig4.update_layout(xaxis_title="Company Size (Number of Job Postings)", 
                          yaxis_title="Average Salary (EGP)")
        st.plotly_chart(fig4, use_container_width=True)

def create_skills_analysis(df):
    """Create skills analysis section"""
    st.markdown("## 🛠️ Skills Analysis")
    
    # Extract and count all skills
    all_skills = []
    for skills_list in df['Skills_List'].dropna():
        all_skills.extend(skills_list)
    
    skill_counts = Counter(all_skills)
    top_skills = dict(skill_counts.most_common(15))
    
    # Create interactive skills analysis with Plotly
    # 1. Skills bar chart
    skills_df = pd.DataFrame(list(top_skills.items()), columns=['Skill', 'Count'])
    
    fig1 = px.bar(skills_df, x='Count', y='Skill', 
                  orientation='h',
                  title='Top 15 Most Required Skills',
                  color='Count',
                  color_continuous_scale='Reds')
    
    fig1.update_layout(xaxis_title="Number of Job Postings Requiring Skill", yaxis_title="Skill")
    fig1.update_traces(hovertemplate="<b>%{y}</b><br>Required in: %{x} jobs<extra></extra>")
    st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Skills co-occurrence heatmap (top 10 skills)
    top_10_skills = list(dict(skill_counts.most_common(10)).keys())
    cooccurrence_matrix = np.zeros((len(top_10_skills), len(top_10_skills)))
    
    for skills_list in df['Skills_List'].dropna():
        skills_in_job = [skill for skill in skills_list if skill in top_10_skills]
        for i, skill1 in enumerate(top_10_skills):
            for j, skill2 in enumerate(top_10_skills):
                if skill1 in skills_in_job and skill2 in skills_in_job:
                    cooccurrence_matrix[i][j] += 1
    
    # Create heatmap with Plotly
    fig2 = px.imshow(cooccurrence_matrix,
                     x=top_10_skills,
                     y=top_10_skills,
                     title='Skills Co-occurrence Heatmap (Top 10 Skills)',
                     color_continuous_scale='YlOrRd',
                     aspect="auto")
    
    fig2.update_layout(xaxis_title="Skills", yaxis_title="Skills")
    fig2.update_traces(hovertemplate="<b>%{x}</b> + <b>%{y}</b><br>Co-occurrence: %{z}<extra></extra>")
    st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Skills by Industry
    st.markdown("### 🏭 Skills by Industry")
    
    # Get top 5 industries and their most common skills
    top_industries = df['Industry'].value_counts().head(5).index
    
    industry_skills_data = []
    for industry in top_industries:
        industry_jobs = df[df['Industry'] == industry]
        industry_skills = []
        for skills_list in industry_jobs['Skills_List'].dropna():
            industry_skills.extend(skills_list)
        
        industry_skill_counts = Counter(industry_skills)
        top_industry_skills = dict(industry_skill_counts.most_common(5))
        
        for skill, count in top_industry_skills.items():
            industry_skills_data.append({
                'Industry': industry,
                'Skill': skill,
                'Count': count
            })
    
    if industry_skills_data:
        industry_skills_df = pd.DataFrame(industry_skills_data)
        fig3 = px.bar(industry_skills_df, x='Skill', y='Count', color='Industry',
                      title='Top Skills by Industry',
                      barmode='group')
        fig3.update_layout(xaxis_title="Skill", yaxis_title="Count")
        fig3.update_xaxes(tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)

def create_geographic_analysis(df):
    """Create geographic analysis section"""
    st.markdown("## 🌍 Geographic Analysis")
    
    top_locations = df['Location'].value_counts().head(15)
    
    # Create interactive geographic analysis with Plotly
    # 1. Location bar chart
    fig1 = px.bar(x=top_locations.values, y=top_locations.index, 
                  orientation='h',
                  title='Top 15 Cities/Regions by Job Postings',
                  hover_data=[top_locations.values],
                  color=top_locations.values,
                  color_continuous_scale='Greens')
    
    fig1.update_layout(xaxis_title="Number of Job Postings", yaxis_title="Location")
    fig1.update_traces(hovertemplate="<b>%{y}</b><br>Jobs: %{x}<extra></extra>")
    st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Location pie chart
    fig2 = px.pie(values=top_locations.values, names=top_locations.index,
                   title='Geographic Distribution',
                   hover_data=[top_locations.values])
    
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    fig2.update_layout(showlegend=True)
    st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Location details with salary information
    st.markdown("### 📊 Location Details")
    location_details = df.groupby('Location').agg({
        'Avg_Salary': ['mean', 'median', 'count'],
        'Company': 'nunique',
        'Industry': lambda x: x.value_counts().index[0] if len(x) > 0 else 'N/A'
    }).round(2)
    
    location_details.columns = ['Avg_Salary_Mean', 'Avg_Salary_Median', 'Job_Count', 'Company_Count', 'Most_Common_Industry']
    location_details = location_details.sort_values('Job_Count', ascending=False).head(15)
    
    # Format for display
    display_df = location_details.reset_index()
    display_df['Avg_Salary_Mean'] = display_df['Avg_Salary_Mean'].apply(lambda x: f"{x:,.0f} EGP")
    display_df['Avg_Salary_Median'] = display_df['Avg_Salary_Median'].apply(lambda x: f"{x:,.0f} EGP")
    
    st.dataframe(display_df, use_container_width=True)

def create_temporal_analysis(df):
    """Create temporal analysis section"""
    st.markdown("## 📅 Temporal Analysis")
    
    # Job posting timeline
    daily_jobs = df.groupby(df['Posting_Date'].dt.date).size()
    
    # Create interactive temporal analysis with Plotly
    # 1. Daily job postings timeline
    daily_jobs_df = daily_jobs.reset_index()
    daily_jobs_df.columns = ['Date', 'Job_Count']
    
    fig1 = px.line(daily_jobs_df, x='Date', y='Job_Count',
                   title='Job Posting Trends Over Time',
                   markers=True)
    
    fig1.update_layout(xaxis_title="Date", yaxis_title="Number of Job Postings")
    fig1.update_traces(hovertemplate="<b>%{x}</b><br>Jobs Posted: %{y}<extra></extra>")
    st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Monthly job postings
    monthly_jobs = df.groupby(df['Posting_Date'].dt.to_period('M')).size()
    monthly_jobs_df = monthly_jobs.reset_index()
    monthly_jobs_df.columns = ['Month', 'Job_Count']
    monthly_jobs_df['Month'] = monthly_jobs_df['Month'].astype(str)
    
    fig2 = px.bar(monthly_jobs_df, x='Month', y='Job_Count',
                  title='Monthly Job Posting Distribution',
                  color='Job_Count',
                  color_continuous_scale='Blues')
    
    fig2.update_layout(xaxis_title="Month", yaxis_title="Number of Job Postings")
    fig2.update_traces(hovertemplate="<b>%{x}</b><br>Jobs Posted: %{y}<extra></extra>")
    fig2.update_xaxes(tickangle=45)
    st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Job posting patterns by day of week
    df['DayOfWeek'] = df['Posting_Date'].dt.day_name()
    day_of_week_jobs = df['DayOfWeek'].value_counts()
    
    fig3 = px.bar(x=day_of_week_jobs.index, y=day_of_week_jobs.values,
                  title='Job Postings by Day of Week',
                  color=day_of_week_jobs.values,
                  color_continuous_scale='Viridis')
    
    fig3.update_layout(xaxis_title="Day of Week", yaxis_title="Number of Job Postings")
    fig3.update_traces(hovertemplate="<b>%{x}</b><br>Jobs Posted: %{y}<extra></extra>")
    st.plotly_chart(fig3, use_container_width=True)

def create_interactive_filters(df):
    """Create interactive filters in sidebar"""
    st.sidebar.markdown("## 🔍 Filters")
    
    # Industry filter
    industries = ['All'] + sorted(df['Industry'].unique().tolist())
    selected_industry = st.sidebar.selectbox("Select Industry", industries)
    
    # Job type filter
    job_types = ['All'] + sorted(df['Job_Type'].unique().tolist())
    selected_job_type = st.sidebar.selectbox("Select Job Type", job_types)
    
    # Experience level filter
    experience_levels = ['All'] + sorted(df['Experience_Level'].unique().tolist())
    selected_experience = st.sidebar.selectbox("Select Experience Level", experience_levels)
    
    # Salary range filter
    min_salary = df['Avg_Salary'].min()
    max_salary = df['Avg_Salary'].max()
    salary_range = st.sidebar.slider(
        "Salary Range (EGP)",
        min_value=float(min_salary),
        max_value=float(max_salary),
        value=(float(min_salary), float(max_salary))
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_industry != 'All':
        filtered_df = filtered_df[filtered_df['Industry'] == selected_industry]
    
    if selected_job_type != 'All':
        filtered_df = filtered_df[filtered_df['Job_Type'] == selected_job_type]
    
    if selected_experience != 'All':
        filtered_df = filtered_df[filtered_df['Experience_Level'] == selected_experience]
    
    filtered_df = filtered_df[
        (filtered_df['Avg_Salary'] >= salary_range[0]) & 
        (filtered_df['Avg_Salary'] <= salary_range[1])
    ]
    
    return filtered_df

def create_job_details_table(df):
    """Create job details table"""
    st.markdown("## 📋 Job Details")
    
    # Show filtered results
    if len(df) > 0:
        # Select columns to display
        display_columns = ['Title', 'Company', 'Location', 'Industry', 'Job_Type', 
                          'Experience_Level', 'Avg_Salary', 'Skills_Required']
        
        # Format the display dataframe
        display_df = df[display_columns].copy()
        display_df['Avg_Salary'] = display_df['Avg_Salary'].apply(lambda x: f"{x:,.0f} EGP")
        display_df = display_df.rename(columns={
            'Title': 'Job Title',
            'Avg_Salary': 'Average Salary',
            'Skills_Required': 'Required Skills'
        })
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name=f"filtered_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No jobs match the selected filters. Please adjust your criteria.")

def main():
    """Main function to run the dashboard"""
    # Load data
    try:
        df = load_and_clean_data('jobs_large_1000.csv')
    except FileNotFoundError:
        st.error("❌ jobs_large_1000.csv file not found! Please make sure the file is in the same directory.")
        return
    
    # Create sidebar filters
    filtered_df = create_interactive_filters(df)
    
    # Main dashboard
    create_metrics_section(filtered_df)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏭 Industry Analysis", 
        "💰 Salary Analysis", 
        "🛠️ Skills Analysis",
        "🌍 Geographic Analysis", 
        "📅 Temporal Analysis",
        "📋 Job Details"
    ])
    
    with tab1:
        create_industry_analysis(filtered_df)
    
    with tab2:
        create_salary_analysis(filtered_df)
    
    with tab3:
        create_skills_analysis(filtered_df)
    
    with tab4:
        create_geographic_analysis(filtered_df)
    
    with tab5:
        create_temporal_analysis(filtered_df)
    
    with tab6:
        create_job_details_table(filtered_df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📊 Jobs Market Analysis Dashboard | Created with Streamlit & Seaborn</p>
        <p>Data Source: jobs_large_1000.csv | Total Records: {}</p>
    </div>
    """.format(len(df)), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
