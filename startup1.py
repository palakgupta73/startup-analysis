import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='wide',page_title='StartUp Analysis',page_icon='🥹')
df=pd.read_csv('startup_cleaned.csv')
df.date=pd.to_datetime(df.date,errors='coerce')

df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
def load_investor_details(investor):
    st.title(investor)
    #load recent investments of investors
    last_5=df[df['investors'].str.contains(investor)].head()[['date','startup','vertical','city','round','amount']]
    st.subheader('Most Recent Investments')
    st.dataframe(last_5)
    
    col1,col2=st.columns(2)
    with col1:
        big_series=df[df['investors'].str.contains(investor)].groupby('startup')['amount'].sum().sort_values(ascending=False).head()
        st.subheader('Biggest Investments')
        fig, ax= plt.subplots()
        ax.bar(big_series.index,big_series.values)
        st.pyplot(fig)

    with col2:
        ver_series=df[df['investors'].str.contains(investor)].groupby('vertical')['amount'].sum()

        st.subheader('Sectors Invested in')
        fig1,ax1=plt.subplots()
        ax1.pie(ver_series,labels=ver_series.index,autopct='%0.01f%% ')
        st.pyplot(fig1)

    col3,col4=st.columns(2)
    with col3:
        stage_series=df[df['investors'].str.contains(investor)].groupby('round')['amount'].sum()

        st.subheader('Stages Invested in')
        fig2,ax2=plt.subplots()
        ax2.pie(stage_series,labels=stage_series.index,autopct='%0.01f%% ')
        st.pyplot(fig2)
    with col4:
        
        city_series=df[df['investors'].str.contains(investor)].groupby('city')['amount'].sum()

        st.subheader('City Invested in')
        fig3,ax3=plt.subplots()
        ax3.pie(city_series,labels=city_series.index,autopct='%0.01f%% ')
        st.pyplot(fig3)

    
    yoy=df[df['investors'].str.contains(investor)].groupby('year')['amount'].sum()
    st.subheader('Yearly Investment')
    fig4,ax4=plt.subplots()
    ax4.plot(yoy.index,yoy.values)
    st.pyplot(fig4)


    st.subheader('Similar Investments by Other Investors')

    # Get the verticals 
    investor_verticals = df[df['investors'].str.contains(investor)]['vertical'].unique()
    investor_subverticals = df[df['investors'].str.contains(investor)]['subvertical'].unique() if 'subvertical' in df.columns else []

    # Get records from other investors in the same verticals
    similar_investments = df[
        (~df['investors'].str.contains(investor)) & 
        ((df['vertical'].isin(investor_verticals)) | 
         (df['subvertical'].isin(investor_subverticals) if 'subvertical' in df.columns else False))
    ]

    # Show similar investments
    if not similar_investments.empty:
        st.dataframe(similar_investments[['startup', 'vertical', 'subvertical', 'investors', 'city', 'round', 'amount']].drop_duplicates().reset_index(drop=True), use_container_width=True)
    else:
        st.info('No similar investments found by other investors.')


def load_startup_details(startup):
    st.title(startup)

    # Filter data for the selected startup
    st_df = df[df['startup'].str.lower() == startup.lower()]

    if st_df.empty:
        st.warning("No data found for the selected startup.")
        return
    
    # Display key details from the first record

    founder = st_df['founder'].dropna().iloc[0] if 'founder' in st_df.columns else "Not Available"
    vertical = st_df['vertical'].iloc[0]
    subvertical = st_df['subvertical'].iloc[0] if 'subvertical' in st_df.columns else "Not Available"
    city = st_df['city'].iloc[0]
    funding_round = st_df['round'].iloc[0]

    st.subheader("Basic Information")
    col1,col2,col3,col4=st.columns(4)
    with col1:
        st.markdown(f"<p style='font-size:14px; margin-bottom:2px;'>Founder(s)</p><p style='font-size:24px;'>{founder}</p>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<p style='font-size:14px; margin-bottom:2px;'>Industry</p><p style='font-size:24px;'>{vertical}</p>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"<p style='font-size:14px; margin-bottom:2px;'>Sub-industry</p><p style=' font-size:24px;'>{subvertical}</p>", unsafe_allow_html=True)

    with col4:
        st.markdown(f"<p style='font-size:14px; margin-bottom:2px;'>Location</p><p style='font-size:24px;'>{city}</p>", unsafe_allow_html=True)


    

    # Show total funding
    col5,col6=st.columns(2)
    with col5:
        total_funding = st_df['amount'].sum()
        st.metric('Total Funding Raised', str(round(total_funding)) + ' Cr')

    # Show number of rounds
    with col6:
        funding_rounds = st_df.shape[0]
        st.metric('Funding Rounds Count', funding_rounds)

    # Show recent funding history of the startup
    st.subheader("Funding Rounds Data")
    st.dataframe(st_df[['date', 'amount', 'round', 'investors']].reset_index(drop=True),use_container_width=True)

    

    # Show top investors
    all_investors = st_df['investors'].dropna().str.split(',')
    all_investors = all_investors.explode().str.strip()
    top_investors = all_investors.value_counts().head()

    st.subheader('Top Investors')
    fig, ax = plt.subplots()
    ax.bar(top_investors.index, top_investors.values)
    ax.set_ylabel('Number of Rounds Participated')
    ax.set_xticklabels(top_investors.index, rotation=45)
    st.pyplot(fig)

    # Show year-wise funding trend
    st.subheader('Year-wise Funding Trend')
    trend = st_df.groupby('year')['amount'].sum()
    fig2, ax2 = plt.subplots()
    ax2.plot(trend.index, trend.values, marker='o')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Funding Amount (Cr)')
    st.pyplot(fig2)

    # Find similar companies in the same vertical or subvertical
    st.subheader("Similar Companies")
    if 'subvertical' in st_df.columns:
        similar_df = df[(df['subvertical'] == subvertical) & (df['startup'].str.lower() != startup.lower())]
    else:
        similar_df = df[(df['vertical'] == vertical) & (df['startup'].str.lower() != startup.lower())]

    if similar_df.empty:
        st.info("No similar companies found in the same industry.")
    else:
        st.dataframe(similar_df[['startup', 'city', 'vertical', 'subvertical', 'round', 'amount']].drop_duplicates())



def load_overall_details():
    #st.title()
    col1,col2,col3,col4=st.columns(4)
    #total amount invetsed
    total=round(df['amount'].sum())
    with col1:
        st.metric('Total',str (total) + 'Cr')
    #Maximum amount infused in a startup
    max_amount=df.groupby('startup')['amount'].max().sort_values(ascending=False).head(1).values[0]
    with col2:
        st.metric('Maximum amount infused in a startup',max_amount)
    #avg ticket size
    avg_funding=round(df.groupby('startup')['amount'].sum().mean())
    with col3:
        st.metric('Average Ticket Size',str(avg_funding)+'Cr')
    #no. of startups funded
    count_of_startups=df['startup'].nunique()
    with col4:
        st.metric('Number of Startups Funded',count_of_startups)

    

    st.header('MoM Graph')
    #select_year=st.selectbox('Select Type',sorted(df['year'].unique()))
    select_option=st.selectbox('Select Type',['Total','Count'])
    
    if select_option=='Total':
        temp_df=df.groupby(['year','month'])['amount'].sum().reset_index()
        select_year=st.selectbox('Select Type',sorted(df['year'].unique()))
        #temp_df['x-axis']=temp_df[temp_df['year']==select_year]['month']
        temp_df1=temp_df.loc[temp_df['year'] ==select_year]
        fig4,ax4=plt.subplots()
        ax4.set_title(select_year)
        ax4.plot(temp_df1.month,temp_df1.amount)
        st.pyplot(fig4)

    else:
        temp_df=df.groupby(['year','month'])['amount'].count().reset_index()
        select_year=st.selectbox('Select Type',sorted(df['year'].unique()))
        
        temp_df1=temp_df.loc[temp_df['year'] ==select_year]
        fig4,ax4=plt.subplots()
        ax4.set_title(select_year)
        ax4.plot(temp_df1.month,temp_df1.amount)
        st.pyplot(fig4)
    
    
    temp_df['x-axis']=temp_df['month'].astype('str') + '-' + temp_df['year'].astype('str')
    fig5,ax5=plt.subplots()
    ax5.set_title('MoM graph for all the years')
    
    ax5.set_xticks(range(len(temp_df['amount'])))
    ax5.set_xticklabels(temp_df['x-axis'], rotation=90,fontsize=6)
    ax5.plot(temp_df['amount'])
    st.pyplot(fig5)
    

    st.subheader('📊 Sector Analysis')
    col5,col6=st.columns(2)
    with col5:
     # SECTOR ANALYSIS
        
        sector_count = df['vertical'].value_counts().head(10)
        fig1, ax1 = plt.subplots()
        ax1.pie(sector_count, labels=sector_count.index, autopct='%0.1f%%')
        ax1.set_title('Top Sectors (by Count)')
        st.pyplot(fig1)

    with col6:
        sector_sum = df.groupby('vertical')['amount'].sum().sort_values(ascending=False).head(10)
        fig2, ax2 = plt.subplots(figsize=(10, 6.5)) 

        # Plot vertical bars (x: sectors, y: total funding)
        ax2.bar(sector_sum.index, sector_sum.values, color='steelblue')
        ax2.set_title('Top Sectors (by Total Funding)')
        ax2.set_xlabel('Sector')
        ax2.set_ylabel('Total Funding (Cr)')
        plt.xticks(rotation=45, ha='right')  

        st.pyplot(fig2)


    # FUNDING TYPES
    st.subheader('💰 Types of Funding Rounds')
    round_series = df['round'].value_counts().head(10)
    fig3, ax3 = plt.subplots()
    ax3.bar(round_series.index, round_series.values)
    ax3.set_title('Top Funding Rounds')
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3)

    # CITY WISE FUNDING
    st.subheader('🏙️ City-wise Funding')
    city_series = df.groupby('city')['amount'].sum().sort_values(ascending=False).head(10)
    fig4, ax4 = plt.subplots()
    ax4.bar(city_series.index, city_series.values)
    ax4.set_title('Top Cities by Total Funding')
    ax4.tick_params(axis='x', rotation=45)
    st.pyplot(fig4)

    # TOP STARTUPS OVERALL - TABLE + BAR CHART
    st.subheader('🚀 Top Funded Startups (Overall)')
    top_startups = df.groupby('startup')['amount'].sum().sort_values(ascending=False).head(10)

    # Display as DataFrame
    #st.dataframe(top_startups.reset_index().rename(columns={'amount': 'Total Funding (Cr)'}), hide_index=True)

    # Display as Bar Chart
    fig_startup, ax_startup = plt.subplots(figsize=(10, 5))
    ax_startup.bar(top_startups.index, top_startups.values, color='skyblue')
    ax_startup.set_ylabel('Total Funding (Cr)')
    ax_startup.set_title('Top 10 Funded Startups')
    ax_startup.set_xticklabels(top_startups.index, rotation=45, ha='right')  # Rotate for better readability
    st.pyplot(fig_startup)

    # TOP STARTUPS YEAR-WISE
    # st.subheader('📅 Top Startups Year-wise')
    st.subheader("📌 Top 3 Startups by Funding (Year-wise)")

    # Dropdown to select year
    selected_year = st.selectbox("Select a Year", sorted(df['year'].dropna().unique()))

    # Filter and sort top 3 startups for the selected year
    top_startups = (
        df[df['year'] == selected_year]
        .groupby('startup')['amount']
        .sum()
        .sort_values(ascending=False)
        .head(3)
    )

    # Convert to DataFrame for display
    top_startups_df = top_startups.reset_index()
    top_startups_df.index = top_startups_df.index + 1  # Start index from 1

    # st.write(f"### Top 3 Startups in {selected_year}")
    # st.dataframe(top_startups_df, use_container_width=True)

    # Optional: Bar chart
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.bar(top_startups_df['startup'], top_startups_df['amount'], color='skyblue')
    ax.set_title(f"Top 3 Startups in {selected_year}")
    ax.set_ylabel("Funding Amount (Cr)")
    st.pyplot(fig)

    # TOP INVESTORS
    st.subheader('👤 Top Investors (Total Funding Across Startups)')

    from collections import Counter
    investor_list = df['investors'].dropna().apply(lambda x: [i.strip() for i in x.split(',')])
    investor_funding = {}

    for i, row in enumerate(investor_list):
        for investor in row:
            investor_funding[investor] = investor_funding.get(investor, 0) + df['amount'].iloc[i]

    # Convert to DataFrame and sort
    investor_df = pd.DataFrame(sorted(investor_funding.items(), key=lambda x: x[1], reverse=True), columns=['Investor', 'Total Funding']).head(10)

    # Display as table
    #st.dataframe(investor_df, hide_index=True)

    # Plot as vertical bar chart with labels
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(investor_df['Investor'], investor_df['Total Funding'], color='skyblue')
    ax.set_title('Top 10 Investors by Total Funding')
    ax.set_ylabel('Total Funding (in Cr)')
    ax.set_xlabel('Investor')
    plt.xticks(rotation=45, ha='right')

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    st.pyplot(fig)


    # FUNDING HEATMAP
    st.subheader('🔥 Funding Heatmap (Year vs City)')
    heatmap_data = df.pivot_table(index='city', columns='year', values='amount', aggfunc='sum', fill_value=0)
    st.dataframe(heatmap_data.style.background_gradient(cmap='YlGnBu'), use_container_width=True)


    





    
    

    

#st.dataframe(df)
st.sidebar.title('Indian Startup Funding Analysis')
option=st.sidebar.selectbox('Select one option',['--Select--','Overall Analysis','Startup','Investor'])
if option =='Overall Analysis':
    st.title('Overall Analysis')
    #selected_investor=st.sidebar.selectbox('Select Investor',sorted(set(df['investors'].str.split(',').sum())))
    #btn3=st.sidebar.button('Show Overall Analysis')
    #if btn3:
    load_overall_details()
    
elif option=='Startup':
    st.title('Startup details')
    selected_startup=st.sidebar.selectbox('Select startup',sorted(df['startup'].unique().tolist()))
    btn1=st.sidebar.button('Find startup details')
    if  btn1:
        load_startup_details(selected_startup)
elif option=='Investor':
    st.title('Investor details')
    selected_investor=st.sidebar.selectbox('Select Investor',sorted(set(df['investors'].str.split(',').sum())))
    btn2=st.sidebar.button('Find investor details')
    if btn2:
        load_investor_details(selected_investor)

else:
    st.write('Startup Analysis')
