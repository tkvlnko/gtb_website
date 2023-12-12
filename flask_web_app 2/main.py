from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
import plotly.graph_objects as go
import os 

import plotly.express as px
import plotly.graph_objects as go
app = Flask(__name__)

matplotlib.use('Agg')


def op_pyear_worldwide(df):
    x = np.arange(1970, 2017)
    y = df['iyear'].value_counts().sort_index(ascending=True)
    mean = df['iyear'].value_counts().mean()
    median = df['iyear'].median()

    fig, ax = plt.subplots(figsize=(20,6))

    ax.plot(x, y)
    ax.axhline(y=mean, color='lightblue', linestyle='--', label=f'mean: {round(mean)}')
    ax.axhline(y=median, color='c', linestyle='--', label=f'median: {round(median)}')

    ax.set_xlabel("year")
    ax.set_ylabel("number of operations")
    ax.set_title("number of terroristic operations per year (worldwide)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper center')

    return fig

def table_prpc(df):
    top_provstates_per_country = df.groupby('country_txt')['provstate'].value_counts().groupby(level=0).nlargest(5).reset_index(level=1, drop=True).reset_index()
    top_provstates_per_country = pd.merge(top_provstates_per_country, df[['country_txt', 'provstate', 'region_txt', 'success']], on=['country_txt', 'provstate'], how='left')
    
    return top_provstates_per_country[-20:-13].to_html(classes='data', header="true")   

def op_by_country(df):
    traces = []
    countries = sorted(df['country_txt'].unique())

    mean_values = {country: np.mean(df[df['country_txt'] == country]['iyear'].value_counts()) for country in countries}


    for country in countries:
        trace = go.Scatter(x=df[df['country_txt'] == country]['iyear'].value_counts().sort_index(ascending=True).index,
                        y=df[df['country_txt'] == country]['iyear'].value_counts().sort_index(ascending=True).values,
                        mode='lines',
                        name=country,
                        visible=False)  
        traces.append(trace)

    traces[0]['visible'] = True

    buttons = []
    for i, country in enumerate(countries):
        button = dict(method='update',
                    label=country,
                    args=[{'visible': [i == j for j in range(len(countries))]},  # Set the visibility of traces
                            {'annotations': [{
                                'x': 0.5, 
                                'y': 1.15, 
                                'xref': 'paper', 
                                'yref': 'paper', 
                                'text': f'Mean ({country}): {mean_values[country]:.2f} attacks per 50 years', 
                                'showarrow': False, 
                                'font': {'size': 16}
                            }],
                            'yaxis': {'title': 'Number of Attacks'}}])  # Update annotations
        buttons.append(button)
        

    layout = dict(updatemenus=[dict(active=0, buttons=buttons)], xaxis=dict(range=[1970,2018]))
    fig = go.Figure(data=traces, layout=layout)
    return fig 

def op_by_country_list(df):
    PATH_TO_INFO = './static/src/info_country.txt'
    PATH_TO_INFO_HTML = './templates/graphs/info_country.html'

    countries_rate = dict()
    for country in sorted(df['country_txt'].unique()):
        countries_rate[country] = df[df['country_txt'] == country].shape[0]
    countries_rate_sorted = sorted(countries_rate.items(), key=lambda item: item[1], reverse=True)

    with open(PATH_TO_INFO, 'w') as f:
        for country in countries_rate_sorted[:10]:
            f.write(f"{str(country)}\n")

    contents = open(PATH_TO_INFO, "r")
    with open(PATH_TO_INFO_HTML, "w") as e:
        for lines in contents.readlines():
            e.write("<pre>" + lines + "</pre>\n")

def top_n_countries(df):
    N_TOP_COUNTRIES = 7
    color_palette = ["#224c6e", "#2a658f", "#3883a9", "#5aaed2", "#8ad5eb", "#b9e5f3", "#d4eff8", "#c8f4fc", "#b9f9fd", "#aafcff"]

    traces = []
    countries_rate = dict()
    for country in sorted(df['country_txt'].unique()):
        countries_rate[country] = df[df['country_txt'] == country].shape[0]
        
    countries_rate_sorted = sorted(countries_rate.items(), key=lambda item: item[1], reverse=True)

    for i, (country, n) in enumerate(countries_rate_sorted[:N_TOP_COUNTRIES]):
        trace = go.Scatter(x=df[df['country_txt'] == country]['iyear'].value_counts().sort_index(ascending=True).index,
                        y=df[df['country_txt'] == country]['iyear'].value_counts().sort_index(ascending=True).values,
                        mode='lines',
                        name=country, 
                        line=dict(color=color_palette[i % len(color_palette)]), 
                        )  
        traces.append(trace)

    layout = go.Layout(width=1000, height=500, plot_bgcolor='rgba(196, 199, 212, 0.05)')
    fig = go.Figure(data=traces, layout=layout)
    return fig

def df_info(df):
    PATH_TO_INFO = './static/src/info.txt'
    PATH_TO_INFO_HTML = './templates/graphs/info.html'
    _ = df.info(buf = open(PATH_TO_INFO, 'w'), verbose=True, show_counts=True)
    contents = open(PATH_TO_INFO, "r")
    with open(PATH_TO_INFO_HTML, "w") as e:
        for lines in contents.readlines():
            e.write("<pre>" + lines + "</pre>\n")

def locations(df):
    df['iyear'] = df['iyear'].astype(str)
    fig = px.scatter_geo(df, 
        lat='latitude', 
        lon='longitude', 
        hover_name='country_txt', 
        title='Attacks Around the World', 
        projection='natural earth',
        color='iyear',
        animation_frame='iyear',
        color_discrete_sequence=["#2a658f"],
        opacity=0.5,
        width=1000, height=800)
    return fig

def big_sunburst(df):
    top_provstates_per_country = df.groupby('country_txt')['provstate'].value_counts().groupby(level=0).nlargest(5).reset_index(level=1, drop=True).reset_index()
    top_provstates_per_country = pd.merge(top_provstates_per_country, df[['country_txt', 'provstate', 'region_txt', 'success']], on=['country_txt', 'provstate'], how='left')
    color_palette = ["#224c6e", "#2a658f", "#3883a9", "#5aaed2", "#8ad5eb", "#b9e5f3", "#d4eff8", "#c8f4fc", "#b9f9fd", "#aafcff"]


    fig = px.sunburst(top_provstates_per_country, 
                    path=["region_txt", "country_txt", "provstate"], 
                    values="success", 
                    color='region_txt',
                    color_discrete_map={
                            'Middle East & North Africa': color_palette[0],  
                            'South Asia': color_palette[1], 
                            'Sub-Saharan Africa': color_palette[2],  
                            'South America': color_palette[3],  
                            'Western Europe': color_palette[4],  
                            'Central America & Caribbean': color_palette[5],  
                            'Southeast Asia': color_palette[6],  
                            'Eastern Europe': color_palette[7],  
                            'North America': color_palette[8]
                            },
                    width=1000, 
                    height=800,
                    )
    return fig

def success_by_target_pie(df):
    color_palette = ["#224c6e", "#2a658f", "#3883a9", "#5aaed2", "#8ad5eb", "#b9e5f3", "#d4eff8", "#c8f4fc", "#b9f9fd", "#aafcff"]
    fig = px.sunburst(df, path=['targtype1_txt'],
                values='success',
                title='Success Rate Based on Target Type',
                color='targtype1_txt',
                color_discrete_sequence=color_palette,
                width=500, 
                height=500
                )
    fig.update_layout(paper_bgcolor='white')
    return fig

def success_by_weap_pie(df):
    color_palette = ["#224c6e", "#2a658f", "#3883a9", "#5aaed2", "#8ad5eb", "#b9e5f3", "#d4eff8", "#c8f4fc", "#b9f9fd", "#aafcff"]
    fig = px.sunburst(df, path=['weaptype1_txt'],
                values='success',
                title='Success Rate Based on Weapon Type',
                color='weaptype1_txt',
                color_discrete_sequence=color_palette,
                width=500, 
                height=500
                )
    fig.update_layout(paper_bgcolor='white')
    return fig

def success_on_target(df):
    df2 = df[['weaptype1_txt', 'targtype1_txt', 'success']].copy()
    df2 = df2[df2["weaptype1_txt"].str.contains("Unknown") == False] 
    df2['num'] = 1

    color_palette = ["#191970", "#7DF9FF", "#6082B6", "#ADD8E6", "#088F8F", "#95a5a6", "#bdc3c7", "#e5e8e8", "#dfe6e9", "#ced6d9", "#aeb6bf", "#8395a7"]
    fig = px.histogram(df2, x="targtype1_txt", 
                y=["num"], 
                color='weaptype1_txt', 
                color_discrete_sequence=color_palette,
                title="Success Rate Based on Target Type",
                width=1000, 
                height=600)
    return fig

def correlation(df):
    color_palette = ["#224c6e", "#2a658f", "#3883a9", "#5aaed2", "#8ad5eb"]

    df4 = df.sort_values(by=['nkill'], ascending=False)
    df4 = df4[['iyear', 'iday', 'imonth', 'country_txt', 'region_txt', 'nkill', 'nwound' ]].dropna()[20:]
    df4 = df4.sort_values(by=['nwound'], ascending=False)
    df4 = df4[['iyear', 'iday', 'imonth', 'country_txt', 'region_txt', 'nkill', 'nwound' ]][20:]

    fig = px.scatter(df4, x="nkill", y="nwound", 
                    color="nwound",
                    trendline="ols", 
                    color_continuous_scale=color_palette,
                    ) 

    fig.update_layout(
        title="Correlation between Number of Wounded and Number of Kills",
        yaxis_title="Number of Wounded",
        xaxis_title="Number of Kills",
        width=700, height=800
    )
    return fig

def n_victims(df):
    df = df.sort_values(by=['nkill'], ascending=False)
    df = df[['iyear', 'iday', 'imonth', 'country_txt', 'region_txt', 'nkill', 'nwound' ]].head(10)
    return df.to_html(classes='data', header="true")   

def success_vs_total(df):
    df2 = df[df["weaptype1_txt"].str.contains("Unknown") == False] 
    targets_rate = dict()
    targets_rate_success = dict()
    for target in sorted(df['targtype1_txt'].unique()):
        targets_rate[target] = df[df['targtype1_txt'] == target].shape[0]
        targets_rate_success[target] = df[(df['targtype1_txt'] == target) & (df['success'] == 1)].shape[0]

        
    targets_rate_sorted = sorted(targets_rate.items(), key=lambda item: item[1], reverse=True)
    targets_rate_success_sorted = sorted(targets_rate_success.items(), key=lambda item: item[1], reverse=True)

    success = list(dict(targets_rate_success_sorted).values())[:10]
    total = list(dict(targets_rate_sorted).values())[:10]

    fig = px.bar(df2, 
                x=list(dict(targets_rate_sorted).keys())[:10], 
                y=[success, total], 
                barmode="group", 
                color_discrete_sequence=["#ADD8E6", "#191970"],
                title="Success Rate Based on Target Type", height=800, width=800)
    return fig

def get_nattacks_by_country(df, country, year):
    df2 = df[(df["iyear"] == int(year))]
    df2 = df2.groupby('country_txt').size().reset_index(name='operations').sort_values(by='operations', ascending=False).reset_index()
    top3 = df2.head(3)

    req = df2.loc[df2["country_txt"] == country]

    if top3.empty:
        top3 = req.copy()
    elif country not in top3.loc[:, 'country_txt'].tolist():
        top3 = pd.concat([top3, req], ignore_index=True)

    top3.index += 1   
    result = req.iloc[0]['operations']
    return result, top3.loc[:, ['operations', 'country_txt']].rename(columns={"country_txt": "country"})

def read_csv_optimized(file_path):
    dtypes = {'eventid': 'int32', 'iyear': 'int16', 'imonth': 'int8', 'iday': 'int8', 'extended': 'int8', 'country': 'int16', 'region': 'int8', 'latitude': 'float32', 'longitude': 'float32', 'success': 'int8'}
    chunk_size = 1000
    chunks = pd.read_csv(file_path, encoding='latin-1', low_memory=False, dtype=dtypes, chunksize=chunk_size)
    df = pd.concat(chunks)
    return df

def preprocess_data(df):
    df.drop(['summary', 'addnotes', 'scite1', 'scite2', 'scite3', 'target1', 'motive'], axis=1, inplace=True)
    df = df.head(6)
    return df

@app.route('/', methods=['GET', 'POST'])
def homepage():
    df = read_csv_optimized('./data/globalterrorism.csv')
    result, country, year = '--', '--', '--'

    if request.method == 'POST':
        country = request.form.get('country')
        year = request.form.get('year')
        result, top3 = get_nattacks_by_country(df, country, year)
        top3.to_html('./templates/graphs/top3.html')

    return render_template('home.html', result=result, country=country, year=year)


@app.route('/sourcecode', methods=['POST', 'GET'])
def sourcecode():
    df = read_csv_optimized('./data/globalterrorism.csv')
    df = preprocess_data(df)
    shape = df.shape

    # info 
    df_info(df)

    # info operations per country
    op_by_country_list(df)

    plots = [
        (op_pyear_worldwide, 'plot1.png'),
        (op_by_country, 'plot2.html'),
        (top_n_countries, 'plot3.html'),
        (locations, 'plot4.html'),
        (big_sunburst, 'plot5.html'),
        (success_by_target_pie, 'plot6.html'),
        (success_by_weap_pie, 'plot7.html'),
        (success_on_target, 'plot8.html'),
        (table_prpc, 'plot9.html'),
        (n_victims, 'plot10.html'),
        (correlation, 'plot11.html'),
        (success_vs_total, 'plot12.html')
    ]

    for plotfunc, filename in plots:
        if not os.path.isfile(f"./templates/graphs/{filename}"):
            if filename.endswith('.png'):
                plot = plotfunc(df)
                plot.savefig(f'./templates/graphs/{filename}')
            else:
                plot = plotfunc(df)
                plot.writehtml(f'./templates/graphs/{filename}')

    return render_template('sourcecode.html', tables=[df.to_html(classes='data')], 
                            titles=df.columns.values,
                            shape=shape,
                            )

if __name__ == '__main__':
    app.run(host='0.0.0.0')
