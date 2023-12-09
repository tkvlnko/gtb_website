from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
import plotly.graph_objects as go
import os 


# import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
app = Flask(__name__)

matplotlib.use('Agg')


def op_pyear_worldwide(df): 
    x = np.array(np.arange(1970, 2017))
    y = df['iyear'].value_counts().sort_index(ascending=True)
    mean = df['iyear'].value_counts().mean()
    median = df['iyear'].median()

    plt.figure(figsize=(20,6))

    plt.plot(x, y)
    plt.axhline(y = mean, color = 'lightblue', linestyle = '--', label=f'mean: {round(mean)}') 
    plt.axhline(y = median, color = 'c', linestyle = '--', label=f'median: {round(median)}') 

    plt.xlabel("year")  
    plt.ylabel("number of operations") 
    plt.title("Number of terroristic operations per year (worldwide)") 
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper center') 
    return plt 

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
    if country not in top3.loc[:, 'country_txt'].tolist():
        top3 = pd.concat([top3, req], ignore_index=False)

    top3.index += 1   
    result = req.iloc[0]['operations']
    return result, top3.loc[:, ['operations', 'country_txt']].rename(columns={"country_txt": "country"})


@app.route('/', methods=['GET', 'POST'])
def homepage():
    df = pd.read_csv('./data/globalterrorism.csv', encoding='latin-1', low_memory=False)
    result, country, year = '--', '--', '-- '

    if request.method == 'POST':
        country = request.form.get('country')
        year = request.form.get('year')
        result, top3 = get_nattacks_by_country(df, country, year)
        top3.to_html('./templates/graphs/top3.html')


    return render_template('home.html', result=result, country=country, year=year)


@app.route('/sourcecode', methods=['POST', 'GET'])
def sourcecode():
    df_preprocess = pd.read_csv('./data/globalterrorism.csv', encoding='latin-1', low_memory=False)
    df2 = pd.read_csv('./data/globalterrorism.csv', encoding='latin-1', low_memory=False).drop(['summary', 'addnotes', 'scite1', 'scite2', 'scite3', 'target1', 'motive'], axis=1).head(6)
    df = pd.read_csv('./data/globalterrorism.csv', encoding='latin-1', low_memory=False).dropna(axis=1)

    shape = df_preprocess.shape
    
    # info 
    df_info(df)

    # info operations per country
    op_by_country_list(df)

    if os.path.isfile("./templates/graphs/plot1.png"):
        pass
    else:
        plot1 = op_pyear_worldwide(df)
        plot1.savefig('./templates/graphs/plot1.png')


    if os.path.isfile("./templates/graphs/plot2.html"):
        pass
    else:
        plot2 = op_by_country(df)
        plot2.write_html('./templates/graphs/plot2.html')

    
    if os.path.isfile("./templates/graphs/plot3.html"):
        pass
    else:
        plot3 = top_n_countries(df)
        plot3.write_html('./templates/graphs/plot3.html')


    if os.path.isfile('./templates/graphs/plot4.html'):
        pass
    else:
        plot4 = locations(df_preprocess)
        plot4.write_html('./templates/graphs/plot4.html')

    
    if os.path.isfile('./templates/graphs/plot5.html'):
        pass
    else:
        plot5 = big_sunburst(df_preprocess)
        plot5.write_html('./templates/graphs/plot5.html')

    if os.path.isfile('./templates/graphs/plot6.html'):
        pass
    else:
        plot6 = success_by_target_pie(df)
        plot6.write_html('./templates/graphs/plot6.html')

    if os.path.isfile('./templates/graphs/plot7.html'):
        pass
    else:
        plot7 = success_by_weap_pie(df)
        plot7.write_html('./templates/graphs/plot7.html')

    if os.path.isfile('./templates/graphs/plot8.html'):
        pass
    else:
        plot8 = success_on_target(df)
        plot8.write_html('./templates/graphs/plot8.html')

    if os.path.isfile('./templates/graphs/plot9.html'):
        pass
    else:
        plot9 = table_prpc(df_preprocess)
        open("./templates/graphs/plot9.html", "w").write(plot9)

    if os.path.isfile('./templates/graphs/plot10.html'):
        pass
    else:
        plot10 = n_victims(df_preprocess)
        open("./templates/graphs/plot10.html", "w").write(plot10)

    if os.path.isfile('./templates/graphs/plot11.html'):
        pass
    else:
        plot11 = correlation(df_preprocess)
        plot11.write_html('./templates/graphs/plot11.html')

    if os.path.isfile('./templates/graphs/plot12.html'):
        pass
    else:
        plot12 = success_vs_total(df_preprocess)
        plot12.write_html('./templates/graphs/plot12.html')




    return render_template('sourcecode.html', tables=[df2.to_html(classes='data')], 
                            titles=df2.columns.values,
                            shape=shape,
                            )

if __name__ == '__main__':
    app.run(debug=True)