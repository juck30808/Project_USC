# 在TERMINAL輸入 python dashapp.py 執行
# http://127.0.0.1:8050/ 這個網址觀看儀錶板
# 關閉於TERMINAL按下Ctrl + C

##=====函式庫=====
#--基本類
import math                                         #用來取頁數無條件進位
import webbrowser                                   #幫忙開啟網頁
import os                                           #檢查WERKZEUG_RUN_MAIN是否更新 避免重複開啟網頁
import base64                                       #上傳檔案轉碼給dash用
import io                                           #上傳下載檔案類
#--Dash儀錶板類
import dash                                         #dash的函式庫
import dash_core_components as dcc                  #dash核心組件 跑圖表
import dash_html_components as html                 #dash的html元素組件 主要排版
from dash.dependencies import Input, Output , State #dash的callback元件
import dash_table                                   #dash的表格元件
#--資料表類
import pandas as pd                                 #pandas 資料表
import numpy as np                                  #numpy 資料表
#--圖表類
import plotly.express as px                         #圖表函式庫
import plotly.graph_objects as go                   #圖表物件函式庫
import folium                                       #地圖圖表
from folium import plugins                          #地圖圖表插件
from folium.plugins import HeatMap                  #地圖熱圖
#--預測模型類
from sklearn.preprocessing import StandardScaler    #數值標準化
from sklearn.neighbors import KNeighborsRegressor   #KNN模型
from sklearn.ensemble import RandomForestRegressor  #隨機森林模型
from sklearn.tree import DecisionTreeRegressor      #決策樹模型
from sklearn.svm import SVR                         #SVM回歸模型
from sklearn.neural_network import MLPRegressor     #MLP模型
from sklearn.preprocessing import PolynomialFeatures#多項式回歸模型
from sklearn import linear_model                    #線性模型
import joblib                                       #模型讀取與儲存

##=====讀取資料表格=====
all_data = pd.read_csv('data2.csv')                 #讀檔 全部資料
all_data = all_data.reset_index(drop=True)          #重新index編號
#按編號重複拿掉取第一個
all_data = all_data.iloc[all_data['編號'].drop_duplicates(keep='first', inplace=False).index].reset_index(drop=True)
address_data_rank = all_data.groupby(['縣市'])['單價元平方公尺'].mean().reset_index()
address_data_rank = address_data_rank.sort_values(by='單價元平方公尺',ascending=False).reset_index(drop=True)
result_data = pd.read_csv('result_data.csv')        #讀檔 預測結果對照檔案資料
#增加資料 與正確答案誤差值
result_data['Ans_diff'] = np.abs(result_data['Original_Ans'] - result_data['Regressor_Ans'])
#增加資料 誤差值百分比
result_data['Ans_rate'] = result_data['Ans_diff'] / result_data['Original_Ans']
bins = [-1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,4,10,100,500]
result_data['Ans_rate_bin'] = pd.cut(result_data['Ans_rate'], bins=bins).astype('string')
rate_data = result_data.groupby(['Type','Ans_rate_bin']).size().reset_index(name='count')
show_result_data = result_data                      #剛開始 展示資料 = 全部資料
##=====function=====
#--Dash
#建立勾選選項物件
def generate_clicklist(id_,list_data,default_data=[]):
    option_data = []
    for i in list_data:
        option_data.append({'label': i, 'value': i})
    return dcc.Checklist(id=id_,options=option_data,value=default_data,labelStyle={'display': 'inline-block'})
#建立核選選項物件
def generate_RadioItems(id_,list_data,default_data):
    option_data = []
    for i in list_data:
        option_data.append({'label': i, 'value': i})
    return dcc.RadioItems(id=id_,options=option_data,value=default_data,labelStyle={'display': 'inline-block'})
#建立群組長條圖表
def generate_barplot(data_,x_,y_,hub_,hub_order=[],title_="",legend_title_='',x_title_='',y_title_=''):
    if legend_title_ == '':
        legend_title_ = hub_
    if x_title_ == '':
        x_title_ = x_
    if y_title_ == '':
        y_title_ = y_
    if hub_order==[]:
        hub_order = list(data_[hub_].unique())
    go_bar = []
    for i in hub_order:
        x_data = data_.loc[data_[hub_]==i,x_]
        y_data = data_.loc[data_[hub_]==i,y_]
        go_bar.append(go.Bar(name=i, x=x_data, y=y_data))
    fig = go.Figure(data=go_bar)
    fig.update_layout(barmode='group',title=title_,height=550)
    fig.update_layout(legend_title_text = legend_title_)
    fig.update_xaxes(title_text=x_title_)
    fig.update_yaxes(title_text=y_title_)
    # fig = px.bar(data_, x="Type", y="count",color="Ans_rate_bin",barmode="group")
    return fig
#建立單獨長條圖表
def generate_singlebarplot(data_,x_,y_,title_="",legend_title_='',x_title_='',y_title_=''):
    if x_title_ == '':
        x_title_ = x_
    if y_title_ == '':
        y_title_ = y_
    d = data_[(data_['Ans_rate_bin']=='(-1.0, 0.0]')|(data_['Ans_rate_bin']=='(0.0, 0.1]')|(data_['Ans_rate_bin']=='(0.1, 0.2]')]
    d = d.groupby([x_])[y_].sum().reset_index()
    d['percent'] = [round(d.loc[d[x_]==i,y_].values[0] / (data_.loc[data_[x_]==i,y_].sum()) * 100,2) for i in data_[x_].unique()]
    # fig = px.bar(d, x=x_, y=y_ ,text='percent',height=550,title=title_)
    fig = go.Figure(data=[go.Bar(
            x=d[x_], y=d[y_],
            text=[f"{i}%" for i in d['percent'].values],
            textposition='auto'
        )])
    fig.update_layout(title=title_,height=550)
    fig.update_xaxes(title_text=x_title_)
    fig.update_yaxes(title_text=y_title_)
    return fig
#建立線圖
def generate_line_plot(data_,y1_,y2_,hub_,title_="",legend_title_='',y_title_=''):
    if legend_title_ == '':
        legend_title_ = hub_
    if y_title_=='':
        y_title_ = f"{y1_} : {y2_}"
    c = data_[hub_].unique().tolist()
    go_line = []
    color_ = px.colors.sequential.Rainbow
    color_index = len(color_) / (len(data_[hub_].unique())+1)
    j = 0
    d = data_.loc[data_[hub_]==c[0]]
    d = d.sort_values(by='Original_Ans').reset_index()
    go_line.append(go.Scatter(x=d.index,y=d[y1_],name=y1_,line=dict(color=color_[int(j)], width=4)))
    for i in c:
        j += color_index
        d = data_.loc[data_[hub_]==i]
        d = d.sort_values(by=['Original_Ans','Regressor_Ans']).reset_index()
        # go_line.append(go.Scatter(x=d.index,y=d[y1_],name=f"{i} {y1_}",line=dict(color=color_[int(j)], width=1)))
        go_line.append(go.Scatter(x=d.index,y=d[y2_],name=f"{i} {y2_}",line=dict(color=color_[int(j)], width=1,dash='dash')))
    fig = go.Figure(data=go_line)
    fig.update_layout(title=title_,height=550)
    fig.update_layout(legend_title_text = legend_title_)
    fig.update_layout(
        plot_bgcolor='#333333',
        paper_bgcolor='#333333',
        font_color='#7FDBFF'
    )
    fig.update_xaxes(title_text='index')
    fig.update_yaxes(title_text=y_title_)
    return fig
#建立圓餅圖
def generate_pie(data_,value_,name_,title_=""):
    if len(data_[name_].unique()) <= 1: #只有一個的時候
        d = data_[['Ans_rate_bin',value_]]
        d.columns = [name_,value_]
    else:
        d = data_[(data_['Ans_rate_bin']=='(-1.0, 0.0]')|(data_['Ans_rate_bin']=='(0.0, 0.1]')|(data_['Ans_rate_bin']=='(0.1, 0.2]')]
        d = d.groupby([name_])[value_].sum().reset_index()
    # fig = px.pie(d, values=value_, names=name_,title=title_,height=550)
    fig = go.Figure(data=[go.Pie(labels=d[name_], values=d[value_], textinfo='label+percent',insidetextorientation='radial')])
    fig.update_layout(title=title_,height=550)
    return fig
#建立表格
def generate_table(id_,data_,row_show=10,columns_sort=[],big_data=False): 
    columns_name = []
    if columns_sort != []:
        columns_name = [{"name": i, "id": i} for i in columns_sort]
        add_columns = list(set(data_.columns).difference(columns_sort))
        for i in add_columns:
            columns_name.append({"name": i, "id": i})
    else:
        columns_name = [{"name": i, "id": i} for i in data_.columns]
    return html.Div([dash_table.DataTable(
        id=id_,
        data=data_.iloc[0:row_show].to_dict('records'),
        columns=columns_name,
        page_current=0,
        page_count = math.ceil(data_.shape[0] / row_show),
        page_size=row_show,
        page_action='custom',
        virtualization=big_data,
        sort_action='custom',
        sort_mode='multi',
        sort_by=[]
    )])
#轉換上傳的csv檔為pandas資料表
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(e)
        return None
    return df

#--地圖圖表
def HeatMap_plot(data_):                                #地圖熱圖heatmap.html
    m=folium.Map([0,0])
    HeatMap(data_[['LAT','LONG']].dropna(),
            radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)
    sw = data_[['LAT', 'LONG']].min().values.tolist()
    ne = data_[['LAT', 'LONG']].max().values.tolist()
    m.fit_bounds([sw, ne]) 
    return m.get_root().render()
def utf2asc(s):                                         #轉碼
    return str(str(s).encode('ascii', 'xmlcharrefreplace'))[2:-1]
def Map_plot(data_,address_):                           #地圖詳細資訊
    d = data_.loc[all_data['縣市']==address_]
    if d.shape[0] > 10000: #超過一萬筆就隨機抽一萬筆資料
        d = d.sample(n = 10000, random_state=1)
    m=folium.Map(d[['LAT', 'LONG']].mean().values.tolist())
    mc=plugins.MarkerCluster().add_to(m)
    for _, row in d.iterrows():
        trade_date = f"民國{str(row['交易年月日'])[:(len(str(row['交易年月日']))-4)].lstrip('0')}年" \
                     f"{str(row['交易年月日'])[-4:-2].lstrip('0')}月" \
                     f"{str(row['交易年月日'])[-2:].lstrip('0')}日"
        pop_info = f"<font size=\"3\"><u>{row['土地區段位置建物區段門牌']}</u></font><br>" \
                    f"<table>" \
                    f"<tr><th>交易日期</th><td>{trade_date}</td>" \
                    f"<tr><th>種類</th><td>{row['交易標的']}</td>" \
                    f"<tr><th>建物型態</th><td>{row['建物型態']}</td>" \
                    f"<tr><th>使用分區</th><td>{row['都市土地使用分區']}</td>" \
                    f"<tr><th>土地面積</th><td>{round(row['土地移轉總面積平方公尺'],2)}</td>" \
                    f"<tr><th>建物面積</th><td>{round(row['建物移轉總面積平方公尺'],2)}</td>" \
                    f"<tr><th>車位面積</th><td>{round(row['車位移轉總面積平方公尺'],2)}</td>" \
                    f"<tr><th>樓層</th><td>{row['總樓層數']}</td>" \
                    f"<tr><th>建物現況格局</th><td><b>房:</b>{row['建物現況格局-房']} <b>廳:</b>{row['建物現況格局-廳']} <b>衛:</b>{row['建物現況格局-衛']} <b>隔間:</b>{row['建物現況格局-隔間']}</td>" \
                    f"<tr><th>管理組織</th><td>{row['有無管理組織']}</td>" \
                    f"<tr><th>屋齡</th><td>{row['屋齡']}</td>" \
                    f"<tr><th>總價格</th><td>{row['總價元']}</td>" \
                    f"<tr><th>每平方公尺單價：</th><td>{int(row['單價元平方公尺'])}</td>" \
                    f"</table>"
        folium.Marker(row[['LAT','LONG']], popup=folium.Popup(utf2asc(pop_info), max_width=800,min_width=200)).add_to(mc)
    sw = d[['LAT', 'LONG']].min().values.tolist()
    ne = d[['LAT', 'LONG']].max().values.tolist()
    m.fit_bounds([sw, ne]) 
    return m.get_root().render()

#--預測模型
def SomeLabelEncode_data(data_,Ori_data=False,Polynomial=False):         #含數值標準化轉換
    d = data_[list(data_.columns[list(data_.isnull().sum() == 0)])]
    if Ori_data:
        d = d.drop(['土地區段位置建物區段門牌','交易年月日','交易筆棟數','總價元',
                    '車位總價元','土地移轉面積平方公尺','建物移轉面積平方公尺',
                    '編號','from_data','使用分區或編定','LAT','LONG'], axis=1) #移除不可計算
        d['單價元平方公尺'] = d['單價元平方公尺'].astype('float')
    d[['土地移轉總面積平方公尺','屋齡','總樓層數','建物移轉總面積平方公尺','建物現況格局-房','建物現況格局-廳',
        '建物現況格局-衛','車位移轉總面積平方公尺']] = \
    d[['土地移轉總面積平方公尺','屋齡','總樓層數','建物移轉總面積平方公尺','建物現況格局-房','建物現況格局-廳',
        '建物現況格局-衛','車位移轉總面積平方公尺']].astype('float')

    change_colunms = ['土地移轉總面積平方公尺','總樓層數','建物移轉總面積平方公尺',
                    '建物現況格局-房','建物現況格局-廳','建物現況格局-衛',
                    '車位移轉總面積平方公尺']
    scaler = StandardScaler()
    scaler.fit(d[change_colunms])
    d[change_colunms] = scaler.transform(d[change_colunms])

    # 屋齡分群
    bins = [-2,0,5,10,25,50,75,100,100000]
    labels = ['＜1','1-5','6-10','11-25','26-50','51-75','76-100','＞100']
    d['屋齡分群'] = pd.cut(d['屋齡'], bins=bins, labels=labels)
    # 交易年距離分群
    bins = [-2,1,5,10,100000]
    labels = ['近1年','2-5年','6-10年','超過10年']
    d['交易年距離分群'] = pd.cut(d['交易年距離'], bins=bins, labels=labels)
    d = d.drop(['屋齡','交易年距離'],axis=1)
    if Polynomial:
        d_columns = ['交易標的','都市土地使用分區','建物型態','屋齡分群','交易年距離分群','縣市']
    else:
        d_columns = ['鄉鎮市區','交易標的','都市土地使用分區','建物型態','屋齡分群','交易年距離分群','縣市']
    data_encode = pd.get_dummies(d, columns=d_columns)
    columns = ['建物現況格局-隔間','有無管理組織']
    for col in columns:
        data_encode.loc[data_encode[col]=='無',col] = 0
        data_encode.loc[data_encode[col]=='有',col] = 1  
    return data_encode
def OneLabelEncode_data(data_,Ori_data=False):          #單獨數值dummies分割
    df = data_[list(data_.columns[list(data_.isnull().sum() == 0)])]
    if Ori_data:
        df = df.drop(['土地區段位置建物區段門牌','交易年月日','交易筆棟數','總價元',
                    '車位總價元','土地移轉面積平方公尺','建物移轉面積平方公尺',
                    '編號','from_data','使用分區或編定','LAT','LONG'], axis=1) #移除不可計算
        df['單價元平方公尺'] = df['單價元平方公尺'].astype('float')
    df[['土地移轉總面積平方公尺','屋齡','總樓層數','建物移轉總面積平方公尺','建物現況格局-房','建物現況格局-廳',
        '建物現況格局-衛','車位移轉總面積平方公尺']] = \
    df[['土地移轉總面積平方公尺','屋齡','總樓層數','建物移轉總面積平方公尺','建物現況格局-房','建物現況格局-廳',
        '建物現況格局-衛','車位移轉總面積平方公尺']].astype('float')
    # 屋齡分群
    bins = [-2,0,5,10,25,50,75,100,100000]
    labels = ['＜1','1-5','6-10','11-25','26-50','51-75','76-100','＞100']
    df['屋齡分群'] = pd.cut(df['屋齡'], bins=bins, labels=labels)
    # 交易年距離分群
    bins = [-2,1,5,10,100000]
    labels = ['近1年','2-5年','6-10年','超過10年']
    df['交易年距離分群'] = pd.cut(df['交易年距離'], bins=bins, labels=labels)
    df = df.drop(['屋齡','交易年距離'],axis=1)
    data_encode = pd.get_dummies(df, columns=['鄉鎮市區','交易標的','都市土地使用分區','建物型態','屋齡分群','交易年距離分群','縣市'])
    columns = ['建物現況格局-隔間','有無管理組織']
    for col in columns:
        data_encode.loc[data_encode[col]=='無',col] = 0
        data_encode.loc[data_encode[col]=='有',col] = 1
    return data_encode
try:    #KNN模型
    KNN_model=joblib.load('model/KNN.pkl')
except:
    KNN_model=None
def KNN_Regressor_use(df):
    global KNN_model
    data_encode = SomeLabelEncode_data(all_data,True)
    train = data_encode.drop(result_data.loc[result_data['Type']=='KNN','DataIndex'])
    train_X = train.drop(['單價元平方公尺'],axis=1)
    train_y = train['單價元平方公尺']
    if KNN_model == None:
        knn = KNeighborsRegressor(3, weights='distance')
        KNN_model = knn.fit(train_X, train_y)
        joblib.dump(KNN_model, 'model/KNN.pkl')
    dff = SomeLabelEncode_data(df)
    a_ = {}
    for i in list(train_X.columns):
        a_[i] = []
    dff = pd.concat([pd.DataFrame(a_),dff],axis=0).fillna(0)
    return KNN_model.predict(dff)
try:    #隨機森林模型
    RandomForest_model = joblib.load('model/RandomForest.pkl')
except:
    RandomForest_model=None
def RandomForest_Regressor_use(df):
    global RandomForest_model
    data_encode = OneLabelEncode_data(all_data,True)
    train = data_encode.drop(result_data.loc[result_data['Type']=='RandomTree','DataIndex'])
    train_X = train.drop(['單價元平方公尺'],axis=1)
    train_y = train['單價元平方公尺']
    if RandomForest_model == None:
        RandomTree = RandomForestRegressor()
        RandomForest_model = RandomTree.fit(train_X, train_y)
        joblib.dump(RandomForest_model, 'model/RandomForest.pkl')
    dff = OneLabelEncode_data(df)
    a_ = {}
    for i in list(train_X.columns):
        a_[i] = []
    dff = pd.concat([pd.DataFrame(a_),dff],axis=0).fillna(0)
    return RandomForest_model.predict(dff)
try:    #決策樹模型
    DecisionTree_model=joblib.load('model/DecisionTree.pkl')
except:
    DecisionTree_model=None
def DecisionTree_Regressor_use(df):
    global DecisionTree_model
    data_encode = OneLabelEncode_data(all_data,True)
    train = data_encode.drop(result_data.loc[result_data['Type']=='DecisionTree','DataIndex'])
    train_X = train.drop(['單價元平方公尺'],axis=1)
    train_y = train['單價元平方公尺']
    if DecisionTree_model == None:
        DecisionTree_model = DecisionTreeRegressor().fit(train_X, train_y)
        joblib.dump(DecisionTree_model, 'model/DecisionTree.pkl')
    dff = OneLabelEncode_data(df)
    a_ = {}
    for i in list(train_X.columns):
        a_[i] = []
    dff = pd.concat([pd.DataFrame(a_),dff],axis=0).fillna(0)
    return DecisionTree_model.predict(dff)
try:    #SVM回歸模型
    SVR_model = joblib.load('model/SVR.pkl')
except:
    SVR_model = None
def SVM_Regressor_use(df):
    global SVR_model
    data_encode = SomeLabelEncode_data(all_data,True)
    train = data_encode.drop(result_data.loc[result_data['Type']=='SVR','DataIndex'])
    train_X = train.drop(['單價元平方公尺'],axis=1)
    train_y = train['單價元平方公尺']
    if SVR_model == None:
        model = SVR()
        SVR_model = model.fit(train_X, train_y)
        joblib.dump(SVR_model, 'model/SVR.pkl')
    dff = SomeLabelEncode_data(df)
    a_ = {}
    for i in list(train_X.columns):
        a_[i] = []
    dff = pd.concat([pd.DataFrame(a_),dff],axis=0).fillna(0)
    return SVR_model.predict(dff)
try:    #MLP模型
    MLP_model = joblib.load('model/MLP.pkl')
except:
    MLP_model = None
def MLP_Regressor_use(df):
    global MLP_model
    data_encode = SomeLabelEncode_data(all_data,True)
    train = data_encode.drop(result_data.loc[result_data['Type']=='MLP','DataIndex'])
    train_X = train.drop(['單價元平方公尺'],axis=1)
    train_y = train['單價元平方公尺']
    if MLP_model == None:
        model = MLPRegressor()
        MLP_model = model.fit(train_X, train_y)
        joblib.dump(MLP_model, 'model/MLP.pkl')
    dff = SomeLabelEncode_data(df)
    a_ = {}
    for i in list(train_X.columns):
        a_[i] = []
    dff = pd.concat([pd.DataFrame(a_),dff],axis=0).fillna(0)
    return MLP_model.predict(dff)
try:    #Polynomial模型
    Polynomial_model = joblib.load('model/Polynomial.pkl')
except:
    Polynomial_model = None
def Polynomial_Regressor_use(df):
    global Polynomial_model
    data_encode = SomeLabelEncode_data(all_data.drop('鄉鎮市區',axis=1),True,Polynomial=True)
    train = data_encode.drop(result_data.loc[result_data['Type']=='Polynomial','DataIndex'])
    train_X = train.drop(['單價元平方公尺'],axis=1)
    train_y = train['單價元平方公尺']
    polyfeat = PolynomialFeatures(degree=2)
    if Polynomial_model == None:
        X_trainpoly = polyfeat.fit_transform(train_X)
        Polynomial_model = linear_model.Lasso(alpha=50000).fit(X_trainpoly, train_y)
        joblib.dump(Polynomial_model, 'model/Polynomial.pkl')
    dff = SomeLabelEncode_data(df.drop('鄉鎮市區',axis=1),Polynomial=True)
    a_ = {}
    for i in list(train_X.columns):
        a_[i] = []
    dff = pd.concat([pd.DataFrame(a_),dff],axis=0).fillna(0)
    X_testpoly = polyfeat.fit_transform(dff)
    return Polynomial_model.predict(X_testpoly)

##=====介面=====
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']   #使用css格式
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,suppress_callback_exceptions=True)    #調用dash儀錶板在app

app.layout = html.Div([ #===主要頁面
    dcc.Location(id='url', refresh=False),              #網址
    html.H4('實價登錄數據分析'),                         #大標題
    html.Button('Home', id='home_btn'),                 #首頁選項按鈕
    html.Button('Map', id='map_btn'),                   #地圖選項按鈕
    html.Button('Regressor Model', id='Regressor_btn'), #預測模型按鈕
    html.Button('Upload Regressor', id='Upload_btn'),   #上傳預測按鈕
    html.Button('Original Data', id='ori_btn'),         #原始資料按鈕
    html.Div(id='page-content')                         #主要頁面切換
])
@app.callback(Output('url', 'pathname'),            #Output改變網址
              Input('home_btn', 'n_clicks'),        #首頁按鈕
              Input('map_btn', 'n_clicks'),         #地圖按鈕
              Input('Regressor_btn', 'n_clicks'),   #預測模型按鈕
              Input('Upload_btn', 'n_clicks'),      #上傳預測按鈕
              Input('ori_btn', 'n_clicks')          #原始資料按紐
              )
def displayClick(btn1,btn2,btn3,btn4,btn5):
    #用來偵測最新的callback資料的prop_id狀態 也就是剛剛按下的按鈕id
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'map_btn' in changed_id:
        return '/map'
    elif 'Regressor_btn' in changed_id:
        return '/model'
    elif 'Upload_btn' in changed_id:
        return '/upload'
    elif 'ori_btn' in changed_id:
        return '/original'
    else:
        return '/'

index_page = html.Div([ #===首頁
    # Markdown
    dcc.Markdown('''
    ##### 實價登錄實作練習，對台灣房地的資訊來分析預測房價。
    * Map
    > 實價登錄的熱圖與詳細資訊。  
    左邊為實價登錄資料分布熱圖，右邊為地圖詳細資訊
    * Regressor Model
    > 實價登錄的各模型預測結果圖表，來看哪個模型預測情況和準確度。  
    使用 KNN,隨機森林,決策樹,SVM回歸,MLP類神經網路,Polynomial多項式回歸線圖 模型來預測
    * Upload Regressor
    > 可將csv資料上傳，並根據選擇的模型預測價格結果。
    * Original Data
    > 資料原始資料表格。
    * * *
    **作者：**林子敬、王采彤、陳世銓
    ''')
])

map_layout = html.Div([ #===地圖頁面
    html.H6('實價登錄地圖'),#次標題
    #建立縣市的核選選項
    generate_RadioItems('address', all_data['縣市'].unique().tolist(),'臺北市'),
    html.Div(id='address_data',style={'width': '100%','display': 'inline-block'}),
    #左邊的地圖熱圖
    html.Div([
        html.Iframe(id='heatmap', srcDoc = HeatMap_plot(all_data), width='100%' , height = '600')
    ],style={'width': '50%','display': 'inline-block'}),
    #右邊的地圖詳細資訊
    html.Div([
        dcc.Loading(id="map_loading",children=html.Iframe(id='map',width= '100%' , height='600'), type= "circle")
    ],style={'width': '50%','display': 'inline-block'})
])
#地圖詳細資訊 : 縣市核選選項 的callback
@app.callback(
    Output('map_loading','children'),   #讀取圖案
    Output('address_data','children'),  #縣市資訊
    Input('address', 'value')           #偵測輸入變動 id為address 的 value 資料
)
def update_plot_src(input_value):                                                               #更新地圖詳細資料的function
    html_str = Map_plot(all_data,input_value)                                                   #重新執行一次建立地圖html
    d = all_data.loc[all_data['縣市']==input_value]
    address_data = f"{input_value}單價元平方公尺： 平均：{d['單價元平方公尺'].mean():0.2f} " \
            f"最大：{d['單價元平方公尺'].max():0.2f} 最小：{d['單價元平方公尺'].min():0.2f} " \
            f"全縣市平均單價元排行：{address_data_rank.loc[address_data_rank['縣市']==input_value].index[0]+1}"
    return html.Iframe(id='map',srcDoc=html_str,width= '100%' , height='600') , address_data

model_layout = html.Div([ #===預測模型頁面
    html.H6('實價登錄預測模型'),#次標題
    #建立預測模型的勾選選項
    generate_clicklist('Regressor_Type', rate_data['Type'].unique().tolist(),[]),
    #建立每個預測模型準確度的資料
    html.Div([
        dcc.Graph(id='result_barplot')
    ],style={'width': '70%','display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='result_pie')
    ],style={'width': '30%','display': 'inline-block'}),
    dcc.Graph(id='result_line'),
    generate_table('result_table_page',show_result_data,20,columns_sort=['DataIndex','Address','Type','Original_Ans','Regressor_Ans','Ans_diff','Ans_rate'])
])
@app.callback(
    Output('result_barplot', 'figure'), #輸出更改 result_barplot 的 figure 資料
    Output('result_pie', 'figure'),     #輸出更改 result_pie 的 figure 資料
    Output('result_line', 'figure'),    #輸出更改 result_line 的 figure 資料
    Input('Regressor_Type', "value")    #偵測輸入變動 Regressor_Type 的 value 資料
)
def update_result_plot(value):
    if len(value) == 0: #空的就選全部
        value = rate_data['Type'].unique().tolist()
    #刷出符合項
    bool_ = rate_data['Type']==value[0]
    bool2_ = result_data['Type']==value[0]
    if len(value) > 1:
        for i in value[1:]:
            bool_ = (bool_) | (rate_data['Type']==i)
            bool2_ = (bool2_) | (result_data['Type']==i)
    #找出類別符合的
    d = rate_data.loc[bool_]
    d2 = result_data.loc[bool2_]
    #各模型群組直條圖建立
    fig = generate_barplot(d,'Type','count','Ans_rate_bin',
            ['(-1.0, 0.0]','(0.0, 0.1]','(0.1, 0.2]','(0.2, 0.3]','(0.3, 0.4]',
            '(0.4, 0.5]','(0.5, 0.6]','(0.6, 0.7]','(0.7, 0.8]','(0.8, 0.9]',
            '(0.9, 1.0]','(1.0, 2.0]','(2.0, 4.0]','(4.0, 10.0]','(10.0, 100.0]',
            '(100.0, 500.0]'],legend_title_='誤差百分比區間',title_='預測模型誤差比例數量')
    #判定是否只有選取一個模型 一個就選單模型各誤差區間圓餅圖 反之為選取模型20%誤差內比較直條圖
    if len(d['Type'].unique()) <= 1:
        fig2 = generate_pie(d,'count','Type',f"{d['Type'].unique().tolist()[0]} 各誤差比例")
    else:
        fig2 = generate_singlebarplot(d,'Type','count','誤差20%以下比例')
    #原始答案與選取模型的預測答案線圖
    fig3 = generate_line_plot(d2,'Original_Ans','Regressor_Ans','Type',"預測線圖","預測模型",'單價元平方公尺')
    return fig , fig2 , fig3

#預測模型詳細資料 : (預測模型勾選選項 & 預測模型詳細資料目前頁數 & 預測模型詳細資料排序) 的callback
@app.callback(
    Output('result_table_page', 'data'),        #表格的資料
    Output('result_table_page', 'page_count'),  #表格的總頁數
    Input('Regressor_Type', "value"),           #預測模型選項
    Input('result_table_page', "page_current"), #目前頁數
    Input('result_table_page', "page_size"),    #顯示幾列
    Input('result_table_page', "sort_by")       #表格排序情形
)
def update_table(value,page_current,page_size,sort_by):
    global show_result_data
    if len(value) == 0: #空的就選全部
        value = result_data['Type'].unique().tolist()
    bool_ = result_data['Type']==value[0]
    if len(value) > 1:
        for i in value[1:]:
            bool_ = (bool_) | (result_data['Type']==i)
    show_result_data = result_data.loc[bool_]
    page_count = math.ceil(show_result_data.shape[0] / page_size) #總頁數 為 全部資料數 ÷ 顯示列數 的無條件進位

    # 排序程式碼
    if len(sort_by):
        dff = show_result_data.sort_values(
            [col['column_id'] for col in sort_by],
            ascending=[
                col['direction'] == 'asc'
                for col in sort_by
            ],
            inplace=False
        )
    else:
        #沒有需要排序就是原本的資料
        dff = show_result_data
    # 回傳資料為 資料表裡位置在[ (目前頁數 × 顯示列數) : ((目前頁數+1) × 顯示列數) ]
    # 並以字典做成串列像是[{'表格1':[資料1],'表格1':[資料2]}]的形式傳回
    # 第二個傳回值為總頁數
    return dff.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records') , page_count

Upload_layout = html.Div([ #==上傳預測頁面
    html.H6('實價登錄上傳資料預測'),              #次標題
    html.Div([
        html.Div('csv檔 編碼UTF-8'),
        # 上傳檔案區
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                '拖移.csv 編碼需為UTF-8 ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        )],style={'width': '50%','display': 'inline-block'}),
    html.Div([ #模型按鈕區
        html.Button('KNN Model', id='KNN_Regressor_btn'), 
        html.Button('RandomForest Model', id='RandomForest_Regressor_btn'), 
        html.Button('DecisionTree Model', id='DecisionTree_Regressor_btn'),
        html.Button('SVR Model', id='SVM_Regressor_btn'),
        html.Button('MLP Model', id='MLP_Regressor_btn'),
        html.Button('Polynomial Model', id='Polynomial_Regressor_btn')
    ]),
    # dcc.Loading為執行時會跑讀取圖案，其中並更新使用模型
    dcc.Loading(id="loading",children=html.Div(id='use-model-name'), type= "circle"),
    # 輸出表格
    html.Div(id='output-data-upload',style={'width': '100%','display': 'inline-block'})
])
@app.callback(
    Output('upload-data', 'children'),  #回傳上傳檔案名稱
    Input('upload-data', 'filename')    #看是否上傳好檔案
)
def Just_new_Updata(filename):
    if filename == None:
        return html.Div([
            '拖移.csv 編碼需為UTF-8 ',
            html.A('Select Files')
        ])
    else:
        return html.Div(filename)

@app.callback(
    Output("loading", "children"),                      #讀取圖案
    Output('output-data-upload', 'children'),           #回傳預測完的表格
    Output('use-model-name','children'),                #回傳使用的模組
    Input('KNN_Regressor_btn', 'n_clicks'),             #KNN按鈕
    Input('RandomForest_Regressor_btn', 'n_clicks'),    #隨機森林按鈕
    Input('DecisionTree_Regressor_btn', 'n_clicks'),    #決策樹按鈕
    Input('SVM_Regressor_btn', 'n_clicks'),             #SVM回歸按鈕
    Input('MLP_Regressor_btn', 'n_clicks'),             #MLP回歸按鈕
    Input('Polynomial_Regressor_btn', 'n_clicks'),      #多項式回歸按鈕
    State('upload-data','contents'),                    #檢查上傳的csv檔案
    State('upload-data', 'filename')                    #檢查檔案名稱
)
def update_output(btn1,btn2,btn3,btn4,btn5,btn6,contents, names):
    if contents is not None:                                                    #如果沒有檔案不執行
        df = parse_contents(contents, names)                                    #轉換成pandas表格
        need_columns = [
            '鄉鎮市區','交易標的','土地移轉總面積平方公尺','都市土地使用分區','總樓層數','建物型態','建物移轉總面積平方公尺',
            '建物現況格局-房','建物現況格局-廳','建物現況格局-衛','建物現況格局-隔間','有無管理組織','車位移轉總面積平方公尺',
            '屋齡','交易年距離','縣市']
        if len(set(df.columns).difference(set(need_columns)))>=1:
            return html.Div(id='use-model-name') , [] , f"DATA COLUMNS ERROR! Only need \n{need_columns}"
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0] #看哪個按鈕
        if 'KNN_Regressor_btn' in changed_id:
            df['預測單價元平方公尺'] = KNN_Regressor_use(df)                     #使用KNN算出價格補上
        elif 'RandomForest_Regressor_btn' in changed_id:
            df['預測單價元平方公尺'] = RandomForest_Regressor_use(df)            #使用隨機森林
        elif 'DecisionTree_Regressor_btn' in changed_id:
            df['預測單價元平方公尺'] = DecisionTree_Regressor_use(df)            #使用決策樹
        elif 'SVM_Regressor_btn' in changed_id:
            df['預測單價元平方公尺'] = SVM_Regressor_use(df)                     #使用SVM回歸
        elif 'MLP_Regressor_btn' in changed_id:
            df['預測單價元平方公尺'] = MLP_Regressor_use(df)                     #使用MLP回歸
        elif 'Polynomial_Regressor_btn' in changed_id:
            df['預測單價元平方公尺'] = Polynomial_Regressor_use(df)              #使用多項式線圖
        #做成表格
        data_table = dash_table.DataTable(
            id='updata_result',
            data=df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in df.columns],
            virtualization=True,
            export_format="csv", #輸出csv檔案用
            page_action='none'
        )
        return html.Div(id='use-model-name') , data_table , changed_id[:changed_id.find('_btn')]
    else:
        return html.Div(id='use-model-name') , [] , 'None'

Original_layout = html.Div([ #===原始資料頁面
    html.H6('實價登錄原資料'),                              #次標題
    generate_table('ori_data',all_data,15,big_data=True)   #生成表格
])
@app.callback(
    Output('ori_data', 'data'),                 #表格的資料
    Input('ori_data', "page_current"),          #目前頁數
    Input('ori_data', "page_size"),             #顯示幾列
    Input('ori_data', "sort_by")                #表格排序情形
)
def update_ori_data_table(page_current,page_size,sort_by):
    # 排序程式碼
    if len(sort_by):
        dff = all_data.sort_values(
            [col['column_id'] for col in sort_by],
            ascending=[
                col['direction'] == 'asc'
                for col in sort_by
            ],
            inplace=False
        )
    else:
        #沒有需要排序就是原本的資料
        dff = all_data
    return dff.iloc[page_current*page_size:(page_current+ 1)*page_size].to_dict('records')

@app.callback(
    Output('page-content', 'children'), # 輸出頁面
    Input('url', 'pathname')            # 偵測網址變動
)
def display_page(pathname):
    if pathname == '/map':          #網址段為 /map 時，載入地圖頁面
        return map_layout
    elif pathname == '/model':      #網址段為 /model 時，載入預估模型頁面
        return model_layout
    elif pathname == '/upload':     #網址段為 /upload 時，載入上傳預估頁面
        return Upload_layout
    elif pathname == '/original':   #網址段為 /original 時，載入原始資料頁面
        return Original_layout
    else:                           #網址是其他狀況時，載入首頁
        return index_page

def main():
    port = 8050
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new("http://localhost:{}".format(port))
    app.run_server(debug=True,port=port)  #運作html網頁 Debug模式設False可以運作快點 但不會偵錯

if __name__ == '__main__':      #主要執行程式
    main()