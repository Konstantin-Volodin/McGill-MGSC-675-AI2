#%%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

#%% Train vs Valid Loss
models = {'resnet18': ('ResNet18', 'rgb(27,158,119)', 0.2), 
          'resnet34': ('ResNet34', 'rgb(217,95,2)', 0.2), 
          'effnets': ('EfficientNetS', 'rgb(117,112,179)', 1),
          'effnetm': ('EfficientNetM', 'rgb(231,41,138)', 0.2)}

fig1 = go.Figure()

for k,v in models.items():
    df_tl = pd.read_csv(f'results/{k}_tl.csv', names=['time', 'step', 'Training Loss'], skiprows=1)
    df_tl['Model'] = v[0]
    df_vl = pd.read_csv(f'results/{k}_vl.csv', names=['time', 'step', 'Validation Loss'], skiprows=1)
    df_vl['Model'] = v[0]

    fig1.add_trace(go.Scatter(x=df_vl.step, y=df_vl['Validation Loss'],
                                line={'color':v[1], 'width': v[2]*2},
                                name=f'{v[0]} - Validation Loss'))
    
    fig1.add_trace(go.Scatter(x=df_tl.step, y=df_tl['Training Loss'],
                                line={'color':v[1], 'width':v[2], 'dash': 'dot'},
                                name=f'{v[0]} - Training Loss'))
fig1.update_layout(template = 'none',
                   legend = {'yanchor':'top', 'y':1.1,
                             'xanchor': 'right', 'x':1})
fig1.show(renderer='browser')


#%% Final Model
df_tl = pd.read_csv(f'results/effnet_sl_tl.csv', names=['time', 'step', 'Training Loss'], skiprows=1)
df_tl['Model'] = "EfficientNetS"
df_vl = pd.read_csv(f'results/effnet_sl_vl.csv', names=['time', 'step', 'Validation Loss'], skiprows=1)
df_vl['Model'] = "EfficientNetS"
df_va = pd.read_csv(f'results/effnet_sl_va.csv', names=['time', 'step', 'Validation Accuracy'], skiprows=1)
df_va['Model'] = "EfficientNetS"


fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_vl.step, y=df_vl['Validation Loss'],
                            line={'color':'rgb(27,158,119)', 'width': 1},
                            name=f'EfficientNetS - Validation Loss'))

fig2.add_trace(go.Scatter(x=df_va.step, y=df_va['Validation Accuracy'],
                            line={'color':'rgb(217,95,2)', 'width':1},
                            name=f'EfficientNetS - Validation Accuracy'))
fig2.update_layout(template = 'none',
                   legend = {'yanchor':'top', 'y':1,
                             'xanchor': 'right', 'x':1})
fig2.show(renderer='browser')

# %%
