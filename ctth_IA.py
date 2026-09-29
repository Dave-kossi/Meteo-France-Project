#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 14:17:13 2025

@author: lebastardt
"""

# Vérifier pourquoi on a pas les même hist2D en frac et non frac pour les num_ct != 10

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.preprocessing import StandardScaler


import joblib

import os
os.environ['KERAS_BACKEND'] = 'torch'

import keras

import tensorflow as tf

import pickle


import torch
import torch.nn as nn


mpl.rcParams['interactive'] = False
np.set_printoptions(suppress=True)

pd.set_option('display.max_rows', None)

seed = 42
np.random.seed(seed)

plot_figures = False
compute_feature_importance = False

path_expe = './Experiences/'
path_data = './Data/'


list_CT = ['Cloud free land', 'Cloud free sea', 'Land contaminated\nby snow',
            'Sea contaminated\nby snow/ice', 'Very low clouds', 'Low clouds', 'Mid-level clouds',
            'High clouds', 'Very high clouds', 'Fractional clouds', 'High semitransparent\nthin clouds',
            'High semitransparent\nmeanly thick clouds', 'High semitransparent\nthick clouds',
            'High semitransparent above\nlow or medium clouds', 'High semitransparent\nabove snow/ice' ]

# Cloud Types
#  1 -- Cloud free land
#  2 -- Cloud free sea
#  3 -- Land contaminated by snow
#  4 -- Sea contaminated by snow/ice
#  5 -- Very low clouds
#  6 -- Low clouds
#  7 -- Mid-level clouds
#  8 -- High clouds
#  9 -- Very high clouds
# 10 -- Fractional clouds
# 11 -- High semitransparent thin clouds
# 12 -- High semitransparent meanly thick clouds
# 13 -- High semitransparent thick clouds
# 14 -- High semitransparent above low or medium clouds
# 15 -- High semitransparent above snow/ice



##### Fonctions #####


def multi_output_mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))



def run_model_keras_MLP(xtrain, ytrain, valid, param):
    
    model = keras.models.Sequential()
    model.add(keras.layers.Input(xtrain.shape[1], name="InputLayer"))
    #model.add(keras.layers.Dropout(0.2))
    num_layer = 0
    for i in param['hiden_layers_sizes']:
        num_layer += 1
        model.add(keras.layers.Dense(i, kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l2(0.01), name='Dense_n'+str(num_layer)))
        model.add(keras.layers.LayerNormalization(name='Normalization_n'+str(num_layer)))
        if (num_layer<len(param['hiden_layers_sizes'])):
            model.add(keras.layers.Activation(param['activation'], name='Activation_n'+str(num_layer)))
        #model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(1, name='Dense_Output'))


    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9)

    opt = keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer = opt,
                loss      = param['loss_function'].lower(),
                metrics   = [multi_output_mae] )
    
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=param['EarlyStopping'], restore_best_weights=True, mode='min')
  

    history = model.fit(xtrain,
            ytrain,
            validation_data = valid,
            epochs          = param['max_iter'],
            batch_size      = param['batch_size'],
            verbose         = 2,
            callbacks=[callback]) 
    
    return model, history





def feature_importance(mod, x_test, y_test):
    importance = np.zeros((x_test.shape[1]))
    for i in range(x_test.shape[1]):
        x_permuted = x_test.copy()
        np.random.shuffle(x_permuted[:, i])
        y_pred_permuted = mod.predict(x_permuted).squeeze()
        importance[i] = np.nanmean(np.abs(y_pred_permuted-y_test))
  
    return importance


def forward(x):
    return x**(1/5)

def inverse(x):
    return x**5


##### Fin fonctions #####



# Canaux IR de MTG-FCI
list_chan_model = ['WV_063','WV_073','IR_087','IR_105','IR_123','IR_133']


# Simulations RTTOV en ciel clair
clear_sky_data = True
list_chan_ClearSky = ['WV_063','WV_073','IR_087','IR_105','IR_123','IR_133']

# Données ERA5 de température
ERA5_temperature_data = True
ERA5_pressure_levels_profile_T = [1000, 850, 700, 500, 300, 200, 100]


# Données statistiques du voisinage
surrounding_chan = True
surrounding_chan_stat_list = ['C', 'W', 'std']    # C : coldest / W : warmest / std : standard deviation


# Configuration du modèle
param_mod = {}
param_mod['hiden_layers_sizes'] = (64, 32)    # Couches cachées
param_mod['max_iter'] = 200                  # Nombre d'epochs
param_mod['EarlyStopping'] = 10               # Early stopping
param_mod['activation'] = 'relu'              # Fonction d'activation
param_mod['learning_rate'] = 0.001            # Taux d'apprentissage
param_mod['batch_size'] = 256                 # Taille des paquets
param_mod['loss_function'] = "MAE"            # Fonction de perte (MSE, MAE, Smooth ou Huber)


loss_fn_map = {
    "MSE": nn.MSELoss(),
    "MAE": nn.L1Loss(),
    "Smooth": nn.SmoothL1Loss(),
    "Huber" : nn.HuberLoss()
}
criterion = loss_fn_map[param_mod['loss_function']]

model_name = 'Reference'


# Création des dossiers pour ranger les figures et le modèle
os.makedirs(path_expe+model_name+'/Figures/', mode=0o750, exist_ok=True)
os.makedirs(path_expe+model_name+'/Model/', mode=0o750, exist_ok=True)



# Liste des variables pour l'entraînement
list_training_variables = list_chan_model.copy()
if(clear_sky_data==True):
    list_training_variables += [chan+"_ClearSky" for chan in list_chan_ClearSky]
if(surrounding_chan == True):
    list_training_variables += [chan+'_'+param+'5' for chan in list_chan_model for param in surrounding_chan_stat_list]
if(ERA5_temperature_data == True):
    list_training_variables += ['T_'+str(pres_lev) for pres_lev in ERA5_pressure_levels_profile_T]
list_training_variables = sorted(set(list_training_variables), key=list_training_variables.index)         
   

# Variable à prédire 
var_predict_EarthCARE = ['CloudTops_Layers_0']





##################### Préparation des données #####################


# Importation des données
data_all = pd.read_pickle(path_data+'data_cloud_top.pkl')
nb_data = data_all.shape[0]


# Liste des jours présents dans la base de données
list_dates = list(np.unique(data_all['Day_UTC']))

# Listes des dates utilisées pour les jeux d'entraînement, de validation et de test
list_dates_train = list_dates[0:11]
list_dates_validation = list_dates[11:12]
list_dates_test = list_dates[12:13]

# Index des points sélectionnés pour les jeux d'entraînement, de validation et de test
idx = {}
idx['train'] = np.arange(nb_data)[np.isin(data_all['Day_UTC'], list_dates_train)]
idx['validation'] = np.arange(nb_data)[np.isin(data_all['Day_UTC'], list_dates_validation)]
idx['test'] = np.arange(nb_data)[np.isin(data_all['Day_UTC'], list_dates_test)]


print("Répartition du jeu de données :\ntrain: "+str(np.round(100*(idx['train'].shape[0])/(idx['train'].shape[0]+idx['validation'].shape[0]+idx['test'].shape[0]), 3))+" %\nvalidation: "+str(np.round(100*(idx['validation'].shape[0])/(idx['train'].shape[0]+idx['validation'].shape[0]+idx['test'].shape[0]), 3))+" %\ntest: "+str(np.round(100*(idx['test'].shape[0])/(idx['train'].shape[0]+idx['validation'].shape[0]+idx['test'].shape[0]), 3))+" %")


# Données du SAFNWC ("cloud top altitude" et "cloud type")
data_NWCSAF = {}
data_NWCSAF['train'] = data_all[['Cloud_Top_Altitude_SAFNWC', 'Cloud_Type_SAFNWC']].iloc[idx['train']]
data_NWCSAF['validation'] = data_all[['Cloud_Top_Altitude_SAFNWC', 'Cloud_Type_SAFNWC']].iloc[idx['validation']]
data_NWCSAF['test'] = data_all[['Cloud_Top_Altitude_SAFNWC', 'Cloud_Type_SAFNWC']].iloc[idx['test']]



# Sélection des données utilisées dans le modèle
data_model = data_all.drop(['Day_UTC', 'HH_UTC', 'MN_UTC']+['Cloud_Top_Altitude_SAFNWC']+['Cloud_Type_SAFNWC'], axis=1)

# Variable à prédire
var_to_predict = 'Cloud_Top_Altitude_EarthCARE'


# Sélection du pixel central pour chacun des canaux
for chn in list_chan_model:
    data_model[chn] = np.stack(data_model[chn])[:, 12]


print('------------- Statistics of variables -------------')
print('Name | Mean | Minimum | Maximum | Nb of Nan')
for key in data_model.keys():
    print(key+' | '+str(np.round(np.nanmean(data_model[key].astype(float)), 5))+' | '+
          str(np.round(np.nanmin(data_model[key].astype(float)), 5))+' | '+
          str(np.round(np.nanmax(data_model[key].astype(float)), 5))+' | '+
          str(np.sum(np.isnan(data_model[key].astype(float)))))
print('---------------------------------------------------')


x, y = {}, {}
x['train'] = np.asarray(data_model[list_training_variables].iloc[idx['train']]).astype('float32')
y['train'] = np.asarray(data_model[var_to_predict].iloc[idx['train']]).astype('float32')
x['validation'] = np.asarray(data_model[list_training_variables].iloc[idx['validation']]).astype('float32')
y['validation'] = np.asarray(data_model[var_to_predict].iloc[idx['validation']]).astype('float32')
x['test'] = np.asarray(data_model[list_training_variables].iloc[idx['test']]).astype('float32')
y['test'] = np.asarray(data_model[var_to_predict].iloc[idx['test']]).astype('float32')


##################### Fin préparation des données #####################



# Normalisation des données
fic_scaler = path_expe+model_name+'/Model/'+model_name+'_scaler.save'
if(os.path.exists(fic_scaler)):
    scaler = joblib.load(fic_scaler)
else:
    sc=StandardScaler()
    scaler = sc.fit(x['train'])
    joblib.dump(scaler, path_expe+model_name+'/Model/'+model_name+'_scaler.save') 
      
x_scaled= {}  
x_scaled['train'] = scaler.transform(x['train'])
x_scaled['validation'] = scaler.transform(x['validation'])
x_scaled['test'] = scaler.transform(x['test'])



fic_model = path_expe+model_name+'/Model/'+model_name+'_model.keras'
if(os.path.exists(fic_model)):
    # Chargement du modèle s'il existe déjà
    model = keras.models.load_model(fic_model, custom_objects={"multi_output_mae": multi_output_mae})
else:
    # Calcul du modèle
    model, history = run_model_keras_MLP(x_scaled['train'], y['train'], (x_scaled['validation'], y['validation']), param_mod)

    # Tracé de l'évolution de la fonction de perte        
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1) 
    ax.plot(history.history['loss'], label='Train '+param_mod['loss_function'])
    ax.plot(history.history['val_loss'], label='Validation '+param_mod['loss_function'])
    ax.set_yscale('function', functions=(forward, inverse))
    plt.legend()
    plt.show()
    plt.savefig(path_expe+model_name+'/Figures/loss_evolution_'+model_name+'.png', dpi=300)
    
    with open(path_expe+model_name+'/Model/variables.txt', "w") as myfile:
        for key in data_model.keys():
            myfile.write(key+'\n')

    with open(path_expe+model_name+'/Model/model_parameters.txt', "w") as myfile:
        for key, value in param_mod.items():
            myfile.write(str(key) + ' : ' + str(value) + '\n')

# Sauvegarde du modèle
model.save(path_expe+model_name+'/Model/'+model_name+'_model.keras')


# Prédictions de l'altitude du sommet des nuages sur les différents jeux de données
y_pred = {}
y_pred['train'] = model.predict(x_scaled['train'])
y_pred['validation'] = model.predict(x_scaled['validation'])
y_pred['test'] = model.predict(x_scaled['test'])


# Tracé des figures
liste_type_plot = ['validation', 'train', 'test']
if(plot_figures == True):
    print("------ Tracé des figures ------") 
    for type_plot in liste_type_plot:
    
        x_plot_temp, y_plot_temp1, y_plot_temp2 = {}, {}, {}  
    
        x_plot_temp = y[type_plot]
        y_plot_temp1 = y_pred[type_plot][:, 0] 
        y_plot_temp2 = data_NWCSAF[type_plot]['Cloud_Top_Altitude_SAFNWC']
    
        mask_temp = (~np.isnan(x_plot_temp)) & (~np.isnan(y_plot_temp1)) & (~np.isnan(y_plot_temp2))
        
        x_plot_temp = x_plot_temp[mask_temp]
        y_plot_temp1 = y_plot_temp1[mask_temp]   
        y_plot_temp2 = y_plot_temp2[mask_temp]        
    
    
        # CTTH IA
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(1, 1, 1) 
        cax1 = plt.hist2d(x_plot_temp, y_plot_temp1, bins=np.arange(0, 20000, 500), cmap='jet', norm=mpl.colors.LogNorm(vmax=1000), cmin=1)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=16)
        cb.set_label('Counts', fontsize=16)
        plt.xlim(0, 20000)
        plt.ylim(0, 20000) 
        diag_line = plt.plot([0, 20000], [0, 20000], ls="--", c='k')
        plt.grid()
        bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.8)
        plt.text(1000, 16000, "N: "+str(int(x_plot_temp.shape[0]))+"\nMAE: "+str(int(np.nanmean(np.abs(x_plot_temp-y_plot_temp1))))+" m\nMean error: "+str(int(np.nanmean(y_plot_temp1-x_plot_temp)))+" m\nRMSE: "+str(int(np.sqrt(np.nanvar(x_plot_temp-y_plot_temp1))))+" m\nR2: "+str(np.round(np.corrcoef(x_plot_temp, y_plot_temp1)[0, 1]**2, 3)), bbox=bbox, fontsize=13)
        plt.xticks(fontsize=16, rotation = 45, ha='right')
        plt.yticks(fontsize=16)
        plt.xlabel('EarthCARE - CPR (m)', fontsize=16)
        plt.ylabel('Predicted (m)', fontsize=16)
        fig.tight_layout()
        plt.savefig(path_expe+model_name+'/Figures/hist2d_IA-alti_'+model_name+'_'+type_plot+'.png', dpi=300)
        plt.close()  
    
        
        # CTTH IA (haute résolution)
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(1, 1, 1) 
        cax1 = plt.hist2d(x_plot_temp, y_plot_temp1, bins=np.arange(0, 20000, 100), cmap='jet', norm=mpl.colors.LogNorm(vmax=1000), cmin=1)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=16)
        cb.set_label('Counts', fontsize=16)
        plt.xlim(0, 20000)
        plt.ylim(0, 20000) 
        diag_line = plt.plot([0, 20000], [0, 20000], ls="--", c='k')
        plt.grid()
        bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.8)
        plt.text(1000, 16000, "N: "+str(int(x_plot_temp.shape[0]))+"\nMAE: "+str(int(np.nanmean(np.abs(x_plot_temp-y_plot_temp1))))+" m\nMean error: "+str(int(np.nanmean(y_plot_temp1-x_plot_temp)))+" m\nRMSE: "+str(int(np.sqrt(np.nanvar(x_plot_temp-y_plot_temp1))))+" m\nR2: "+str(np.round(np.corrcoef(x_plot_temp, y_plot_temp1)[0, 1]**2, 3)), bbox=bbox, fontsize=13)
        plt.xticks(fontsize=16, rotation = 45, ha='right')
        plt.yticks(fontsize=16)
        plt.xlabel('EarthCARE - CPR (m)', fontsize=16)
        plt.ylabel('Predicted (m)', fontsize=16)
        fig.tight_layout()
        plt.savefig(path_expe+model_name+'/Figures/hist2d_IA-alti_'+model_name+'_'+type_plot+'_HR.png', dpi=300)
        plt.close()  
    
   
        # CTTH du SAFNWC
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(1, 1, 1) 
        cax1 = plt.hist2d(x_plot_temp, y_plot_temp2, bins=np.arange(0, 20000, 500), cmap='jet', norm=mpl.colors.LogNorm(vmax=1000), cmin=1)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=16)
        cb.set_label('Counts', fontsize=16)
        plt.xlim(0, 20000)
        plt.ylim(0, 20000) 
        diag_line = plt.plot([0, 20000], [0, 20000], ls="--", c='k')
        plt.grid()
        bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.8)
        plt.text(1000, 16000, "N: "+str(int(x_plot_temp.shape[0]))+"\nMAE: "+str(int(np.nanmean(np.abs(x_plot_temp-y_plot_temp2))))+" m\nMean error: "+str(int(np.nanmean(y_plot_temp2-x_plot_temp)))+" m\nRMSE: "+str(int(np.sqrt(np.nanvar(x_plot_temp-y_plot_temp2))))+" m\nR2: "+str(np.round(np.corrcoef(x_plot_temp, y_plot_temp2)[0, 1]**2, 3)), bbox=bbox, fontsize=13)
        plt.xticks(fontsize=16, rotation = 45, ha='right')
        plt.yticks(fontsize=16)
        plt.xlabel('EarthCARE - CPR (m)', fontsize=16)
        plt.ylabel('CTTH - SAFNWC/GEO (m)', fontsize=16)
        fig.tight_layout()
        plt.savefig(path_expe+model_name+'/Figures/hist2d_GEO-alti_'+model_name+'_'+type_plot+'.png', dpi=300)
        plt.close()
        
    
        # CTTH du SAFNWC (haute résolution)
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(1, 1, 1) 
        cax1 = plt.hist2d(x_plot_temp, y_plot_temp2, bins=np.arange(0, 20000, 100), cmap='jet', norm=mpl.colors.LogNorm(vmax=1000), cmin=1)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=16)
        cb.set_label('Counts', fontsize=16)
        plt.xlim(0, 20000)
        plt.ylim(0, 20000) 
        diag_line = plt.plot([0, 20000], [0, 20000], ls="--", c='k')
        plt.grid()
        bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.8)
        plt.text(1000, 16000, "N: "+str(int(x_plot_temp.shape[0]))+"\nMAE: "+str(int(np.nanmean(np.abs(x_plot_temp-y_plot_temp2))))+" m\nMean error: "+str(int(np.nanmean(y_plot_temp2-x_plot_temp)))+" m\nRMSE: "+str(int(np.sqrt(np.nanvar(x_plot_temp-y_plot_temp2))))+" m\nR2: "+str(np.round(np.corrcoef(x_plot_temp, y_plot_temp2)[0, 1]**2, 3)), bbox=bbox, fontsize=13)
        plt.xticks(fontsize=16, rotation = 45, ha='right')
        plt.yticks(fontsize=16)
        plt.xlabel('EarthCARE - CPR (m)', fontsize=16)
        plt.ylabel('CTTH - SAFNWC/GEO (m)', fontsize=16)
        fig.tight_layout()
        plt.savefig(path_expe+model_name+'/Figures/hist2d_GEO-alti_'+model_name+'_'+type_plot+'_HR.png', dpi=300)
        plt.close()
    
        
        
        # Tracé par type de nuages (SAFNWC)
        for num_ct in np.arange(1, 15, 1):
            if(num_ct!=10):
                filter_temp = (data_NWCSAF[type_plot]['Cloud_Type_SAFNWC']==num_ct)
                filter_temp = filter_temp[mask_temp]
                
                if(np.sum(filter_temp>0)):
                    
                    # CTTH IA
                    fig = plt.figure(figsize=(8, 7))
                    ax = fig.add_subplot(1, 1, 1) 
                    cax1 = plt.hist2d(x_plot_temp[filter_temp], y_plot_temp1[filter_temp], bins=np.arange(0, 20000, 100), cmap='jet', norm=mpl.colors.LogNorm(vmax=1000), cmin=1)
                    cb = plt.colorbar()
                    cb.ax.tick_params(labelsize=16)
                    cb.set_label('Counts', fontsize=16)
                    plt.xlim(0, 20000)
                    plt.ylim(0, 20000) 
                    diag_line = plt.plot([0, 20000], [0, 20000], ls="--", c='k')
                    plt.grid()
                    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.8)
                    plt.text(1000, 16000, "N: "+str(int(x_plot_temp[filter_temp].shape[0]))+"\nMAE: "+str(int(np.nanmean(np.abs(x_plot_temp[filter_temp]-y_plot_temp1[filter_temp]))))+" m\nMean error: "+str(int(np.nanmean(y_plot_temp1[filter_temp]-x_plot_temp[filter_temp])))+" m\nRMSE: "+str(int(np.sqrt(np.nanvar(x_plot_temp[filter_temp]-y_plot_temp1[filter_temp]))))+" m\nR2: "+str(np.round(np.corrcoef(x_plot_temp[filter_temp], y_plot_temp1[filter_temp])[0, 1]**2, 3)), bbox=bbox, fontsize=13)
                    plt.title(list_CT[num_ct-1], fontsize=18)
                    plt.xticks(fontsize=16, rotation = 45, ha='right')
                    plt.yticks(fontsize=16)
                    plt.xlabel('EarthCARE - CPR (m)', fontsize=16)
                    plt.ylabel('Predicted (m)', fontsize=16)
                    fig.tight_layout()
                    plt.savefig(path_expe+model_name+'/Figures/hist2d_IA-alti_'+model_name+'_CT'+str(num_ct)+'_'+type_plot+'.png', dpi=300)
                    plt.close()  
                    
                    
                
                    # CTTH du SAFNWC
                    fig = plt.figure(figsize=(8, 7))
                    ax = fig.add_subplot(1, 1, 1) 
                    cax1 = plt.hist2d(x_plot_temp[filter_temp], y_plot_temp2[filter_temp], bins=np.arange(0, 20000, 100), cmap='jet', norm=mpl.colors.LogNorm(vmax=1000), cmin=1)
                    cb = plt.colorbar()
                    cb.ax.tick_params(labelsize=16)
                    cb.set_label('Counts', fontsize=16)
                    plt.xlim(0, 20000)
                    plt.ylim(0, 20000) 
                    diag_line = plt.plot([0, 20000], [0, 20000], ls="--", c='k')
                    plt.grid()
                    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.8)
                    plt.text(1000, 16000, "N: "+str(int(x_plot_temp[filter_temp].shape[0]))+"\nMAE: "+str(int(np.nanmean(np.abs(x_plot_temp[filter_temp]-y_plot_temp2[filter_temp]))))+" m\nMean error: "+str(int(np.nanmean(y_plot_temp2[filter_temp]-x_plot_temp[filter_temp])))+" m\nRMSE: "+str(int(np.sqrt(np.nanvar(x_plot_temp[filter_temp]-y_plot_temp2[filter_temp]))))+" m\nR2: "+str(np.round(np.corrcoef(x_plot_temp[filter_temp], y_plot_temp2[filter_temp])[0, 1]**2, 4)), bbox=bbox, fontsize=13)
                    plt.title(list_CT[num_ct-1], fontsize=18)
                    plt.xticks(fontsize=16, rotation = 45, ha='right')
                    plt.yticks(fontsize=16)
                    plt.xlabel('EarthCARE - CPR (m)', fontsize=16)
                    plt.ylabel('CTTH - SAFNWC/GEO (m)', fontsize=16)
                    fig.tight_layout()
                    plt.savefig(path_expe+model_name+'/Figures/hist2d_GEO-alti_'+model_name+'_CT'+str(num_ct)+'_'+type_plot+'.png', dpi=300)
                    plt.close()
                
            
    

# Tracé de l'importance des variables par permutations
if(compute_feature_importance == True):

    print("------ Analyse de l'importance des variables ------")    

    importance = feature_importance(model, x_scaled['test'], y['test'].squeeze())

    idx_sort = np.argsort(importance)

    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(1, 1, 1) 
    cax1 = plt.barh(np.arange(x_scaled['test'].shape[1]), importance[idx_sort])
    ax.set_yticks(np.arange(x_scaled['test'].shape[1]), np.array([var for var in list_training_variables])[idx_sort])

    fig.tight_layout()

    plt.savefig(path_expe+model_name+'/Figures/Feature_importance_permutation_'+model_name+'.png', dpi=300)
    plt.close()  





