#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 3rd March 2020

Script used to produce variograms in the analyses in Sitka Spruced project.

The input for the script is the yht file produced by ASReml.
For all pairs of residuals, calculate pairwise differences 1/2[resid1 - resid2]^2. Then, displacement column and row Average the residual differences within the displacement group.

There are two options - variogram function produces variogram of all calculated values.
Variogram020 function produces output for displacement values within 0-20.


@author: joannailska
"""

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
mpl.rcParams['agg.path.chunksize'] = 10000
from matplotlib import cm
import glob as glob

def merge_spatial_residuals(site,resid_path):
    '''
    Function merging the original table of phenotypes with the residual phenotype.
    Return dataframe with the general info (site, block, entry, plot) and residual phenotype.
    '''
    
    if site.split('_')[0]=='Torridge':
        codes_trait = {'1':'height_yr2', '2':'height_yr4', '3':'height_yr6', '4':'height_yr11', '5':'density_yr10', '6':'flsc_1', '7':'flsc_2','8':'flsc_3'}
    else:
        codes_trait = {'1':'height_yr2', '2':'height_yr4', '3':'height_yr6', '4':'height_yr11', '5':'flsc_1', '6':'flsc_2','7':'flsc_3'}

    original='../{}/{}_spatial_filled.csv'.format(site, site)
    or_df=pd.read_csv(original, na_values='*')

    s=site.split('_')[0]
    residuals=glob.glob("{}/*yht".format(resid_path))
    N=len(residuals)

    for i in range(N):
        r=i+1
        res=("{}/{}_{}{}.yht".format(resid_path,site,model,r))
        resid=[]
        for line in open(res, 'r').readlines()[1:]:
            tmp=line.strip().split()[2]
            if tmp!='-0.1000E-36':
                resid.append(tmp)
            else:
                resid.append(np.nan)
        
                
        trait_col='res_{}'.format(codes_trait[res[-5]])
        or_df.loc[:,trait_col]=resid

    cols=list(or_df.columns[:5])+list(or_df.columns[-len(residuals)-2:])
    df=or_df[cols]
        
    df.dropna(subset=['entry'], inplace=True)

    return df

## Numba based functions to calculate pairwise differences for a large number of comparisons
@njit
def resid_difference(a):
    N=len(a)
    diff=[]

    for i in range(N):
        for j in range(N):
            if i!=j:
                diff.append(0.5*((a[i]-a[j])**2))
    return diff

@njit
def pos_difference(a):
    N=len(a)
    diff=[]

    for i in range(N):
        for j in range(N):
            if i!=j:
                diff.append(np.absolute(a[i]-a[j]))
    return diff

def variogram(df, trait, model):
    '''
    Function returning a variogram of the residuals for a set of given trait records in a given dataframe
    '''
    
    ## Create an empty dataframe to store all pairwise differences
    diff_df = pd.DataFrame(columns=['difference','col_diff', 'row_diff'])
    ## Remove missing residual records for a given trait, and extract residual, column and row values
    ## into numpy arrays
    df2=df.dropna(subset=[trait])
    vals=df2[trait].astype('float').values
    cols=df2['new_column'].values
    rows=df2['new_row'].values
    
    ## Calculate the differences using numba based functions
    diff=resid_difference(vals)
    diff_cols=pos_difference(cols)
    diff_rows=pos_difference(rows)
    
    ## Fill up the dataframe
    diff_df['difference']=diff
    diff_df['col_diff']=diff_cols
    diff_df['row_diff']=diff_rows
    
    ## Calculate a mean for a given displacement unit
    mean_diff = diff_df.groupby(['col_diff','row_diff']).mean().reset_index()
    ## Add a 0,0 point
    mean_diff.loc[len(mean_diff)+1]=[0,0,0]
    
    ## Sort cols in increment, rows in decrement
    mean_diff = mean_diff.sort_values(['col_diff', 'row_diff'], ascending=[True, False])

    ## Format the values to be in the right shapes for 3d plot
    x = list(set(mean_diff['col_diff'].values))
    y = list(set(mean_diff['row_diff'].values))
    x.sort()
    y.sort()
    X,Y = np.meshgrid(y,x)
    ## Reshape the table with differences
    Z = mean_diff.pivot(index='col_diff', columns='row_diff', values='difference').values
    
    fig = plt.figure()

    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z) #, rstride=1,cstride=1, cmap=cm.coolwarm) #, rstride=1, cstride=1,
    # ax.set_zlim(0,0.02)
    ax.set_xlabel("Column displacement")
    ax.set_ylabel("Row displacement")
    ax.set_zlabel("Residual difference")
    ax.invert_xaxis()
    ax.set_title("Variogram for\n{}, {}".format(trait,model))
#     plt.savefig("Variogram_{}.jpg".format(trait), dpi=199)
    
    return

def variogram020(df, trait,model):
    '''
    Function returning a variogram of the residuals for a set of given trait records in a given dataframe.
    Limited to the displacement values within 0-20.
    '''
    
    ## Create an empty dataframe to store all pairwise differences
    diff_df = pd.DataFrame(columns=['difference','col_diff', 'row_diff'])
    ## Remove missing residual records for a given trait, and extract residual, column and row values
    ## into numpy arrays
    df2=df.dropna(subset=[trait])
    vals=df2[trait].astype('float').values
    cols=df2['new_column'].values
    rows=df2['new_row'].values
    
    ## Calculate the differences using numba based functions
    diff=resid_difference(vals)
    diff_cols=pos_difference(cols)
    diff_rows=pos_difference(rows)
    
    ## Fill up the dataframe
    diff_df['difference']=diff
    diff_df['col_diff']=diff_cols
    diff_df['row_diff']=diff_rows
    
    ## Calculate a mean for a given displacement unit
    mean_diff = diff_df.groupby(['col_diff','row_diff']).mean().reset_index()
    ## Add a 0,0 point
    mean_diff.loc[len(mean_diff)+1]=[0,0,0]
    
    ## Sort cols in increment, rows in decrement
    mean_diff = mean_diff.sort_values(['col_diff', 'row_diff'], ascending=[True, False])
    
    ## Extract only values within 0-20 displacement
    mean_diff = mean_diff.loc[(mean_diff['col_diff']<21) & (mean_diff['row_diff']<21)]

    ## Format the values to be in the right shapes for 3d plot
    x = list(set(mean_diff['col_diff'].values))
    y = list(set(mean_diff['row_diff'].values))
    x.sort()
    y.sort()
    X,Y = np.meshgrid(y,x)
    ## Reshape the table with differences
    Z = mean_diff.pivot(index='col_diff', columns='row_diff', values='difference').values
    
    fig = plt.figure()

    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z) #, rstride=1,cstride=1, cmap=cm.coolwarm) #, rstride=1, cstride=1,
    # ax.set_zlim(0,0.02)
    ax.set_xlabel("Column displacement")
    ax.set_ylabel("Row displacement")
    ax.set_zlabel("Residual difference")
    ax.invert_xaxis()
    ax.set_title("Variogram for\n{},{}".format(trait,model))
#     plt.savefig("Variogram_{}.jpg".format(trait), dpi=199)
    
    return
    
## Run the functions for all sites, and all height traits
for site in ['Huntly_5', 'Huntly_6','Huntly_7','Llandovery_46','Llandovery_47','Llandovery_48','Torridge_66','Torridge_67','Torridge_68']:

    resid_path='../{}/model_choice/{}'.format(site,model)
    df_block = merge_spatial_residuals(site,resid_path)

    model='block_row'
    resid_path='../{}/model_choice/{}'.format(site,model)
    df_block_row = merge_spatial_residuals(site,resid_path)

    model='block_col'
    resid_path='../{}/model_choice/{}'.format(site,model)
    df_block_col = merge_spatial_residuals(site,resid_path)

    model='block_row_col'
    resid_path='../{}/model_choice/{}'.format(site,model)
    df_block_row_col = merge_spatial_residuals(site,resid_path)

    for trait in ['res_height_yr2','res_height_yr4','res_height_yr6','res_height_yr11']:
        variogram(df_block, trait,'block')
        variogram(df_block_row,trait,'block_row')
        variogram(df_block_col,trait,'block_col')
        variogram(df_block_row_col,trait,'block_row_col')
        variogram020(df_block, trait,'block')
        variogram020(df_block_row,trait,'block_row')
        variogram020(df_block_col,trait,'block_col')
        variogram020(df_block_row_col,trait,'block_row_col')
