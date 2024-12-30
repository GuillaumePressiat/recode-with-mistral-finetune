import os
import pandas as pd
import numpy as np
import re
from random import sample,randint
import hashlib

def get_dataset_partitions(df, train_split=0.8, val_split=0.1, test_split=0.1):
    assert (train_split + test_split + val_split) == 1
    
    # Only allows for equal validation and test splits
    assert val_split == test_split 

    # Specify seed to always have the same split distribution between runs
    df_sample = df.sample(frac=1, random_state=12)
    train_i = int(train_split * len(df))
    val_i = int(val_split * len(df)) + train_i
    indices_or_sections = [train_i,val_i]
    
    train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)
    
    return train_ds, val_ds, test_ds


def get_fictional_scenario(df_dp : pd.DataFrame , 
                  df_das: pd.DataFrame, 
                  n_das :int = None, 
                  max_das : int= 7, 
                  los : int = None, 
                  max_los : int = 10) :
    """
    Function to sample clinical scenario made of 1 principal diagnosis, n_das associated diagnosis and a length of staty
    
    Arguments
    -------------------
    df_dp : DataFrame. 2 colums at least ["libelle","code"]. ICD-10 pincipal diagnosis 
    df_das : DataFrame. 2 colums at least ["libelle","code"]. ICD-10 associated diagnosis
    n_das : int. number of associated diagnosis. Optionnal (the function can sample a number)
    max_das : int. maximal number of DAS if to sampled (n_das = None). Defaut 7.
    los : int. length of stay. Optional  (the function can sample a number)
    max_los :  int. Maximal length of stay if to sampled (n_das = None). Defaut 10.

    Related diagnosis are picked only if DP is a cancer (first letter of ICD-10 code = C).
    In that case, the function sample between 3 situations : 
    - chemotherapy  : DP = Z511, DR : sampled DP
    - radiotherapy  : DP = Z5101, DR : sampled DP
    - classical hopsitalisation :  DP = sampled DP, DR : None

    When n_das = 0 (given or sampled by the function) los between 0-2.

    Results
    ------------------
    df_diags :  DataFrame with 3 colmuns = concept_name, concept_code, condition_status_source_value. 
                Will be use to build the fictinal ICD-10 database. 
    los :       Int. Sampled length of stay
    text_diags: String. The ICD-10 diagnosis in the format 
                "
                Diagnostic principal : libelle (code)
                Diagnostic relié : libelle (code) ou aucun
                Diagnostic principal : libelle (code) ou aucun
                "
    """
    
    # Sample n_das if not given
    if n_das is None:
        n_das = randint(0,max_das)

    # Sample DP from df_dp
    prep = df_dp.sample(1).reset_index().loc[:,["libelle","code"]].rename(columns = {"code" : "concept_code","libelle" : "concept_name"})
    prep["condition_status_source_value"] = "DP"
    
    # if DP is cancer sample between chemotherapy, radiotherapy and classical hopsitalisation
    if prep.concept_code[0][0] =="C" :

        motive = sample(["Z511","Z5101","HC"],1)[0]
        if  motive=="Z511":
            prep["condition_status_source_value"] = "DR"
            prep = prep.append({'concept_name': 'Séance de chimiothérapie pour tumeur', 'concept_code': "Z511", "condition_status_source_value": "DP"},ignore_index=True)
            los = 0
        elif  motive=="Z5101":
            prep["condition_status_source_value"] = "DR"
            prep = prep.append({'concept_name': 'Séance de radiothérapie', 'concept_code': "Z5101", "condition_status_source_value": "DP"},ignore_index=True)
            los = 0
        # if motive = HC then no DR
        else :
            prep = prep.append({'concept_name': np.nan, 'concept_code': np.nan, "condition_status_source_value": "DR"},ignore_index=True)
    # else no DR
    else :
        prep = prep.append({'concept_name': np.nan, 'concept_code': np.nan, "condition_status_source_value": "DR"},ignore_index=True)

    #Choose length of stay and associated diagnosis
    if n_das == 0:

        #los : between 0 and 2 if no associated diagnosis
        if los is None :  # do not modify if already fixed during previous steps
            los = randint(0,2)
        
        #No DAS
        df_diags_prep = prep.append({'concept_name': np.nan, 'concept_code': np.nan, "condition_status_source_value": "DAS"},ignore_index=True)

        
    else : 
        
        #los between 3 and 10 days
        if los is None : # do not modify if already fixed during previous steps
            los = randint(3,max_los)
        
        #pick some DAS from df_das
        prep_ = df_das.sample(n_das).reset_index().loc[:,["libelle","code"]].rename(columns = {"code" : "concept_code","libelle" : "concept_name"})
        prep_["condition_status_source_value"] = "DAS"
        df_diags_prep = pd.concat([prep,prep_])

    # Prepare the final DataFrame of ICD diagnosis (only take lines whith an effective diagnosis)
    df_diags = df_diags_prep[df_diags_prep.concept_code.notna()]

    # Concatenate ICD-10 label and code.
    df_diags_prep["diag"] = np.where(df_diags_prep.concept_name.notna(), df_diags_prep.concept_name + ' (' + df_diags_prep.concept_code.replace(" ","") +')' , 'Aucun' )
    df_diags_prep = df_diags_prep.groupby("condition_status_source_value")['diag'].apply(lambda x: ', '.join(x)).reset_index()

    text_diags = " Diagnostics CM-10 :\n- Diagnostic principal : " +  df_diags_prep.iloc[1,1] +".\n- Diagnostics relié : "+ df_diags_prep.iloc[2,1] +".\n" +"- Diagnostics associés :" +  df_diags_prep.iloc[0,1]
    return df_diags,los,text_diags

def get_visit_occurence_id():
    num = randint(1,1000000000000)
    m = hashlib.sha1(str(num).encode())
    return m.hexdigest()