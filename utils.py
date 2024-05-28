import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


# Función para eliminar outliers

def outliers_control(df, 
                     id_col: str = 'salesforce_contact_id', 
                     method: str = 'std_dev', 
                     num_sd: int = 4, 
                     lower_percentile: int = 5, 
                     upper_percentile: int = 95, 
                     z_score_threshold: int = 3, 
                     contamination: float = 0.03, 
                     iqr_multiplier: float = 1.5):
    
    # Seleccionar solo las columnas numéricas para el tratamiento de outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Creamos una copia del df
    df_clean = df.copy()
    
    # Lista para rastrear ids con outliers
    outlier_ids = set()
    
    if method == 'isolation_forest':
        # Entrenar el modelo Isolation Forest con todas las columnas numéricas
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(df[numeric_cols])
        # Inferir outliers
        outliers_mask = model.predict(df[numeric_cols]) == -1
        # Recopilar ids de usuarios con outliers
        outlier_ids.update(df[outliers_mask][id_col])
        
    else:
        # Eliminar outliers en variables numéricas según el método especificado
        for col in numeric_cols:
            if method == 'std_dev':
                # Calcular los límites para definir los outliers basados en la desviación estándar
                lower_bound = df[col].mean() - num_sd * df[col].std()
                upper_bound = df[col].mean() + num_sd * df[col].std()
            
            elif method == 'percentiles':
                # Calcular los límites para definir los outliers basados en percentiles
                lower_bound = np.percentile(df[col], lower_percentile)
                upper_bound = np.percentile(df[col], upper_percentile)
            
            elif method == 'z_score':
                # Calcular los z-scores para definir los outliers
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                lower_bound = df[col][z_scores <= z_score_threshold].min()
                upper_bound = df[col][z_scores <= z_score_threshold].max()
            
            elif method == 'iqr':
                # Calcular los límites para definir los outliers basados en IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR    
            
            else:
                raise ValueError("Método no válido. Por favor, elija 'std_dev', 'percentiles', 'z_score' o 'isolation_forest'")
            
            # Identificar ids de usuarios con outliers en la columna actual
            outlier_ids.update(df[(df[col] < lower_bound) | (df[col] > upper_bound)][id_col])
    
    # Eliminar registros de usuarios con outliers
    df_clean = df_clean[~df_clean[id_col].isin(outlier_ids)]

    return df_clean



def transform_df(df):

    pass