import pandas as pd
from scipy import stats

def test_norm_and_homo(df:pd.DataFrame, 
                   dependent_field:str, 
                   category_field:str, 
                   categories_list:list[int],
                   show:str = '') -> None:
    if len(categories_list) > 0:
        if 'all' == show:
          shap = stats.shapiro(df[dependent_field])  
          samples = [df[df[category_field] == category][dependent_field] for category in categories_list]
          levene = stats.levene(*samples)
          mean = df[dependent_field].mean()
          std = df[dependent_field].std()
          print(f"Moyenne ({category_field}) {categories_list}: {round(mean, 3)}\n"
                f"Ecart-type ({category_field}) {categories_list}: {round(std, 3)}\n"
                f"Valeurs test shapiro ({category_field}) {categories_list}:\n\t"
                  f"statistic: {round(shap.statistic, 3)}\n\t"
                  f"p-value: {round(shap.pvalue, 3)}\n"
                  f"\t{f'La variable {dependent_field} suit une loi normale car {round(shap.pvalue, 3)} > 0.05' if round(shap.pvalue, 3) > 0.05 else f'{dependent_field} Ne suit pas une loi normale {round(shap.pvalue, 3)} < 0.05'}\n"
                f"Test de Levene ({category_field}) {categories_list}:\n\t" 
                  f"statistic: {round(levene.statistic, 3)}\n\t"
                  f"p-value: {round(levene.pvalue, 3)}")
        else:
          for category_value in categories_list:
            mean = df[df[category_field] == category_value][dependent_field].mean()
            std = df[df[category_field] == category_value][dependent_field].std()
            shap = stats.shapiro(df[df[category_field] == category_value][dependent_field])
            print(f"Moyenne ({category_field}) [{category_value}]: {round(mean, 3)}\n"
                  f"Ecart-type ({category_field}) [{category_value}]: {round(std, 3)}\n"
                  f"Valeurs test shapiro ({category_field}) [{category_value}]:\n\t"
                    f"statistic: {round(shap.statistic, 3)}\n\t"
                    f"p-value: {round(shap.pvalue, 3)}\n"
                  f"\t{f'La variable {dependent_field} suit une loi normale car {round(shap.pvalue, 3)} > 0.05' if round(shap.pvalue, 3) > 0.05 else f'{dependent_field} Ne suit pas une loi normale {round(shap.pvalue, 3)} < 0.05'}\n"
            )
