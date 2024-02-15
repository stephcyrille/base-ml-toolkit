import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split



class ClassifierProcessorTool:
  """
    Classe qui permet de faire la classification supervisée d'un problème de Data Science

    Attributs
    ---------
      df: pd.DataFrame
          Le dataframe surquel se fera les traitements, idéalement un objet pd.DataFrame, // Detqils sur NA etc..
      features: list[str] 
          Les caractéristiques du jeu de données dans la variable cible
      target: str 
          Il s'agit de la valeur à prédire 
      p_model: KNeighborsClassifier | SVC | GaussianProcessClassifier | DecisionTreeClassifier | RandomForestClassifier | MLPClassifier | AdaBoostClassifier | GaussianNB | QuadraticDiscriminantAnalysis
          Model de classification qui serait choisit par défaut ce qui évitera de faire le traitement en testant tous les modèles disponibles
  """

  def __init__(self, 
                df:pd.DataFrame = pd.DataFrame([]),
                features:list[str] = [],
                target:str = '',
                p_model:KNeighborsClassifier | SVC | GaussianProcessClassifier | DecisionTreeClassifier | RandomForestClassifier | MLPClassifier | AdaBoostClassifier | GaussianNB | QuadraticDiscriminantAnalysis =  None
               ) -> None:
    self.df:pd.DataFrame = df
    self.features:list[str] = features
    self.target:str = target
    self.models:list[dict[str, object]] = [
      {"Nearest Neighbors": KNeighborsClassifier()}, 
      {"Linear SVM": SVC()}, 
      {"RBF SVM": SVC()}, 
      {"Gaussian Process": GaussianProcessClassifier()}, 
      {"Decision Tree": DecisionTreeClassifier()}, 
      {"Random Forest": RandomForestClassifier()}, 
      {"Neural Net": MLPClassifier()}, 
      {"AdaBoost": AdaBoostClassifier()}, 
      {"Naive Bayes": GaussianNB()}, 
      {"QDA": QuadraticDiscriminantAnalysis()}
    ]
    self.p_model = p_model
  
  def __str__(self) -> str:
    return f"Super classifier\nFeatures: {self.features}\nTarget: {self.target}\nModels list: {[list(x.keys())[0] for x in self.models]}"

  def _get_features(self, target:str) -> list:
    """
      Méthode permettant de récupérer la liste des caractéristiques du jeu de données. Le principe étant de connaitre la variable cible au préalable. Ici on récupère d'abord toutes les colonnes puis on supprime la colonne à prédire.

      Attributs
      ---------
        target: str
            Il s'agit de la variable cible à prédire
    """
    X = self.df.columns.to_list()
    if target in X:
      X.remove(target)
      return X
    else:
      print(f"Il semblerait que la colonne {target} n'existe pas dans votre dataset")
      return []
  
  def get_avalaible_models(self) -> str:
    """
      Cette méthode permet de récupérer la liste de tous les algorithmes de classification disponibles dans notre classe  
    """
    return f'{[list(x.keys())[0] for x in self.models]}'
  
  def get_features_by_type(self, 
                           type_of_feature:str = '', 
                           target:str = '') -> list[str]:
      """
      Méthode permettant de récupérer la liste des caractéristiques du jeu de données par type de variable pandas. Le principe étant de connaitre la variable cible au préalable. Ici on récupère toutes les colonnes d'un type, puis on supprime la colonne à prédire si elle contient le même type que les variable caractéristique.

      Attributs
      ---------
        type_of_feature: str [int32|int64|float64|...]
            il s'agit du type de variable pour les objets de type Series python
        target: str
            Il s'agit de la variable cible à prédire
    """
      if '' != type_of_feature:
        try:
          columns:list[str] = self.df.select_dtypes(include=[type_of_feature]).columns.to_list()
        except:
          error:str = f"Le type {type_of_feature} n'est pas un dtypes valide en Pandas."
          raise Exception(error)
      else:
        columns:list[str] = self.df.columns.to_list()
      # Vérifions si la veuleur de y est définie et contenue dans la la liste des colonnes
      if '' != target:
        if target in columns:
          columns.remove(target)
        else:
          return columns
      else:
        return []
    
  def process_features_standardisation(self, 
                                       columns:list[str], 
                                       transformer:BaseEstimator
                                       ) -> tuple[pd.DataFrame, BaseEstimator]:
    """
      Dans cette méthode on peut choisire un transformer pour la standardisation parmis:
        - StandardScaler
        - MinMaxScaler
        - MaxAbsScaler 
    """
    preprocessor:BaseEstimator = ColumnTransformer(
      [
        ("scaler", transformer, columns),
      ],
      verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    new_df:pd.DataFrame = preprocessor.fit_transform(self.df) # type: ignore
    return (new_df, transformer)
  
  def _set_the_df_of_features(self, target:str) -> pd.DataFrame:
    features:list[str] = self._get_features(target=target)
    if len(features) > 0:
      df:pd.DataFrame = self.df[features]
      return df
    else:
      raise Exception(f"La liste des caractéristiques de votre jeu de données est vide.")
    
  # reflechir si ici on ajoute pas le jeu de donnée traité après encodage"  
  def fit(self, target:str) -> object:
    df_of_features:pd.DataFrame = self._set_the_df_of_features(target=target)
    df_of_target:pd.DataFrame = self.df[target]
    # Penser stratégie d'implémentation de train test split
    X_train, X_test, y_train, y_test = train_test_split(
      df_of_features, df_of_target, test_size=0.4, random_state=42
    )  
  
  def update_dataframe(self, new_df:pd.DataFrame) -> None:
    self.df = new_df
