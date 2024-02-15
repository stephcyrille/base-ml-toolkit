import pytest
import pandas as pd
from mlbasetoolkit.src.superclassifier import ClassifierProcessorTool


def test_get_features_by_type_with_non_existing_dtype():
  """  
    Test qui vérifie si le type de colonne Pandas est valide 
  """ 
  df = pd.DataFrame([[1,2], [3,4]], columns=['baba', 'toto'])
  cls = ClassifierProcessorTool(df=df)
  with pytest.raises(Exception) as excinfo:
    type_of_feature='int32po'  
    cls.get_features_by_type(type_of_feature=type_of_feature, target='')
  assert str(excinfo.value) == f"Le type {type_of_feature} n'est pas un dtypes valide en Pandas." 

def test_get_features_by_type_without_target():
  """  
    Test qui vérifie si on a le bon retour lorsqu'on ne renseigne pas la colonne Y 
  """
  df = pd.DataFrame([[1,2], [3,4]], columns=['baba', 'toto'])
  cls = ClassifierProcessorTool(df=df)
  features = cls.get_features_by_type(type_of_feature='int64', target='')
  assert features == []

def test_get_features_by_type_with_non_existing_target():
  """  
    Test qui vérifie si on a le bon retour lorsqu'on renseigne une colonne Y non existante 
  """
  df = pd.DataFrame([[1,2], [3,4]], columns=['baba', 'toto'])
  cls = ClassifierProcessorTool(df=df)
  features = cls.get_features_by_type(type_of_feature='int64', target='abc')
  assert features == ['baba', 'toto'], 'Bad Output'

def test_get_features_by_type_with_the_good_parameters():
  """  
    Test qui vérifie si on a le bon retour lorsqu'on renseigne bien les valeurs
  """
  df = pd.DataFrame([[1,2], [3,4]], columns=['baba', 'toto'])
  cls = ClassifierProcessorTool(df=df)
  features = cls.get_features_by_type(type_of_feature='int64', target='baba')
  assert features == ['toto'], 'Bad output!' 
