```Mermaid
classDiagram
   Experiment --> Dataset
   Experiment --> RunManager
   Experiment --> Augmenter
   Experiment --> FeatureSet
   Experiment --> Configuration
   RunManager --> Model
   Experiment --> Reporter
   Dataset <|-- Dataset_csv
   Model <|-- Model_svm
   Model <|-- Model_xgb
   Model <|-- Model_mlp
   Model <|-- Model_svr
   Model <|-- Model_xgr 
   Model <|-- Model_mlp_reg
   FeatureSet <|-- Opensmile_set
   FeatureSet <|-- Spectraloader
   FeatureSet <|-- MLD_set
   FeatureSet <|-- OpenXbow
   FeatureSet <|-- Wav2Vec
   FeatureSet <|-- Trill

   class Experiment{
       + Report reports
       + Dataframe df_test
       + Dataframe df_train

       + load_datasets()
       + fill_train_and_tests()
       + plot_distribution()
       + augment_train()
       + extract_feats()
       + init_runmanager()
       + run()
    }
    class FeatureSet{
        + pd.Dataframe df
        + extract()
    }
    class RunManager{
        + epochs
        + runs

    }
    class Augmenter{
        + augment()
    }
    class Model{        
        + train()
        + predict()
        + predict_sample()
        + store()
        + load()
    }
    class Dataset{
        + Dataframe df_test
        + Dataframe df_train
        + load()
        + split()        
        + prepare_labels()
    }
    class Model_svm{   
        +float C
    }
    class Model_xgb{   
    }
    class Model_mlp{   
        + loss_function
        + optimizer
        + learning_rate
    }

    