from preprocessor import PreProcessor
from feature_engineer import FeatureEngineer
from models import ModelPipeline

def main():
    preprocessor = PreProcessor()
    df = preprocessor.read_data()
    
    fe = FeatureEngineer()
    X_train, X_val, X_test, Y_train, Y_val, Y_test = fe.run(df)
    
    mp = ModelPipeline(X_train, X_val, X_test, Y_train, Y_val, Y_test)
    mp.fit_all_models()
    results = mp.get_results()
    results.to_csv('outputs/model_results.csv')
    mp.plot_roc_curve()
    
if __name__ == "__main__":
    main()