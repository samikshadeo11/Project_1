import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def train(X, y):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=20,  # imbalance handling
        eval_metric='aucpr'
    )
    model.fit(X, y)
    joblib.dump(model, 'models/factoryguard_model.pkl')
    return model
