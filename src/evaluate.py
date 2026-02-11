from sklearn.metrics import average_precision_score

def evaluate(model, X_test, y_test):
    preds = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, preds)
    print("PR-AUC:", pr_auc)
