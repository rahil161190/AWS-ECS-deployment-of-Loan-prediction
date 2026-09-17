import pickle

with open("classifier.pkl", "rb") as f:
    bundle = pickle.load(f)

print("=== Bundle keys ===")
print(list(bundle.keys()))

print("\n=== scaler.feature_names_in_ ===")
print(list(bundle["scaler"].feature_names_in_))

print("\n=== feature_order (model input) ===")
print(bundle["feature_order"])

print("\n=== target_encoder.cols ===")
print(bundle["target_encoder"].cols)

print("\n=== target_encoder type ===")
print(type(bundle["target_encoder"]))

print("\n=== model.n_features_in_ ===")
print(bundle["model"].n_features_in_)

print("\n=== model.feature_names_in_ (if available) ===")
print(getattr(bundle["model"], "feature_names_in_", "not set"))