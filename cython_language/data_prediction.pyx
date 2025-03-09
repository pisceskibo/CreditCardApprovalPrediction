# Libraries
import pandas as pd
import joblib


# Predict for this application
cpdef make_prediction(profile_to_pred_prep: pd.DataFrame):
    cdef str model_path
    try:
        # Load model in local
        model_path = "saved_models/gradient_boosting/gradient_boosting_model.sav"
        model = joblib.load(model_path)
        print("✅ Model loaded successfully!")

        # Change to dataframe
        if isinstance(profile_to_pred_prep, pd.Series):
            profile_to_pred_prep = profile_to_pred_prep.to_frame().T

        # Predict
        probabilities = model.predict_proba(profile_to_pred_prep)
        prediction = model.predict(profile_to_pred_prep)

        print("📊 Probabilities (0, 1):", probabilities)
        print("🎯 Final Prediction:", prediction)

        return prediction
    except FileNotFoundError:
        print("❌ Model file not found! Please check the path")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred: {str(e)}")
        return None