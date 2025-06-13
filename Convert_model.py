from tensorflow.keras.models import load_model

# Carrega o modelo .keras
model = load_model("trained_model.keras")

# Salva como SavedModel (pasta)
model.export("saved_model/")