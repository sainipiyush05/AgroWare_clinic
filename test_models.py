import os
import tensorflow as tf

def test_models():
    models = ["agroware_model.h5", "agroware_model.keras", "final_model.keras", "best_final.keras", "final_phase1.keras", "best_phase1.keras"]
    
    for m in models:
        if os.path.exists(m):
            print(f"\nTesting {m} with tf.keras...")
            try:
                model = tf.keras.models.load_model(m, compile=False)
                print(f"✅ Success with tf.keras for {m}!")
            except Exception as e:
                print(f"❌ Failed with tf.keras: {e}")
                
            print(f"\nTesting {m} with standalone keras...")
            try:
                import keras
                model = keras.saving.load_model(m, compile=False)
                print(f"✅ Success with standalone keras for {m}!")
            except Exception as e:
                print(f"❌ Failed with standalone keras: {e}")

if __name__ == "__main__":
    test_models()
