from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from predict_with_remedies import predict_with_remedies

app = FastAPI(
    title="AgroWare Disease Prediction API",
    description="API for predicting plant diseases and providing multilingual remedies.",
    version="1.0.0"
)

# Enable CORS so your frontend can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Active", "message": "Welcome to AgroWare Disease Prediction API. Send a POST request to /predict."}

@app.post("/predict")
async def predict_disease(
    image: UploadFile = File(...),
    language: str = Form("en"),
    crop: str = Form(None)
):
    temp_image_path = f"temp_{image.filename}"
    try:
        # Save uploaded image temporarily
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Run prediction
        result = predict_with_remedies(
            image_path=temp_image_path,
            language=language,
            crop_filter=crop
        )
        
        return {"success": True, "data": result}
        
    except Exception as e:
        return {"success": False, "error": str(e)}
        
    finally:
        # Clean up temporary image
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
