from fastapi import FastAPI
from app.modules.brain_tumor import router as brain_tumor_router
from app.modules.pneumonia import router as pneumonia_router
from app.modules.blindness import router as blindness_router

app = FastAPI(title="MedAI API")

app.include_router(brain_tumor_router, prefix="/brain-tumor")
app.include_router(pneumonia_router, prefix="/pneumonia")
app.include_router(blindness_router, prefix="/blindness")

@app.get("/")
def root():
    return {"status": "active"}
