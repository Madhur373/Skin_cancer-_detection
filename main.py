from gui import app
import uvicorn

if __name__ == "__main__":
    # Run FastAPI app
    uvicorn.run("gui:app", host="0.0.0.0", port=8080)
