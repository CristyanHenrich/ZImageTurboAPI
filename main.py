import uvicorn
import os

if __name__ == "__main__":
    # Convenience script to run the server
    # You can also run directly with: uvicorn app.main:app --host 0.0.0.0 --port 8000
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
