from src.web.app import create_app
import uvicorn
app = create_app()
uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
