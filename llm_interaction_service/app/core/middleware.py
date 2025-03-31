from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def setup_cors(app: FastAPI):
    origins = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:3010",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:8221",
        "http://localhost:8762",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app