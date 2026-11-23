import sys
import os
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.insert(0, os.getcwd())

from fastapi.testclient import TestClient
from app.main import app

def main():
    print("=== STEP 1: STARTING APP & VERIFYING REAL SERVICES ===")
    with TestClient(app) as client:
        res = client.get("/health")
        print("\n--- GET /health Response ---")
        print(f"Status Code: {res.status_code}")
        print(res.json())
        assert res.status_code == 200, "Health check failed!"
        data = res.json()
        assert data["services"]["postgres"]["status"] == "healthy", "Postgres is not healthy!"
        assert data["services"]["qdrant"]["status"] == "healthy", "Qdrant is not healthy!"
        print("\n✓ SUCCESS: Postgres and Qdrant are both verified healthy and running against REAL hosted services!")

if __name__ == "__main__":
    main()
