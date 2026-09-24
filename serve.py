import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "sam3_api.server:app",
        host=os.getenv("SAM3_HOST", "0.0.0.0"),
        port=int(os.getenv("SAM3_PORT", "8000")),
        workers=1,
    )
