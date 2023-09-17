from fastapi import FastAPI

app = FastAPI()


@app.get("/") # all logic should be here, isolate gpu heavy functions to gpu_api
async def root():
    return {"message": "Hello Wdorld"}

