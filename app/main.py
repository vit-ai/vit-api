from fastapi import FastAPI

app = FastAPI()


@app.get("/") # all logic should be here, isolate gpu heavy functions to 
async def root():
    import ultralytics
    return {"message": "Hello World"}
# https://github.com/ultralytics/yolov5/discussions/5872