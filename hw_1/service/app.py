from fastapi import FastAPI, UploadFile, HTTPException
import joblib
from pydantic import BaseModel, ValidationError
from starlette.responses import StreamingResponse
import csv
from typing import List, Optional
import pandas as pd
import io
from preprocessing import preprocessing

model = joblib.load('model.pkl')

app = FastAPI()

class InputData(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: Optional[str] = None
    engine: Optional[str] = None
    max_power: Optional[str] = None
    torque: Optional[str] = None
    seats: Optional[float] = None

@app.post('/predict_item')
def predict_item(item: InputData):
    try:
        input_dict = item.dict()
        preprocessed_data = preprocessing(input_dict)
        prediction = model.predict(preprocessed_data)

        return {'prediction': float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Ошибка: {str(e)}')

@app.post('/predict_items')
def predict_items(file: UploadFile):
    try:
        if file.content_type != 'text/csv':
            raise HTTPException(status_code=400, detail='Формат файла должен быть csv')

        file_read = file.file.read().decode('utf-8')
        file_csv = io.StringIO(file_read)
        data = pd.read_csv(file_csv)
        data = data.where(pd.notnull(data), None)

        try:
            validated_data = [InputData.model_validate(row) for row in data.to_dict(orient='records')]
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f'Ошибка валидации: {str(e)}')

        data_dicts = [data.dict() for data in validated_data]
        df = pd.DataFrame(data_dicts)

        preprocessed_data = preprocessing(df)
        predictions = model.predict(preprocessed_data)
        df['selling_price_predicted'] = predictions

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        response = StreamingResponse(output, media_type='text/csv')
        response.headers['Content-Disposition'] = 'attachment; filename=predicted_prices.csv'
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Ошибка: {str(e)}')