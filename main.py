from typing import Optional

import pandas as pd
import ai_wonder as wonder

from pydantic import BaseModel, Field

class BuildingInputSchema(BaseModel):
    building_type: str
    sprinkler_system_present: str
    fire_safety_training_conducted: str
    nearest_fire_station_location: str
    types_of_nearby_buildings: str
    electrical_equipment_inspection_conducted: str
    gas_equipment_inspection_conducted: str
    recent_repair_replacement_history: str
    month: int
    building_age: int
    building_area_sqm: int
    building_height_m: int
    number_of_floors: int
    time_to_extinguish_min: Optional[int] =None
    response_time_min: Optional[int]=None
    number_of_fire_extinguishers: Optional[int]=None
    number_of_emergency_exits: int
    number_of_fire_alarms: int
    width_of_nearby_roads_m: int
    distance_to_nearby_buildings_m: int
    temperature_c: float
    humidity: float
    wind_speed_ms: float
    precipitation_mm: float


from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict(
    input: BuildingInputSchema
):
    df = pd.read_csv('hanoi_fire.csv')
    mean_time_to_extinguish_min = df['Time_to_Extinguish_(min)'].mean()
    mean_response_time_min = df['Response_Time_(min)'].mean()
    mean_number_of_fire_extinguishers = df['Number_of_Fire_Extinguishers'].mean()

    # Make datapoint from user input
    point = pd.DataFrame([{
        'Building_Type': input.building_type,
        'Sprinkler_System_Present': input.sprinkler_system_present,
        'Fire_Safety_Training_Conducted': input.fire_safety_training_conducted,
        'Nearest_Fire_Station_Location': input.nearest_fire_station_location,
        'Types_of_Nearby_Buildings': input.types_of_nearby_buildings,
        'Electrical_Equipment_Inspection_Conducted': input.electrical_equipment_inspection_conducted,
        'Gas_Equipment_Inspection_Conducted': input.gas_equipment_inspection_conducted,
        'Recent_Repair_Replacement_History': input.recent_repair_replacement_history,
        'Month': input.month,
        'Building_Age': input.building_age,
        'Building_Area_(sqm)': input.building_area_sqm,
        'Building_Height_(m)': input.building_height_m,
        'Number_of_Floors': input.number_of_floors,
        'Time_to_Extinguish_(min)': mean_time_to_extinguish_min,
        'Response_Time_(min)': mean_response_time_min,
        'Number_of_Fire_Extinguishers': mean_number_of_fire_extinguishers,
        'Number_of_Emergency_Exits': input.number_of_emergency_exits,
        'Number_of_Fire_Alarms': input.number_of_fire_alarms,
        'Width_of_Nearby_Roads_(m)': input.width_of_nearby_roads_m,
        'Distance_to_Nearby_Buildings_(m)': input.distance_to_nearby_buildings_m,
        'Temperature_(_C)': input.temperature_c,
        'Humidity_(%)': input.humidity,
        'Wind_Speed_(m_s)': input.wind_speed_ms,
        'Precipitation_(mm)': input.precipitation_mm,
    }])

    
    state = wonder.load_state('hanoi_fire_state.pkl')
    model = wonder.input_piped_model(state)
    prediction = str(model.predict_proba(point)[0])
    print(f"Prediction of **{state.target}** is **{prediction}**.")
    importances = pd.DataFrame(wonder.local_explanations(state, point), columns=["Feature", "Value", "Importance"])
    return prediction