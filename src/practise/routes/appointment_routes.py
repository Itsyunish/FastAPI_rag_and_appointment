from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from practise.models.user import UserInfo
from practise.utils.mail_utils import send_appointment_email
from practise.db.mongo import appointment_collection

router = APIRouter()

@router.post("/schedule_appointment/")
def schedule_appointment(info: UserInfo):
    try:
        metadata = {
            "name": info.name,
            "email": info.email,
            "appointment_date": info.appointment_date,
            "appointment_time": info.appointment_time
        }

        # email sendng has reached its limit
        response = send_appointment_email(info.email, info.name, info.appointment_date, info.appointment_time)
        metadata["mailtrap"] = {
            "success": response["success"],
            "mailtrap_message_id": response["message_ids"][0]
        }

        # Save to MongoDB
        appointment_collection.insert_one(metadata)

        return JSONResponse(content={
                "message": "Appointment scheduled successfully",
                "appointment_details": {
                    "name": info.name,
                    "email": info.email,
                    "appointment_date": info.appointment_date,
                    "appointment_time": info.appointment_time
            },
                "mailtrap_status": response
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Information saving failed: {str(e)}"})
