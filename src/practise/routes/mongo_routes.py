from fastapi import APIRouter, HTTPException
from practise.db.mongo import metadata_collection
from practise.utils.file_utils import format_size_mb
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/get_metadata/")
def get_metadata():
    try:
        records = list(metadata_collection.find({}, {"_id": 0}))
        for data in records:
            data["size"] = format_size_mb(data["size_bytes"])
        return JSONResponse(content={"metadata": records})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching metadata: {str(e)}")
