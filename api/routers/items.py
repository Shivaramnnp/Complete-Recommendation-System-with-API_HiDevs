from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from api.core.database import get_db
from api.schemas.item import ItemCreate, ItemResponse
from api.repositories.item import item_repository

router = APIRouter(prefix="/items", tags=["Items"])

@router.post("/", response_model=ItemResponse, status_code=status.HTTP_201_CREATED)
def create_item(item_in: ItemCreate, db: Session = Depends(get_db)):
    """
    Add a new item (e.g. movie, product) to the catalog.
    """
    return item_repository.create(db, obj_data=item_in.model_dump())

@router.get("/", response_model=List[ItemResponse])
def read_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    List items from the catalog.
    """
    return item_repository.get_multi(db, skip=skip, limit=limit)

@router.get("/{item_id}", response_model=ItemResponse)
def read_item(item_id: int, db: Session = Depends(get_db)):
    """
    Get details of a specific item.
    """
    db_item = item_repository.get(db, item_id)
    if not db_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with ID {item_id} not found."
        )
    return db_item
