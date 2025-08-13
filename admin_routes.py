from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from passlib.hash import bcrypt
from db import db
import requests

RECAPTCHA_SECRET_KEY = "6LckAoYrAAAAAO-StEFXdETM0q3OxE8l_ARvGoA0"

router = APIRouter()

@router.post("/register_admin/")
async def register_admin(name: str = Form(...), email: str = Form(...), password: str = Form(...)):
    existing = await db.admins.find_one({"email": email})
    if existing:
        return JSONResponse(status_code=400, content={"message": "Email already registered."})
    hashed_password = bcrypt.hash(password)
    admin = {"name": name, "email": email, "password": hashed_password}
    await db.admins.insert_one(admin)
    return {"message": f"Admin {name} registered successfully!"}

@router.post("/login_admin/")  # ✅ Only this one for login
async def login_admin(
    email: str = Form(...),
    password: str = Form(...),
    g_recaptcha_response: str = Form(alias="g-recaptcha-response")
):
    # Step 1: Verify reCAPTCHA
    captcha_verify = requests.post(
        "https://www.google.com/recaptcha/api/siteverify",
        data={
            "secret": RECAPTCHA_SECRET_KEY,
            "response": g_recaptcha_response
        }
    )
    captcha_result = captcha_verify.json()
    if not captcha_result.get("success"):
        return JSONResponse(status_code=400, content={"message": "reCAPTCHA verification failed."})

    # Step 2: Validate login credentials
    user = await db.admins.find_one({"email": email})
    if user and bcrypt.verify(password, user["password"]):
        return {
            "message": f"Welcome back, {user['name']}!",
            "name": user["name"],
            "email": user["email"]
        }
    return JSONResponse(status_code=400, content={"message": "Invalid credentials."})
