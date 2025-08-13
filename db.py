from motor.motor_asyncio import AsyncIOMotorClient

client = AsyncIOMotorClient("mongodb://root:example@13.204.31.17:27017/?authSource=admin")
db = client["ai_teaching_app"]
 