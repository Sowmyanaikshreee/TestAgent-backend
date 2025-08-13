# from pymongo import MongoClient

# # Local MongoDB connection
# local_client = MongoClient("mongodb://localhost:27017")

# # Remote MongoDB connection
# remote_client = MongoClient("mongodb://root:example@13.235.78.101:27017/?authSource=admin")

# # List all databases except system ones
# db_names = [db for db in local_client.list_database_names() if db not in ("admin", "local", "config")]

# for db_name in db_names:
#     local_db = local_client[db_name]
#     remote_db = remote_client[db_name]

#     for collection_name in local_db.list_collection_names():
#         local_collection = local_db[collection_name]
#         remote_collection = remote_db[collection_name]

#         documents = list(local_collection.find())

#         if documents:
#             # Optional: clear existing data on remote
#             remote_collection.delete_many({})
#             remote_collection.insert_many(documents)
#             print(f"✅ Migrated {len(documents)} documents from {db_name}.{collection_name}")
#         else:
#             print(f"ℹ️ No documents in {db_name}.{collection_name}")

# print("🎉 Migration complete.")
from pymongo import MongoClient

# Connect to remote MongoDB
client = MongoClient("mongodb://root:example@13.204.31.17:27017/?authSource=admin")

# List all non-system databases
db_names = [db for db in client.list_database_names() if db not in ("admin", "local", "config")]

for db_name in db_names:
    db = client[db_name]
    collection_names = db.list_collection_names()

    for collection_name in collection_names:
        result = db[collection_name].delete_many({})
        print(f"🗑 Deleted {result.deleted_count} documents from {db_name}.{collection_name}")

print("✅ All data deleted from remote MongoDB.")
