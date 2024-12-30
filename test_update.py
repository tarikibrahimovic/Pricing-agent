from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception(
        "SUPABASE_URL i SUPABASE_KEY environment varijable moraju biti postavljene."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

sneaker_id = 2
data = {"count": 1, "reward": 5}

try:
    response = supabase.table("Product").update(data).eq("id", sneaker_id).execute()

    print("Update successful:", response.data)

except Exception as e:
    print(f"Error updating product: {str(e)}")
