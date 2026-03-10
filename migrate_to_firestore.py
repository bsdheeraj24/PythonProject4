"""
One-time migration script: moves users.json and attendance CSVs to Firebase Firestore.

Usage:
  1. Set the FIREBASE_CREDENTIALS environment variable to your service account JSON string:
       export FIREBASE_CREDENTIALS='{"type":"service_account", ...}'
  2. Run:  python migrate_to_firestore.py
"""

import os
import json
import csv

import firebase_admin
from firebase_admin import credentials, firestore
from werkzeug.security import generate_password_hash


def init_firebase():
    raw = os.environ.get("FIREBASE_CREDENTIALS", "{}")
    cred = credentials.Certificate(json.loads(raw))
    firebase_admin.initialize_app(cred)
    return firestore.client()


def migrate_users(db):
    users_file = "users.json"
    if not os.path.exists(users_file):
        print("users.json not found, skipping user migration.")
        return

    with open(users_file, "r") as f:
        users = json.load(f)

    for username, data in users.items():
        db.collection("users").document(username).set({
            "password_hash": generate_password_hash(data["password"]),
            "role": data["role"],
        })
        print(f"  Migrated user: {username} (role={data['role']})")

    print(f"Users migration complete ({len(users)} users).\n")


def migrate_attendance(db):
    attend_dir = "attendance"
    if not os.path.isdir(attend_dir):
        print("attendance/ directory not found, skipping attendance migration.")
        return

    all_persons = []

    for filename in sorted(os.listdir(attend_dir)):
        if not filename.endswith(".csv"):
            continue

        person_name = filename.replace(".csv", "")
        all_persons.append(person_name)
        filepath = os.path.join(attend_dir, filename)

        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            batch = db.batch()
            count = 0

            for row in reader:
                ref = db.collection("attendance").document()
                batch.set(ref, {
                    "name": row["Name"],
                    "date": row["Date"],
                    "time": row["Time"],
                    "status": row["Status"],
                    "timestamp": firestore.SERVER_TIMESTAMP,
                })
                count += 1

                if count % 500 == 0:
                    batch.commit()
                    batch = db.batch()

            batch.commit()
            print(f"  Migrated {count} records for: {person_name}")

    # Write metadata document
    db.collection("metadata").document("attendance_persons").set({
        "names": all_persons
    })
    print(f"Attendance migration complete ({len(all_persons)} persons).\n")


if __name__ == "__main__":
    print("=== Firebase Firestore Migration ===\n")
    db = init_firebase()

    print("[1/2] Migrating users...")
    migrate_users(db)

    print("[2/2] Migrating attendance records...")
    migrate_attendance(db)

    print("=== Migration finished ===")
