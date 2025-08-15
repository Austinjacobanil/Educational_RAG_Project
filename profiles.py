import json, os

PROFILE_FILE = "profiles.json"

def load_profiles():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_profiles(profiles):
    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)

def get_profile(name, default_profile):
    profiles = load_profiles()
    return profiles.get(name, default_profile.copy())

def update_profile(name, profile):
    profiles = load_profiles()
    profiles[name] = profile
    save_profiles(profiles)
