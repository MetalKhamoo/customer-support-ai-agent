import pandas as pd
import os
import re

FILE = "data/golden/apple_support_golden.csv"

if not os.path.exists(FILE):
    print(f"File not found: {FILE}")
    exit()

df = pd.read_csv(FILE)

print("\n======================================")
print(" AppleSupport Golden Set")
print("======================================")

print(f"\nTotal examples in CSV: {len(df)}")


def classify_intent(text):
    text = str(text).lower()

    # Battery
    if any(x in text for x in [
        "battery", "battery life", "battery drain",
        "battery health", "dies quickly", "power"
    ]):
        return "battery_power"

    # Charging
    if any(x in text for x in [
        "charger", "charging", "charge", "charging port",
        "charging cable", "cable", "won't charge"
    ]):
        return "charging_power"

    # Account / iCloud
    if any(x in text for x in [
        "icloud", "apple id", "appleid", "password",
        "sign in", "login", "account", "phishing"
    ]):
        return "account_icloud"

    # Billing
    if any(x in text for x in [
        "payment", "charged", "charge me", "refund",
        "billing", "invoice", "subscription", "money",
        "cost", "price"
    ]):
        return "billing_payment"

    # Camera / photos
    if any(x in text for x in [
        "camera", "photos", "photo", "pictures",
        "picture", "videos", "video"
    ]):
        return "camera_photos"

    # Calls / connectivity
    if any(x in text for x in [
        "call", "calls", "wifi", "wi-fi", "bluetooth",
        "connect", "connection", "network", "cellular",
        "headphones connect", "car stereo"
    ]):
        return "calls_connectivity"

    # Display
    if any(x in text for x in [
        "screen", "display", "brightness", "touchscreen",
        "true tone", "screen crack", "screen not"
    ]):
        return "display_screen"

    # Audio / media
    if any(x in text for x in [
        "music", "sound", "audio", "itunes", "podcast",
        "headphones", "voicemail", "apple music",
        "spotify"
    ]):
        return "audio_media"

    # Apps
    if any(x in text for x in [
        "app", "apps", "app store", "application",
        "pandora", "netflix", "youtube", "hulu",
        "espn"
    ]):
        return "apps"

    # Software update
    if any(x in text for x in [
        "update", "ios", "software update", "ios 11"
    ]):
        return "software_update"

    # Performance
    if any(x in text for x in [
        "freeze", "freezing", "frozen", "lag",
        "laggy", "slow", "crash", "crashing",
        "restart", "reboot", "hang", "unresponsive"
    ]):
        return "device_performance"

    return "other_unknown"


def classify_action(text, intent):
    text = str(text).lower()

    # Cases that should normally reach a human
    escalation_terms = [
        "refund",
        "charged twice",
        "data breach",
        "someone using my",
        "phishing",
        "replacement",
        "replace",
        "broken",
        "warranty",
        "can't activate",
        "cannot activate",
        "locked",
        "account hacked",
        "security",
        "invoice",
        "unauthorized"
    ]

    if any(x in text for x in escalation_terms):
        return "ESCALATE"

    if intent in [
        "account_icloud",
        "billing_payment"
    ]:
        if any(x in text for x in [
            "can't", "cannot", "unable",
            "not working", "problem", "error"
        ]):
            return "ESCALATE"

    return "AUTO"


# ============================================================
# APPLY AUTOMATIC LABELS
# ============================================================

df["intent"] = df["customer_message"].apply(classify_intent)

df["expected_action"] = df.apply(
    lambda row: classify_action(
        row["customer_message"],
        row["intent"]
    ),
    axis=1
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(FILE, index=False)


# ============================================================
# RESULTS
# ============================================================

print("\n======================================")
print(" Golden set labelled successfully!")
print("======================================")

print("\nIntent distribution:")
print(df["intent"].value_counts())

print("\nAction distribution:")
print(df["expected_action"].value_counts())

print("\nMissing intents:",
      df["intent"].isna().sum())

print("Missing actions:",
      df["expected_action"].isna().sum())

print(f"\nSaved to: {FILE}")