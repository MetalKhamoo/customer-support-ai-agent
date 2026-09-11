import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


DATA_FILE = "data/golden/apple_support_golden.csv"


def classify_intent(text):
    text = str(text).lower()

    if any(x in text for x in [
        "battery", "battery life", "battery drain",
        "battery health", "dies quickly"
    ]):
        return "battery_power"

    if any(x in text for x in [
        "charger", "charging", "charge", "charging port",
        "charging cable", "cable", "won't charge"
    ]):
        return "charging_power"

    if any(x in text for x in [
        "icloud", "apple id", "appleid", "password",
        "sign in", "login", "account", "phishing"
    ]):
        return "account_icloud"

    if any(x in text for x in [
        "payment", "charged", "refund", "billing",
        "invoice", "subscription", "money", "cost", "price"
    ]):
        return "billing_payment"

    if any(x in text for x in [
        "camera", "photos", "photo", "pictures",
        "picture", "videos", "video"
    ]):
        return "camera_photos"

    if any(x in text for x in [
        "call", "calls", "wifi", "wi-fi", "bluetooth",
        "connect", "connection", "network", "cellular"
    ]):
        return "calls_connectivity"

    if any(x in text for x in [
        "screen", "display", "brightness", "touchscreen",
        "true tone"
    ]):
        return "display_screen"

    if any(x in text for x in [
        "music", "sound", "audio", "itunes", "podcast",
        "headphones", "voicemail", "apple music", "spotify"
    ]):
        return "audio_media"

    if any(x in text for x in [
        "app", "apps", "app store", "application",
        "pandora", "netflix", "youtube", "hulu", "espn"
    ]):
        return "apps"

    if any(x in text for x in [
        "update", "ios", "software update"
    ]):
        return "software_update"

    if any(x in text for x in [
        "freeze", "freezing", "frozen", "lag",
        "laggy", "slow", "crash", "crashing",
        "restart", "reboot", "hang", "unresponsive"
    ]):
        return "device_performance"

    return "other_unknown"


df = pd.read_csv(DATA_FILE)

y_true = df["intent"]
y_pred = df["customer_message"].apply(classify_intent)

accuracy = accuracy_score(y_true, y_pred)

print("\n======================================")
print(" Rule-Based Intent Baseline")
print("======================================")

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)