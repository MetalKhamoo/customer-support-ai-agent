"""
Intent taxonomy for the AppleSupport customer-support agent.
"""

INTENTS = {
    "battery_power": {
        "description": "Battery drain, battery health, overheating, or poor battery performance."
    },

    "software_update": {
        "description": "Problems installing, downloading, or completing an iOS/software update."
    },

    "device_performance": {
        "description": "Device freezing, crashing, restarting, becoming slow, or becoming unresponsive."
    },

    "charging_power": {
        "description": "Problems with charging, chargers, cables, or the device not charging."
    },

    "apps": {
        "description": "Problems with applications, including crashes, downloads, updates, or app functionality."
    },

    "account_icloud": {
        "description": "Apple ID, iCloud, password, login, account access, or account-related problems."
    },

    "audio_media": {
        "description": "Apple Music, voicemail, sound, audio playback, or other media-related problems."
    },

    "display_screen": {
        "description": "Screen, display, touchscreen, brightness, or touch-response problems."
    },

    "billing_payment": {
        "description": "Unexpected charges, payments, subscriptions, refunds, or billing-related issues."
    },

    "other_unknown": {
        "description": "Requests that do not clearly fit another supported intent."
    }
}


def get_intent_names():
    """Return the list of supported intent names."""
    return list(INTENTS.keys())


def get_intent_descriptions():
    """Return intent names and descriptions."""
    return INTENTS