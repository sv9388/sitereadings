import os
import yaml

SETTINGS_FILEPATH = os.path.dirname(os.path.realpath(__file__)) + "/data/settings.yaml"

def get_all_settings():

    if not os.path.exists(SETTINGS_FILEPATH):
        return {}

    with open(SETTINGS_FILEPATH, "r") as f:
        settings = yaml.load(f)

    return settings

def get_setting(setting):
    return get_all_settings()[setting]

def set_setting(setting, value):
    settings = get_all_settings()
    settings[setting] = value
    with open(SETTINGS_FILEPATH, "w") as f:
        yaml.dump(settings, f, default_flow_style=False)

