"""
Utilitaires simples pour gestion des enums - correction rapide
"""
import json
from enum import Enum
from datetime import datetime


def safe_enum_value(value):
    """Convertit un enum en string de façon sécurisée"""
    if hasattr(value, 'value'):
        return value.value
    return str(value)


def safe_dict_conversion(data):
    """Convertit récursivement les enums en strings dans un dict"""
    if isinstance(data, dict):
        return {k: safe_dict_conversion(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [safe_dict_conversion(item) for item in data]
    elif hasattr(data, 'value'):  # Enum
        return data.value
    elif isinstance(data, datetime):
        return data.isoformat()
    else:
        return data


class SimpleJSONEncoder(json.JSONEncoder):
    """Encodeur JSON simple pour enums et dates"""
    def default(self, obj):
        if hasattr(obj, 'value'):  # Enum
            return obj.value
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def safe_json_dumps(data, **kwargs):
    """JSON dumps qui gère les enums automatiquement"""
    # D'abord convertir le dict
    clean_data = safe_dict_conversion(data)
    return json.dumps(clean_data, ensure_ascii=False, **kwargs)