from flask import Flask, request, abort
from cryptography.fernet import Fernet
import json
import os
import secrets

user_api_keys_file = "user_api_keys.json"
encryption_key = b"Eex_B_maR0ApEbYvbbSDmf4ULs5vhCKZZ_-3fcrNPc0="


def generate_api_key():
    return secrets.token_hex(16)


def load_user_api_keys():
    with open(user_api_keys_file, "rb") as keys_file:
        encrypted_data = keys_file.read()

    cipher = Fernet(encryption_key)
    decrypted_data = cipher.decrypt(encrypted_data)
    api_keys = json.loads(decrypted_data.decode())
    return api_keys


def save_user_api_keys(api_keys):
    json_data = json.dumps(api_keys).encode()
    cipher = Fernet(encryption_key)
    encrypted_data = cipher.encrypt(json_data)

    with open(user_api_keys_file, "wb") as keys_file:
        keys_file.write(encrypted_data)
