from cryptography.fernet import Fernet
import json
import os
import secrets

user_api_keys_file = "user_api_keys.json"
encryption_key = b"Eex_B_maR0ApEbYvbbSDmf4ULs5vhCKZZ_-3fcrNPc0="


def generate_api_key():
    return secrets.token_hex(16)


def generate_encrypted_keys_file():
    if os.path.exists(user_api_keys_file):
        print(f"{user_api_keys_file} already exists. Skipping generation.")
        return

    user_api_keys = {
        "admin": generate_api_key(),
    }
    print(f'admin key : {user_api_keys["admin"]}')

    cipher = Fernet(encryption_key)
    json_data = json.dumps(user_api_keys).encode()
    encrypted_data = cipher.encrypt(json_data)

    with open(user_api_keys_file, "wb") as keys_file:
        keys_file.write(encrypted_data)

    print(f"Generated and encrypted {user_api_keys_file} successfully.")


if __name__ == "__main__":
    generate_encrypted_keys_file()
