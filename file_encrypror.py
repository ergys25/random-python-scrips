from cryptography.fernet import Fernet
import os

def generate_key():
    return Fernet.generate_key()

def write_key_to_file(key, filename):
    with open(filename, "wb") as key_file:
        key_file.write(key)

def load_key_from_file(filename):
    with open(filename, "rb") as key_file:
        return key_file.read()

def encrypt_file(key, input_file, output_file):
    f = Fernet(key)
    with open(input_file, "rb") as file:
        file_data = file.read()
    encrypted_data = f.encrypt(file_data)
    with open(output_file, "wb") as file:
        file.write(encrypted_data)

def decrypt_file(key, input_file, output_file):
    f = Fernet(key)
    with open(input_file, "rb") as file:
        encrypted_data = file.read()
    decrypted_data = f.decrypt(encrypted_data)
    with open(output_file, "wb") as file:
        file.write(decrypted_data)

if __name__ == "__main__":
    # Generate or load key
    key_filename = "encryption_key.key"
    if not os.path.exists(key_filename):
        key = generate_key()
        write_key_to_file(key, key_filename)
    else:
        key = load_key_from_file(key_filename)

    # Input and output files
    input_file = "/path/to/input/file"
    encrypted_output_file = "/path/to/encrypted/output/file"
    decrypted_output_file = "/path/to/decrypted/output/file"

    # Encrypt the file
    encrypt_file(key, input_file, encrypted_output_file)
    print("File encrypted.")

    # Decrypt the file
    decrypt_file(key, encrypted_output_file, decrypted_output_file)
    print("File decrypted.")
