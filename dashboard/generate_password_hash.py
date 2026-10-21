"""
dashboard/generate_password_hash.py — one-time setup for the dashboard login.

Prompts for a password (hidden input, never echoed, never written to disk or
shell history), bcrypt-hashes it with streamlit-authenticator's Hasher, and
prints a complete auth config to paste into dashboard/auth_config.yaml
(gitignored). Only the hash is printed — never the password.

Usage (from the project root):
    python dashboard/generate_password_hash.py --username analyst
"""

import argparse
import getpass
import secrets
import sys

import streamlit_authenticator as stauth
import yaml

MIN_PASSWORD_LENGTH = 8
COOKIE_NAME = "edge_idps_dashboard_auth"
COOKIE_EXPIRY_DAYS = 1


def build_config(username: str, password_hash: str, first_name: str = "SOC",
                 last_name: str = "Analyst", email: str = "analyst@example.invalid",
                 cookie_key: str = None) -> dict:
    if not stauth.Hasher.is_hash(password_hash):
        raise ValueError("password_hash is not a bcrypt hash")
    return {
        "credentials": {"usernames": {username.strip().lower(): {
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "password": password_hash,
        }}},
        "cookie": {
            "name": COOKIE_NAME,
            # Signs the re-authentication cookie; a fresh random secret per setup
            "key": cookie_key or secrets.token_urlsafe(32),
            "expiry_days": COOKIE_EXPIRY_DAYS,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate a bcrypt-hashed dashboard login")
    parser.add_argument("--username", required=True, help="Login username (stored lowercase)")
    parser.add_argument("--first-name", default="SOC")
    parser.add_argument("--last-name", default="Analyst")
    args = parser.parse_args()

    password = getpass.getpass("Choose a dashboard password: ")
    if len(password) < MIN_PASSWORD_LENGTH:
        sys.exit(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if getpass.getpass("Confirm password: ") != password:
        sys.exit("Passwords did not match.")

    password_hash = stauth.Hasher.hash(password)
    assert stauth.Hasher.check_pw(password, password_hash)
    del password

    config = build_config(args.username, password_hash, args.first_name, args.last_name)
    print("\n# ---- paste everything below into dashboard/auth_config.yaml (gitignored) ----")
    print(yaml.safe_dump(config, sort_keys=False, default_flow_style=False), end="")
    print("# ---- end ----")


if __name__ == "__main__":
    main()
