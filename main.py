"""Minimal runnable script for the FinalPRoject repository."""

def main() -> None:
    """Run a tiny interactive demo so the project is runnable out of the box."""
    print("Welcome to FinalPRoject!")
    name = input("Enter your name (or press Enter to skip): ").strip()
    if name:
        print(f"Hello, {name}! Thanks for trying the script.")
    else:
        print("Hello! Thanks for trying the script.")
    print("This repository now has a runnable Python entry point.")


if __name__ == "__main__":
    main()
