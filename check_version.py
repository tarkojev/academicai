import sys

def verify_python():
    required_version = (3, 11)
    current_version = sys.version_info

    if current_version.major != required_version[0] or current_version.minor != required_version[1]:
        print(f"Python version is not 3.11")
        print(f"Expected: {required_version[0]}.{required_version[1]}")
        print(f"Detected: {current_version.major}.{current_version.minor}")
        sys.exit(1)
    else:
        print(f"Python {sys.version.split()[0]} version confirmed")

if __name__ == "__main__":
    verify_python()
