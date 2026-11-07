class AuthService:
    def login(self, username, password):
        if username == "admin":
            return True
        return False

def helper_function():
    return "hello"