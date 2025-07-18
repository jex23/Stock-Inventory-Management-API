# .env (create this file in your project root)
# DB_USER=james23
# DB_PASSWORD=J@mes2410117
# DB_HOST=179.61.246.136
# DB_PORT=3306
# DB_NAME=stock_inventory

from flask import Flask, request, jsonify, abort
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv
import os
import pymysql
from sqlalchemy import or_
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
import hashlib
from datetime import datetime, timedelta

# Load environment variables from .env file
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

app = Flask(__name__)

# Configure CORS for frontend origins
CORS(app, origins=[
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "https://yourdomain.com"  # Add your production domain
])

app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "your-secret-key-here")

db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    position = db.Column(
        db.Enum("admin", "owner", "supervisor", "manager", "staff"),
        nullable=False,
    )
    contract = db.Column(db.Text, nullable=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)  # Increased size for secure hashes
    pin_hash = db.Column(db.String(255), nullable=False)       # Increased size for secure hashes
    login_attempt = db.Column(db.Integer, nullable=True, default=0)
    status = db.Column(
        db.Enum("enabled", "disabled"), nullable=True, default="enabled"
    )
    created_at = db.Column(
        db.DateTime, server_default=db.func.current_timestamp()
    )
    updated_at = db.Column(
        db.DateTime,
        server_default=db.func.current_timestamp(),
        server_onupdate=db.func.current_timestamp(),
    )
    reset_token = db.Column(db.String(255), nullable=True)
    reset_token_expires = db.Column(db.DateTime, nullable=True)
    is_first_login = db.Column(db.Boolean, nullable=True, default=True)
    last_login = db.Column(db.DateTime, nullable=True)
    session_token = db.Column(db.String(255), nullable=True)

    def to_dict(self, include_sensitive=False):
        data = {
            "id": self.id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "position": self.position,
            "contract": self.contract,
            "username": self.username,
            "email": self.email,
            "login_attempt": self.login_attempt,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_first_login": self.is_first_login,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }
        
        # Only include sensitive data if explicitly requested (for admin purposes)
        if include_sensitive:
            data.update({
                "reset_token": self.reset_token,
                "reset_token_expires": (
                    self.reset_token_expires.isoformat() if self.reset_token_expires else None
                ),
                "session_token": self.session_token,
            })
        
        return data

# Utility functions
def generate_session_token():
    """Generate a secure session token"""
    return secrets.token_urlsafe(32)

def hash_password(password):
    """Hash a password securely"""
    return generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)

def verify_password(password_hash, password):
    """Verify a password against its hash"""
    return check_password_hash(password_hash, password)

def require_role(*allowed_roles):
    """Utility to enforce role-based access for modifying users"""
    role = request.headers.get("X-User-Role")
    if role not in allowed_roles:
        abort(403, description=f"Forbidden: requires role in {allowed_roles}")

def require_auth():
    """Utility to require authentication via session token"""
    token = request.headers.get("Authorization")
    if not token:
        abort(401, description="Missing authorization token")
    
    # Remove 'Bearer ' prefix if present
    if token.startswith("Bearer "):
        token = token[7:]
    
    user = User.query.filter_by(session_token=token).first()
    if not user:
        abort(401, description="Invalid or expired token")
    
    return user

# User Management Endpoints
@app.route("/users", methods=["GET"])
def get_users():
    """Get all users (admin/owner only)"""
    require_role("admin", "owner")
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    search = request.args.get('search', '')
    
    query = User.query
    
    if search:
        query = query.filter(
            or_(
                User.first_name.contains(search),
                User.last_name.contains(search),
                User.username.contains(search),
                User.email.contains(search)
            )
        )
    
    users = query.paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        "users": [user.to_dict() for user in users.items],
        "total": users.total,
        "pages": users.pages,
        "current_page": page,
        "per_page": per_page
    })

@app.route("/users/<int:user_id>", methods=["GET"])
def get_user(user_id):
    """Get a specific user"""
    require_role("admin", "owner", "supervisor", "manager")
    user = User.query.get_or_404(user_id)
    return jsonify(user.to_dict())

@app.route("/users", methods=["POST"])
def create_user():
    """Create a new user (admin/owner only)"""
    require_role("admin", "owner")
    data = request.get_json() or {}
    
    required_fields = [
        "first_name", "last_name", "position",
        "username", "email", "password", "pin"
    ]
    
    for field in required_fields:
        if field not in data or not data[field]:
            abort(400, description=f"Missing or empty field: {field}")
    
    # Validate position
    valid_positions = ["admin", "owner", "supervisor", "manager", "staff"]
    if data["position"] not in valid_positions:
        abort(400, description=f"Invalid position. Must be one of: {valid_positions}")
    
    # Check for existing username or email
    existing = User.query.filter(
        or_(User.username == data["username"], User.email == data["email"]) 
    ).first()
    if existing:
        abort(400, description="Username or email already exists")
    
    # Validate password strength
    password = data["password"]
    if len(password) < 6:
        abort(400, description="Password must be at least 6 characters long")
    
    # Validate PIN
    pin = data["pin"]
    if not pin.isdigit() or len(pin) < 4:
        abort(400, description="PIN must be at least 4 digits")
    
    # Hash passwords securely
    password_hash = hash_password(password)
    pin_hash = hash_password(pin)
    
    user = User(
        first_name=data["first_name"].strip(),
        last_name=data["last_name"].strip(),
        position=data["position"],
        contract=data.get("contract", "").strip() or None,
        username=data["username"].strip(),
        email=data["email"].strip().lower(),
        password_hash=password_hash,
        pin_hash=pin_hash,
        login_attempt=0,
        status=data.get("status", "enabled"),
        is_first_login=data.get("is_first_login", True)
    )
    
    try:
        db.session.add(user)
        db.session.commit()
        return jsonify(user.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Failed to create user: {str(e)}")
        abort(500, description="Failed to create user")

@app.route("/users/<int:user_id>", methods=["PUT"])
def update_user(user_id):
    """Update a user (admin/owner only)"""
    require_role("admin", "owner")
    user = User.query.get_or_404(user_id)
    data = request.get_json() or {}
    
    # Check for unique constraints
    if "username" in data and data["username"] != user.username:
        if User.query.filter_by(username=data["username"]).first():
            abort(400, description="Username already exists")
    
    if "email" in data and data["email"] != user.email:
        if User.query.filter_by(email=data["email"]).first():
            abort(400, description="Email already exists")
    
    # Update allowed fields
    allowed_fields = [
        "first_name", "last_name", "position", "contract",
        "username", "email", "status"
    ]
    
    for key, value in data.items():
        if key in allowed_fields and hasattr(user, key):
            if key in ["first_name", "last_name", "username"]:
                setattr(user, key, value.strip() if value else value)
            elif key == "email":
                setattr(user, key, value.strip().lower() if value else value)
            else:
                setattr(user, key, value)
    
    try:
        db.session.commit()
        return jsonify(user.to_dict())
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Failed to update user: {str(e)}")
        abort(500, description="Failed to update user")

@app.route("/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    """Delete a user (admin/owner only)"""
    require_role("admin", "owner")
    user = User.query.get_or_404(user_id)
    
    try:
        db.session.delete(user)
        db.session.commit()
        return "", 204
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Failed to delete user: {str(e)}")
        abort(500, description="Failed to delete user")

# Authentication Endpoints
@app.route("/login", methods=["POST"])
def login():
    """Login with username/email and password"""
    data = request.get_json() or {}
    
    # Accept either username, email, or identifier
    username = data.get("username")
    email = data.get("email")
    identifier = data.get("identifier")  # Backward compatibility
    password = data.get("password")
    
    # Validate input
    if not password:
        abort(400, description="Missing password")
    
    if not (username or email or identifier):
        abort(400, description="Missing login credentials. Provide username, email, or identifier")
    
    # Build query to find user
    query_conditions = []
    
    if username:
        query_conditions.append(User.username == username)
    if email:
        query_conditions.append(User.email == email.lower())
    if identifier:
        query_conditions.append(
            or_(User.username == identifier, User.email == identifier.lower())
        )
    
    # Find user
    user = User.query.filter(or_(*query_conditions)).first()
    
    if not user:
        abort(401, description="Invalid credentials")
    
    if user.status == "disabled":
        abort(403, description="Account is disabled")
    
    # Verify password
    if not verify_password(user.password_hash, password):
        user.login_attempt = (user.login_attempt or 0) + 1
        
        # Lock account after 5 failed attempts
        if user.login_attempt >= 5:
            user.status = "disabled"
            db.session.commit()
            abort(403, description="Account locked due to multiple failed login attempts")
        
        db.session.commit()
        abort(401, description="Invalid credentials")
    
    # Successful login
    user.login_attempt = 0
    user.last_login = datetime.utcnow()
    user.session_token = generate_session_token()
    db.session.commit()
    
    return jsonify({
        "message": "Login successful",
        "user": user.to_dict(),
        "token": user.session_token,
        "expires_in": 3600  # 1 hour
    })

@app.route("/logout", methods=["POST"])
def logout():
    """Logout and invalidate session token"""
    user = require_auth()
    user.session_token = None
    db.session.commit()
    
    return jsonify({"message": "Logout successful"})

@app.route("/verify-pin", methods=["POST"])
def verify_pin():
    """Verify user PIN for sensitive operations"""
    user = require_auth()
    data = request.get_json() or {}
    pin = data.get("pin")
    
    if not pin:
        abort(400, description="Missing PIN")
    
    if not verify_password(user.pin_hash, pin):
        abort(401, description="Invalid PIN")
    
    return jsonify({"message": "PIN verified successfully"})

# Password Management Endpoints
@app.route("/users/<int:user_id>/change-password", methods=["POST"])
def change_password(user_id):
    """Change password (first login or with proper authentication)"""
    data = request.get_json() or {}
    new_password = data.get("password")
    new_pin = data.get("pin")
    
    if not new_password:
        abort(400, description="Missing password")
    
    user = User.query.get_or_404(user_id)
    
    # Allow password change on first login or with proper authentication
    if not user.is_first_login:
        # Require authentication for subsequent password changes
        authenticated_user = require_auth()
        if authenticated_user.id != user_id:
            require_role("admin", "owner")  # Only admin/owner can change other users' passwords
    
    # Validate password strength
    if len(new_password) < 6:
        abort(400, description="Password must be at least 6 characters long")
    
    # Hash and update password
    user.password_hash = hash_password(new_password)
    
    # Update PIN if provided
    if new_pin:
        if not new_pin.isdigit() or len(new_pin) < 4:
            abort(400, description="PIN must be at least 4 digits")
        user.pin_hash = hash_password(new_pin)
    
    user.is_first_login = False
    db.session.commit()
    
    return jsonify({
        "message": "Password updated successfully",
        "user": user.to_dict()
    })

@app.route("/forgot-password", methods=["POST"])
def forgot_password():
    """Request password reset"""
    data = request.get_json() or {}
    email = data.get("email")
    
    if not email:
        abort(400, description="Missing email")
    
    user = User.query.filter_by(email=email.lower()).first()
    if not user:
        # Don't reveal if email exists
        return jsonify({"message": "If the email exists, a reset link has been sent"})
    
    # Generate reset token
    reset_token = secrets.token_urlsafe(32)
    user.reset_token = reset_token
    user.reset_token_expires = datetime.utcnow() + timedelta(hours=1)
    db.session.commit()
    
    # TODO: Send email with reset link
    # send_reset_email(user.email, reset_token)
    
    return jsonify({
        "message": "If the email exists, a reset link has been sent",
        "reset_token": reset_token  # Remove this in production
    })

@app.route("/reset-password", methods=["POST"])
def reset_password():
    """Reset password with token"""
    data = request.get_json() or {}
    token = data.get("token")
    new_password = data.get("password")
    
    if not token or not new_password:
        abort(400, description="Missing token or password")
    
    user = User.query.filter_by(reset_token=token).first()
    if not user or not user.reset_token_expires or user.reset_token_expires < datetime.utcnow():
        abort(400, description="Invalid or expired reset token")
    
    # Validate password strength
    if len(new_password) < 6:
        abort(400, description="Password must be at least 6 characters long")
    
    # Update password and clear reset token
    user.password_hash = hash_password(new_password)
    user.reset_token = None
    user.reset_token_expires = None
    user.login_attempt = 0  # Reset failed attempts
    user.status = "enabled"  # Reactivate account if disabled
    db.session.commit()
    
    return jsonify({"message": "Password reset successful"})

# User Profile Endpoints
@app.route("/profile", methods=["GET"])
def get_profile():
    """Get current user profile"""
    user = require_auth()
    return jsonify(user.to_dict())

@app.route("/profile", methods=["PUT"])
def update_profile():
    """Update current user profile"""
    user = require_auth()
    data = request.get_json() or {}
    
    # Users can only update certain fields
    allowed_fields = ["first_name", "last_name", "email"]
    
    # Check email uniqueness
    if "email" in data and data["email"].lower() != user.email:
        if User.query.filter_by(email=data["email"].lower()).first():
            abort(400, description="Email already exists")
    
    for key, value in data.items():
        if key in allowed_fields and hasattr(user, key):
            if key == "email":
                setattr(user, key, value.strip().lower())
            else:
                setattr(user, key, value.strip() if value else value)
    
    try:
        db.session.commit()
        return jsonify(user.to_dict())
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Failed to update profile: {str(e)}")
        abort(500, description="Failed to update profile")

# Health Check
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    })

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "error": "Bad Request",
        "description": error.description,
        "status_code": 400
    }), 400

@app.errorhandler(401)
def unauthorized(error):
    return jsonify({
        "error": "Unauthorized",
        "description": error.description,
        "status_code": 401
    }), 401

@app.errorhandler(403)
def forbidden(error):
    return jsonify({
        "error": "Forbidden",
        "description": error.description,
        "status_code": 403
    }), 403

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not Found",
        "description": "Resource not found",
        "status_code": 404
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal Server Error",
        "description": "An internal error occurred",
        "status_code": 500
    }), 500

if __name__ == "__main__":
    # Create tables if they don't exist
    with app.app_context():
        db.create_all()
        print("Database tables created/verified successfully!")
    
    print("Starting Flask application...")
    app.run(host="0.0.0.0", port=4567, debug=True)