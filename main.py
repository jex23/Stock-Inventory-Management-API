# main.py
from __future__ import annotations
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, Text, Enum, TIMESTAMP, func, text, DECIMAL, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel, EmailStr
from decimal import Decimal
import hashlib
import enum
import os
import smtplib
import secrets
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from urllib.parse import quote_plus
from typing import Dict, Any, List, Optional


# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("🔧 Environment variables loaded from .env file")
except ImportError:
    print("📦 python-dotenv not found. Using system environment variables only.")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")

# Database configuration with environment variable support
import os
from urllib.parse import quote_plus

def get_database_url():
    """Get database URL from environment variables or fallback to default"""
    
    # Try to get individual components from environment variables
    db_user = os.getenv("DB_USER", "james23")
    db_password = os.getenv("DB_PASSWORD", "J@mes2410117")
    db_host = os.getenv("DB_HOST", "179.61.246.136")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "stock_inventory")
    
    # Check for complete DATABASE_URL environment variable first
    if os.getenv("DATABASE_URL"):
        return os.getenv("DATABASE_URL")
    
    # URL encode the password to handle special characters like @
    encoded_password = quote_plus(db_password)
    
    # Construct the database URL
    database_url = f"mysql+pymysql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
    
    return database_url

# Get the database URL
DATABASE_URL = get_database_url()

print(f"🔌 Connecting to database: {DATABASE_URL.replace(quote_plus('J@mes2410117'), '*****')}")  # Hide password in logs

try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    
    # Test the connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    
    DB_CONNECTED = True
    print("✅ Database connection successful!")
    
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    print("🔄 Running without database connection...")
    DB_CONNECTED = False
    engine = None
    SessionLocal = None
    Base = None

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production-temp-key-12345")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
MAX_LOGIN_ATTEMPTS = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))

# Email configuration
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_FROM = os.getenv("EMAIL_FROM", EMAIL_USERNAME)
EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "Stock Inventory System")
RESET_TOKEN_EXPIRE_MINUTES = int(os.getenv("RESET_TOKEN_EXPIRE_MINUTES", "30"))

# Check email configuration
if not EMAIL_USERNAME or not EMAIL_PASSWORD:
    print("⚠️  WARNING: Email configuration missing! Set EMAIL_USERNAME and EMAIL_PASSWORD for password reset functionality.")
    EMAIL_CONFIGURED = False
else:
    EMAIL_CONFIGURED = True
    print("📧 Email configuration loaded successfully!")

# Warn about default secret key
if SECRET_KEY == "your-secret-key-change-this-in-production-temp-key-12345":
    print("⚠️  WARNING: Using default SECRET_KEY! Set SECRET_KEY environment variable for production.")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Enums
class UserPosition(str, enum.Enum):
    admin = "admin"
    owner = "owner"
    supervisor = "supervisor"
    manager = "manager"
    staff = "staff"

class UserStatus(str, enum.Enum):
    enabled = "enabled"
    disabled = "disabled"

# Enums for Stock
class StockUnit(str, enum.Enum):
    kg = "kg"
    g = "g"
    mg = "mg"
    lb = "lb"
    oz = "oz"
    l = "l"
    ml = "ml"
    pcs = "pcs"
    box = "box"
    pack = "pack"
    sack = "sack"
    bottle = "bottle"
    can = "can"
    jar = "jar"
    roll = "roll"

class StockCategory(str, enum.Enum):
    finished_product = "finished product"
    raw_material = "raw material"

class ProcessManagement(Base):
    __tablename__ = "process_management"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    process_id_batch = Column(String(50), nullable=True)
    stock_id = Column(Integer, ForeignKey('stock.id'), nullable=False)
    users_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    finished_product_id = Column(Integer, ForeignKey('product.id'), nullable=False)
    archive = Column(Integer, default=0)
    manufactured_date = Column(TIMESTAMP, default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    stock = relationship("Stock", foreign_keys=[stock_id])
    user = relationship("User", foreign_keys=[users_id])
    finished_product = relationship("Product", foreign_keys=[finished_product_id])

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    position = Column(Enum(UserPosition), nullable=False)
    contract = Column(Text, nullable=True)
    username = Column(String(100), nullable=False, unique=True)
    email = Column(String(150), nullable=False, unique=True)
    password_hash = Column(String(64), nullable=False)
    pin_hash = Column(String(64), nullable=False)
    login_attempt = Column(Integer, default=0)
    status = Column(Enum(UserStatus), default=UserStatus.enabled)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, default=func.current_timestamp(), onupdate=func.current_timestamp())
    reset_token = Column(String(255), nullable=True)
    reset_token_expires = Column(TIMESTAMP, nullable=True)
    is_first_login = Column(Integer, default=1, nullable=True)  # Using Integer for MySQL boolean compatibility
    
    # Relationships
    stocks = relationship("Stock", back_populates="user")

class FinishedProductCategory(Base):
    __tablename__ = "finished_product_category"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)

class Product(Base):
    __tablename__ = "product"
    
    id = Column(Integer, primary_key=True, autoincrement=True)  # Auto-increment primary key
    name = Column(String(50), nullable=False)
    price = Column(DECIMAL(10, 2), nullable=False)
    unit = Column(Enum(StockUnit), nullable=False, default=StockUnit.pcs)  # ADDED
    quantity = Column(DECIMAL(10, 2), nullable=False, default=0.00)       # ADDED
    
    # Relationships
    stocks = relationship("Stock", back_populates="product")

class Supplier(Base):
    __tablename__ = "supplier"
    
    id = Column(Integer, primary_key=True, autoincrement=True)  # Updated to auto-increment
    name = Column(String(50), nullable=False)
    contact_num = Column(String(15), nullable=False)
    email_add = Column(String(50), nullable=False)
    address = Column(String(50), nullable=False)
    
    # Relationships
    stocks = relationship("Stock", back_populates="supplier")

class Stock(Base):
    __tablename__ = "stock"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    batch = Column(String(50), nullable=False)
    piece = Column(Integer, nullable=False)
    # REMOVED: quantity and unit - now in Product table
    
    category = Column(Enum(StockCategory, values_callable=lambda obj: [e.value for e in obj]), nullable=False)
    
    archive = Column(Integer, default=0)
    product_id = Column(Integer, ForeignKey('product.id'), nullable=False)
    supplier_id = Column(Integer, ForeignKey('supplier.id'), nullable=False)
    users_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    used = Column(Integer, default=0)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    product = relationship("Product", back_populates="stocks")
    supplier = relationship("Supplier", back_populates="stocks")
    user = relationship("User", back_populates="stocks")

# Pydantic Models for Users
class UserCreate(BaseModel):
    first_name: str
    last_name: str
    position: UserPosition
    contract: Optional[str] = None
    username: str
    email: EmailStr
    password: str
    pin: str

class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    position: Optional[UserPosition] = None
    contract: Optional[str] = None
    username: Optional[str] = None
    email: Optional[EmailStr] = None

class UserLogin(BaseModel):
    username_or_email: str  # Can be either username or email
    password: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

class UpdateUserStatusRequest(BaseModel):
    status: UserStatus

class ChangePasswordFirstLogin(BaseModel):
    username_or_email: str
    current_password: str
    new_password: str

class UserResponse(BaseModel):
    id: int
    first_name: str
    last_name: str
    position: UserPosition
    contract: Optional[str] = None
    username: str
    email: str
    status: UserStatus
    login_attempt: int
    created_at: datetime
    updated_at: datetime
    is_first_login: Optional[bool] = None
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse
    is_first_login: bool = False

class TokenData(BaseModel):
    username: Optional[str] = None

# Pydantic Models for New Tables
class FinishedProductCategoryCreate(BaseModel):
    name: str

class FinishedProductCategoryUpdate(BaseModel):
    name: Optional[str] = None

class FinishedProductCategoryResponse(BaseModel):
    id: int
    name: str
    
    class Config:
        from_attributes = True

class ProductCreate(BaseModel):
    name: str
    price: Decimal
    unit: StockUnit = StockUnit.pcs  # ADDED
    quantity: Decimal = Decimal('0.00')  # ADDED

class ProductUpdate(BaseModel):
    name: Optional[str] = None
    price: Optional[Decimal] = None
    unit: Optional[StockUnit] = None  # ADDED
    quantity: Optional[Decimal] = None  # ADDED

class ProductResponse(BaseModel):
    id: int
    name: str
    price: Decimal
    unit: StockUnit  # ADDED
    quantity: Decimal  # ADDED
    
    class Config:
        from_attributes = True

# Pydantic Models for Stock
class StockCreate(BaseModel):
    batch: str
    piece: int
    # REMOVED: quantity and unit - now in Product table
    category: StockCategory
    product_id: int
    supplier_id: int
    users_id: int

class StockUpdate(BaseModel):
    batch: Optional[str] = None
    piece: Optional[int] = None
    # REMOVED: quantity and unit - now in Product table
    category: Optional[StockCategory] = None
    product_id: Optional[int] = None
    supplier_id: Optional[int] = None
    users_id: Optional[int] = None
    archive: Optional[bool] = None
    used: Optional[bool] = None

class StockResponse(BaseModel):
    id: int
    batch: str
    piece: int
    # REMOVED: quantity and unit - now retrieved from Product
    category: StockCategory
    archive: bool
    product_id: int
    supplier_id: int
    users_id: int
    used: bool
    created_at: datetime
    updated_at: datetime
    
    # Related data - now includes product unit and quantity
    product_name: Optional[str] = None
    product_unit: Optional[StockUnit] = None    # ADDED: From Product table
    product_quantity: Optional[Decimal] = None  # ADDED: From Product table
    supplier_name: Optional[str] = None
    user_name: Optional[str] = None
    
    class Config:
        from_attributes = True

class BatchStockItem(BaseModel):
    piece: int
    category: StockCategory
    product_id: int
    supplier_id: int

class BatchStockCreate(BaseModel):
    items: List[BatchStockItem]
    users_id: Optional[int] = None  # Will be auto-populated from current user

class BatchResponse(BaseModel):
    batch_number: str
    items_created: int
    items: List['StockResponse']

class BatchSummary(BaseModel):
    batch_number: str
    total_items: int
    total_product_quantity: Decimal  # CHANGED: Updated field name
    categories: Dict[str, int]
    created_at: datetime
    user_name: str
    items: List[StockResponse]

class BatchSummaryResponse(BaseModel):
    batch_number: str
    total_items: int
    total_product_quantity: Decimal  # CHANGED: Updated field name
    categories: Dict[str, int]
    created_at: datetime
    user_name: str
    
    class Config:
        from_attributes = True

class BatchArchiveRequest(BaseModel):
    archive: bool

BatchResponse.update_forward_refs()
BatchSummary.update_forward_refs()


class ProcessManagementCreate(BaseModel):
    stock_id: int
    users_id: int
    finished_product_id: int

class ProcessManagementUpdate(BaseModel):
    stock_id: Optional[int] = None
    users_id: Optional[int] = None
    finished_product_id: Optional[int] = None
    archive: Optional[bool] = None

class ProcessManagementResponse(BaseModel):
    id: int
    process_id_batch: Optional[str] = None
    stock_id: int
    users_id: int
    finished_product_id: int
    archive: bool
    manufactured_date: datetime
    updated_at: datetime
    
    # Related data
    stock_batch: Optional[str] = None
    user_name: Optional[str] = None
    finished_product_name: Optional[str] = None
    
    class Config:
        from_attributes = True

class BatchProcessItem(BaseModel):
    stock_id: int
    finished_product_id: int

class BatchProcessCreate(BaseModel):
    items: List[BatchProcessItem]
    users_id: Optional[int] = None  # Will be auto-populated from current user

class ProcessBatchResponse(BaseModel):
    process_batch_number: str
    items_created: int
    items: List['ProcessManagementResponse']

class ProcessBatchSummary(BaseModel):
    process_batch_number: str
    total_items: int
    manufactured_date: datetime
    user_name: str
    items: List[ProcessManagementResponse]

class ProcessBatchSummaryResponse(BaseModel):
    process_batch_number: str
    total_items: int
    manufactured_date: datetime
    user_name: str
    
    class Config:
        from_attributes = True

class ProcessBatchArchiveRequest(BaseModel):
    archive: bool

# Update forward references
ProcessBatchResponse.update_forward_refs()
ProcessBatchSummary.update_forward_refs()



# Updated Supplier Models - Removed id from Create model
class SupplierCreate(BaseModel):
    name: str
    contact_num: str
    email_add: str
    address: str

class SupplierUpdate(BaseModel):
    name: Optional[str] = None
    contact_num: Optional[str] = None
    email_add: Optional[str] = None
    address: Optional[str] = None

class SupplierResponse(BaseModel):
    id: int
    name: str
    contact_num: str
    email_add: str
    address: str
    
    class Config:
        from_attributes = True

# Stock Statistics Response Model
class StockStatsResponse(BaseModel):
    total_stocks: int
    active_stocks: int
    archived_stocks: int
    used_stocks: int
    finished_products: int
    raw_materials: int
    total_product_quantity: Decimal = Decimal('0.00')

# Add batch-related utility functions
def generate_batch_number(db: Session) -> str:
    """Generate next batch number in format batch-XXXXXX"""
    try:
        # Get the latest batch number from database
        latest_stock = db.query(Stock).order_by(Stock.created_at.desc()).first()
        
        if not latest_stock or not latest_stock.batch:
            # If no stocks exist or no batch number, start with batch-000001
            return "batch-000001"
        
        # Extract number from latest batch (e.g., "batch-000045" -> 45)
        latest_batch = latest_stock.batch
        if latest_batch.startswith("batch-"):
            try:
                batch_num = int(latest_batch.split("-")[1])
                next_num = batch_num + 1
                return f"batch-{next_num:06d}"
            except (IndexError, ValueError):
                # If batch format is unexpected, start from 000001
                return "batch-000001"
        else:
            # If latest batch doesn't follow our format, start from 000001
            return "batch-000001"
            
    except Exception as e:
        print(f"⚠️ Error generating batch number: {e}")
        # Fallback to timestamp-based batch number
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"batch-{timestamp}"

def generate_process_batch_number(db: Session) -> str:
    """Generate next process batch number in format process-XXXXXX"""
    try:
        # Get the latest process batch number from database
        latest_process = db.query(ProcessManagement)\
            .filter(ProcessManagement.process_id_batch.isnot(None))\
            .order_by(ProcessManagement.manufactured_date.desc())\
            .first()
        
        if not latest_process or not latest_process.process_id_batch:
            # If no processes exist or no batch number, start with process-000001
            return "process-000001"
        
        # Extract number from latest batch (e.g., "process-000045" -> 45)
        latest_batch = latest_process.process_id_batch
        if latest_batch.startswith("process-"):
            try:
                batch_num = int(latest_batch.split("-")[1])
                next_num = batch_num + 1
                return f"process-{next_num:06d}"
            except (IndexError, ValueError):
                # If batch format is unexpected, start from 000001
                return "process-000001"
        else:
            # If latest batch doesn't follow our format, start from 000001
            return "process-000001"
            
    except Exception as e:
        print(f"⚠️ Error generating process batch number: {e}")
        # Fallback to timestamp-based batch number
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"process-{timestamp}"

def get_process_batch_items(db: Session, process_batch_number: str) -> List[ProcessManagement]:
    """Get all process management items for a specific batch"""
    return db.query(ProcessManagement)\
        .filter(ProcessManagement.process_id_batch == process_batch_number)\
        .all()

def get_batch_stocks(db: Session, batch_number: str) -> List[Stock]:
    """Get all stocks for a specific batch"""
    return db.query(Stock).filter(Stock.batch == batch_number).all()

def get_all_batches(db: Session) -> List[Dict[str, Any]]:
    """Get all unique batches with summary information"""
    try:
        # Get all unique batch numbers with their creation dates
        batches_query = db.query(Stock.batch, func.min(Stock.created_at).label('created_at'))\
            .group_by(Stock.batch)\
            .order_by(func.min(Stock.created_at).desc())\
            .all()
        
        batches = []
        for batch_number, created_at in batches_query:
            # Get stocks for this batch
            batch_stocks = get_batch_stocks(db, batch_number)
            
            if batch_stocks:
                # Calculate summary statistics
                total_items = len(batch_stocks)
                
                # FIXED: Calculate total product quantity from related products
                total_product_quantity = Decimal('0.00')
                for stock in batch_stocks:
                    product = db.query(Product).filter(Product.id == stock.product_id).first()
                    if product:
                        # Multiply product quantity by piece count
                        total_product_quantity += product.quantity * stock.piece
                
                # Count by category
                categories = {}
                for stock in batch_stocks:
                    category = stock.category.value if hasattr(stock.category, 'value') else str(stock.category)
                    categories[category] = categories.get(category, 0) + 1
                
                # Get user name from first stock
                first_stock = batch_stocks[0]
                user = db.query(User).filter(User.id == first_stock.users_id).first()
                user_name = f"{user.first_name} {user.last_name}" if user else "Unknown User"
                
                batches.append({
                    "batch_number": batch_number,
                    "total_items": total_items,
                    "total_product_quantity": total_product_quantity,  # FIXED
                    "categories": categories,
                    "created_at": created_at,
                    "user_name": user_name
                })
        
        return batches
        
    except Exception as e:
        print(f"⚠️ Error getting batches: {e}")
        return []

# Utility Functions
def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return hash_password(plain_password) == hashed_password

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def generate_otp() -> str:
    """Generate a 6-digit OTP"""
    return str(random.randint(100000, 999999))

def send_otp_email(email: str, otp: str, user_name: str) -> bool:
    """Send OTP email to user"""
    if not EMAIL_CONFIGURED:
        print("❌ Email not configured. Cannot send OTP.")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = f"{EMAIL_FROM_NAME} <{EMAIL_FROM}>"
        msg['To'] = email
        msg['Subject'] = "Password Reset OTP - Stock Inventory System"
        
        # Email body
        body = f"""
        Hello {user_name},

        You have requested to reset your password for the Stock Inventory System.

        Your One-Time Password (OTP) is: {otp}

        This OTP will expire in {RESET_TOKEN_EXPIRE_MINUTES} minutes.

        If you did not request this password reset, please ignore this email or contact your administrator.

        Best regards,
        Stock Inventory System Team
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to Gmail SMTP server
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()  # Enable TLS encryption
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        
        # Send email
        text = msg.as_string()
        server.sendmail(EMAIL_FROM, email, text)
        server.quit()
        
        print(f"📧 OTP email sent successfully to {email}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False



# Database Dependency
def get_db():
    if not DB_CONNECTED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available. Please check database configuration."
        )
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication Dependencies
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    return user

async def get_admin_or_owner_user(current_user: User = Depends(get_current_user)):
    """Dependency to ensure only admin or owner can access certain endpoints"""
    if current_user.position not in [UserPosition.admin, UserPosition.owner]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin or owner can perform this action"
        )
    return current_user

# FastAPI App
app = FastAPI(
    title="Stock Inventory API Documentation",
    description="A secure user authentication system with role-based access control and password reset for Stock Inventory Management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",    # Vite dev server
        "http://127.0.0.1:5173",   # Alternative localhost
        "http://localhost:3000",    # Create React App (if you switch)
        "http://127.0.0.1:3000",   # Alternative localhost for CRA
        # Add your production domain here when deploying
        # "https://your-domain.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

print("🌐 CORS middleware configured for cross-origin requests")

# Create tables
if DB_CONNECTED:
    Base.metadata.create_all(bind=engine)

# =============================================================================
# USER AUTHENTICATION ENDPOINTS
# =============================================================================

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Register a new user. Available to all authenticated users.
    """
    print(f"🔍 Creating user with contract: '{user_data.contract}'")
    
    # Check if username already exists
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = hash_password(user_data.password)
    hashed_pin = hash_password(user_data.pin)
    
    db_user = User(
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        position=user_data.position,
        contract=user_data.contract,  # This will handle None/NULL properly
        username=user_data.username,
        email=user_data.email,
        password_hash=hashed_password,
        pin_hash=hashed_pin,
        is_first_login=1  # Set first login flag for new users
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    print(f"✅ User created with contract: '{db_user.contract}'")
    
    return db_user

@app.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update user information. Available to all authenticated users.
    """
    # Find the user to update
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if username is being updated and if it already exists
    if user_update.username and user_update.username != user.username:
        existing_user = db.query(User).filter(User.username == user_update.username).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
    
    # Check if email is being updated and if it already exists
    if user_update.email and user_update.email != user.email:
        existing_user = db.query(User).filter(User.email == user_update.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already exists"
            )
    
    # Update only the fields that are provided (not None)
    update_data = user_update.dict(exclude_unset=True)
    
    print(f"🔍 Updating user {user_id} with data: {update_data}")
    
    for field, value in update_data.items():
        if hasattr(user, field):
            print(f"   Setting {field} = '{value}'")
            setattr(user, field, value)
    
    # Commit the changes
    db.commit()
    db.refresh(user)
    
    print(f"✅ User {user_id} updated. Contract is now: '{user.contract}'")
    
    return user

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific user by ID. Available to all authenticated users.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user

@app.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a user. Available to all authenticated users.
    Note: This permanently removes the user from the database.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Prevent self-deletion
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    db.delete(user)
    db.commit()
    
    return {"message": f"User {user.username} has been deleted successfully"}

@app.post("/login", response_model=Token)
async def login_user(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """
    Login user with username/email and password. Account will be disabled after 5 failed attempts.
    Accepts either username or email address for login.
    """
    username_or_email = user_credentials.username_or_email.strip()
    
    # Determine if input is email or username
    is_email = "@" in username_or_email
    
    # Query user by email or username
    if is_email:
        user = db.query(User).filter(User.email == username_or_email).first()
    else:
        user = db.query(User).filter(User.username == username_or_email).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Check if user status is enabled
    if user.status == UserStatus.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled. Contact administrator."
        )
    
    # Check if account is locked due to too many attempts
    if user.login_attempt >= MAX_LOGIN_ATTEMPTS:
        # Disable the account
        user.status = UserStatus.disabled
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account locked due to too many failed login attempts. Contact administrator."
        )
    
    # Verify password
    if not verify_password(user_credentials.password, user.password_hash):
        # Increment login attempts
        user.login_attempt += 1
        db.commit()
        
        remaining_attempts = MAX_LOGIN_ATTEMPTS - user.login_attempt
        if remaining_attempts <= 0:
            user.status = UserStatus.disabled
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account locked due to too many failed login attempts."
            )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid credentials. {remaining_attempts} attempts remaining."
        )
    
    # Reset login attempts on successful login
    user.login_attempt = 0
    db.commit()
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user,
        "is_first_login": bool(user.is_first_login) if user.is_first_login is not None else False
    }

@app.post("/change-password-first-login", response_model=Token)
async def change_password_first_login(
    password_data: ChangePasswordFirstLogin,
    db: Session = Depends(get_db)
):
    """
    Change password for first-time login users.
    This endpoint allows users to change their password on first login without requiring authentication.
    """
    username_or_email = password_data.username_or_email.strip()
    
    # Determine if input is email or username
    is_email = "@" in username_or_email
    
    # Query user by email or username
    if is_email:
        user = db.query(User).filter(User.email == username_or_email).first()
    else:
        user = db.query(User).filter(User.username == username_or_email).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Check if user status is enabled
    if user.status == UserStatus.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled. Contact administrator."
        )
    
    # Check if this is actually a first login
    if user.is_first_login is None or user.is_first_login == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password change for first login is not required for this account."
        )
    
    # Verify current password
    if not verify_password(password_data.current_password, user.password_hash):
        # Increment login attempts
        user.login_attempt += 1
        db.commit()
        
        remaining_attempts = MAX_LOGIN_ATTEMPTS - user.login_attempt
        if remaining_attempts <= 0:
            user.status = UserStatus.disabled
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account locked due to too many failed attempts."
            )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid current password. {remaining_attempts} attempts remaining."
        )
    
    # Update password and mark first login as complete
    new_password_hash = hash_password(password_data.new_password)
    user.password_hash = new_password_hash
    user.is_first_login = 0  # Mark first login as complete
    user.login_attempt = 0  # Reset login attempts
    db.commit()
    
    # Create access token for immediate login
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user,
        "is_first_login": False
    }

@app.post("/forgot-password")
async def forgot_password(
    request: ForgotPasswordRequest,
    db: Session = Depends(get_db)
):
    """
    Send OTP to user's email for password reset.
    """
    if not EMAIL_CONFIGURED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Email service not configured. Contact administrator."
        )
    
    # Find user by email
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        # Don't reveal if email exists or not for security
        return {"message": "If the email exists in our system, an OTP has been sent."}
    
    # Check if user is enabled
    if user.status == UserStatus.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled. Contact administrator."
        )
    
    # Generate OTP and expiry time
    otp = generate_otp()
    expires_at = datetime.utcnow() + timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES)
    
    # Store OTP in database
    user.reset_token = otp
    user.reset_token_expires = expires_at
    db.commit()
    
    # Send OTP email
    user_name = f"{user.first_name} {user.last_name}"
    email_sent = send_otp_email(user.email, otp, user_name)
    
    if not email_sent:
        # Clear the OTP if email failed
        user.reset_token = None
        user.reset_token_expires = None
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send OTP email. Please try again later."
        )
    
    return {"message": "If the email exists in our system, an OTP has been sent."}

@app.post("/reset-password")
async def reset_password(
    request: ResetPasswordRequest,
    db: Session = Depends(get_db)
):
    """
    Reset password using OTP.
    """
    # Find user by email
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email or OTP."
        )
    
    # Check if user has a reset token
    if not user.reset_token or not user.reset_token_expires:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No password reset request found. Please request a new OTP."
        )
    
    # Check if OTP is expired
    if datetime.utcnow() > user.reset_token_expires:
        # Clear expired token
        user.reset_token = None 
        user.reset_token_expires = None
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OTP has expired. Please request a new one."
        )
    
    # Verify OTP
    if user.reset_token != request.otp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid OTP."
        )
    
    # Update password
    new_password_hash = hash_password(request.new_password)
    user.password_hash = new_password_hash
    
    # Clear reset token
    user.reset_token = None
    user.reset_token_expires = None
    
    # Reset login attempts and enable account if disabled
    user.login_attempt = 0
    if user.status == UserStatus.disabled:
        user.status = UserStatus.enabled
    
    db.commit()
    
    return {"message": "Password reset successfully."}

@app.get("/users/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current authenticated user information"""
    return current_user

@app.get("/users", response_model=List[UserResponse])
async def get_all_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all users. Available to all authenticated users."""
    users = db.query(User).all()
    
    # Debug logging
    print(f"🔍 Found {len(users)} users")
    for user in users[:3]:  # Log first 3 users for debugging
        print(f"   User {user.id}: contract = '{user.contract}' (type: {type(user.contract)})")
    
    return users

@app.put("/users/{user_id}/status")
async def update_user_status(
    user_id: int,
    status_request: UpdateUserStatusRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update user status (enable/disable). Available to all authenticated users."""
    print(f"🔍 Updating user {user_id} status to: {status_request.status}")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Prevent disabling self
    if user.id == current_user.id and status_request.status == UserStatus.disabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot disable your own account"
        )
    
    old_status = user.status
    user.status = status_request.status
    
    # Reset login attempts when enabling account
    if status_request.status == UserStatus.enabled:
        user.login_attempt = 0
    
    db.commit()
    
    print(f"✅ User {user_id} status updated from {old_status} to {user.status}")
    
    return {"message": f"User status updated to {status_request.status.value}"}

@app.put("/users/{user_id}/reset-attempts")
async def reset_login_attempts(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Reset login attempts for a user. Available to all authenticated users."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.login_attempt = 0
    db.commit()
    
    return {"message": "Login attempts reset successfully"}

# =============================================================================
# FINISHED PRODUCT CATEGORY ENDPOINTS
# =============================================================================

@app.get("/categories", response_model=List[FinishedProductCategoryResponse])
async def get_all_categories(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all finished product categories. Available to all authenticated users."""
    categories = db.query(FinishedProductCategory).all()
    return categories

@app.get("/categories/{category_id}", response_model=FinishedProductCategoryResponse)
async def get_category_by_id(
    category_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific category by ID. Available to all authenticated users."""
    category = db.query(FinishedProductCategory).filter(FinishedProductCategory.id == category_id).first()
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found"
        )
    return category

@app.post("/categories", response_model=FinishedProductCategoryResponse, status_code=status.HTTP_201_CREATED)
async def create_category(
    category_data: FinishedProductCategoryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new finished product category with auto-increment ID. Available to all authenticated users."""
    # Check if category name already exists
    existing_category = db.query(FinishedProductCategory).filter(FinishedProductCategory.name == category_data.name).first()
    if existing_category:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Category name already exists"
        )
    
    # Create category (ID will be auto-assigned)
    db_category = FinishedProductCategory(name=category_data.name)
    db.add(db_category)
    db.commit()
    db.refresh(db_category)
    
    return db_category

@app.put("/categories/{category_id}", response_model=FinishedProductCategoryResponse)
async def update_category(
    category_id: int,
    category_update: FinishedProductCategoryUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a finished product category. Available to all authenticated users."""
    category = db.query(FinishedProductCategory).filter(FinishedProductCategory.id == category_id).first()
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found"
        )
    
    # Check if new name already exists (if name is being updated)
    if category_update.name and category_update.name != category.name:
        existing_category = db.query(FinishedProductCategory).filter(FinishedProductCategory.name == category_update.name).first()
        if existing_category:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Category name already exists"
            )
    
    # Update only the fields that are provided (not None)
    update_data = category_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        if hasattr(category, field):
            setattr(category, field, value)
    
    db.commit()
    db.refresh(category)
    
    return category

@app.delete("/categories/{category_id}")
async def delete_category(
    category_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a finished product category. Available to all authenticated users."""
    category = db.query(FinishedProductCategory).filter(FinishedProductCategory.id == category_id).first()
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found"
        )
    
    db.delete(category)
    db.commit()
    
    return {"message": f"Category '{category.name}' has been deleted successfully"}

# =============================================================================
# PRODUCT ENDPOINTS
# =============================================================================

@app.get("/products", response_model=List[ProductResponse])
async def get_all_products(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all products. Available to all authenticated users."""
    products = db.query(Product).all()
    return products

@app.get("/products/{product_id}", response_model=ProductResponse)
async def get_product_by_id(
    product_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific product by ID. Available to all authenticated users."""
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found"
        )
    return product

@app.post("/products", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
async def create_product(
    product_data: ProductCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new product with unit and quantity. Available to all authenticated users."""
    
    # Check if product name already exists
    existing_product_name = db.query(Product).filter(Product.name == product_data.name).first()
    if existing_product_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Product name already exists"
        )
    
    # Validate quantity
    if product_data.quantity < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Quantity cannot be negative"
        )
    
    # Create product with unit and quantity
    db_product = Product(
        name=product_data.name,
        price=product_data.price,
        unit=product_data.unit,      # ADDED
        quantity=product_data.quantity  # ADDED
    )
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    
    return db_product

# REPLACE the update_product function validation section:
    # Validate quantity if being updated
    if product_update.quantity is not None and product_update.quantity < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Quantity cannot be negative"
        )


@app.put("/products/{product_id}", response_model=ProductResponse)
async def update_product(
    product_id: int,
    product_update: ProductUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a product. Available to all authenticated users."""
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found"
        )
    
    # Check if new name already exists (if name is being updated)
    if product_update.name and product_update.name != product.name:
        existing_product = db.query(Product).filter(Product.name == product_update.name).first()
        if existing_product:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Product name already exists"
            )
    
    # Update only the fields that are provided (not None)
    update_data = product_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        if hasattr(product, field):
            setattr(product, field, value)
    
    db.commit()
    db.refresh(product)
    
    return product

@app.delete("/products/{product_id}")
async def delete_product(
    product_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a product. Available to all authenticated users."""
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found"
        )
    
    db.delete(product)
    db.commit()
    
    return {"message": f"Product '{product.name}' has been deleted successfully"}

# =============================================================================
# SUPPLIER ENDPOINTS (UPDATED WITH AUTO-INCREMENT)
# =============================================================================

@app.get("/suppliers", response_model=List[SupplierResponse])
async def get_all_suppliers(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all suppliers. Available to all authenticated users."""
    suppliers = db.query(Supplier).all()
    return suppliers

@app.get("/suppliers/{supplier_id}", response_model=SupplierResponse)
async def get_supplier_by_id(
    supplier_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific supplier by ID. Available to all authenticated users."""
    supplier = db.query(Supplier).filter(Supplier.id == supplier_id).first()
    if not supplier:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Supplier not found"
        )
    return supplier

@app.post("/suppliers", response_model=SupplierResponse, status_code=status.HTTP_201_CREATED)
async def create_supplier(
    supplier_data: SupplierCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new supplier with auto-increment ID. Available to all authenticated users."""
    
    # Check if supplier name already exists
    existing_supplier_name = db.query(Supplier).filter(Supplier.name == supplier_data.name).first()
    if existing_supplier_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Supplier name already exists"
        )
    
    # Check if email already exists
    existing_supplier_email = db.query(Supplier).filter(Supplier.email_add == supplier_data.email_add).first()
    if existing_supplier_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Supplier email already exists"
        )
    
    # Create supplier (ID will be auto-assigned)
    db_supplier = Supplier(
        name=supplier_data.name,
        contact_num=supplier_data.contact_num,
        email_add=supplier_data.email_add,
        address=supplier_data.address
    )
    db.add(db_supplier)
    db.commit()
    db.refresh(db_supplier)
    
    return db_supplier

@app.put("/suppliers/{supplier_id}", response_model=SupplierResponse)
async def update_supplier(
    supplier_id: int,
    supplier_update: SupplierUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a supplier. Available to all authenticated users."""
    supplier = db.query(Supplier).filter(Supplier.id == supplier_id).first()
    if not supplier:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Supplier not found"
        )
    
    # Check if new name already exists (if name is being updated)
    if supplier_update.name and supplier_update.name != supplier.name:
        existing_supplier = db.query(Supplier).filter(Supplier.name == supplier_update.name).first()
        if existing_supplier:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Supplier name already exists"
            )
    
    # Check if new email already exists (if email is being updated)
    if supplier_update.email_add and supplier_update.email_add != supplier.email_add:
        existing_supplier = db.query(Supplier).filter(Supplier.email_add == supplier_update.email_add).first()
        if existing_supplier:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Supplier email already exists"
            )
    
    # Update only the fields that are provided (not None)
    update_data = supplier_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        if hasattr(supplier, field):
            setattr(supplier, field, value)
    
    db.commit()
    db.refresh(supplier)
    
    return supplier

@app.delete("/suppliers/{supplier_id}")
async def delete_supplier(
    supplier_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a supplier. Available to all authenticated users."""
    supplier = db.query(Supplier).filter(Supplier.id == supplier_id).first()
    if not supplier:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Supplier not found"
        )
    
    db.delete(supplier)
    db.commit()
    
    return {"message": f"Supplier '{supplier.name}' has been deleted successfully"}

# =============================================================================
# STOCK ENDPOINTS
# =============================================================================

@app.get("/stocks/stats", response_model=StockStatsResponse)
async def get_stock_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get stock statistics. Available to all authenticated users."""
    print("📊 Starting get_stock_stats function")
    try:
        # First check if the Stock table exists and has any data
        print("🔍 Checking if Stock table has any data...")
        
        try:
            # Quick check to see if table exists and has data
            table_exists_check = db.execute(text("SELECT COUNT(*) FROM stock")).scalar()
            print(f"✅ Table exists. Row count: {table_exists_check}")
            
            if table_exists_check == 0:
                print("📝 Stock table is empty - returning default zero values")
                return StockStatsResponse(
                    total_stocks=0,
                    active_stocks=0,
                    archived_stocks=0,
                    used_stocks=0,
                    finished_products=0,
                    raw_materials=0
                )
                
        except Exception as table_error:
            print(f"⚠️ Table check failed (table might not exist): {table_error}")
            print("📝 Returning default zero values due to table access issue")
            return StockStatsResponse(
                total_stocks=0,
                active_stocks=0,
                archived_stocks=0,
                used_stocks=0,
                finished_products=0,
                raw_materials=0
            )
        
        print("🔍 Table has data, proceeding with detailed queries...")
        
        # Get basic counts
        total_stocks = db.query(Stock).count() or 0
        print(f"✅ Total stocks: {total_stocks}")
        
        archived_stocks = db.query(Stock).filter(Stock.archive == 1).count() or 0
        print(f"✅ Archived stocks: {archived_stocks}")
        
        used_stocks = db.query(Stock).filter(Stock.used == 1).count() or 0
        print(f"✅ Used stocks: {used_stocks}")
        
        active_stocks = max(0, total_stocks - archived_stocks)
        print(f"✅ Active stocks: {active_stocks}")
        
        # Get category counts with error handling
        try:
            finished_products = db.query(Stock).filter(Stock.category == StockCategory.finished_product).count() or 0
            print(f"✅ Finished products: {finished_products}")
        except Exception as e:
            print(f"⚠️ Error getting finished products count: {e}")
            finished_products = 0
            
        try:
            raw_materials = db.query(Stock).filter(Stock.category == StockCategory.raw_material).count() or 0
            print(f"✅ Raw materials: {raw_materials}")
        except Exception as e:
            print(f"⚠️ Error getting raw materials count: {e}")
            raw_materials = 0
        
        response_data = StockStatsResponse(
            total_stocks=total_stocks,
            active_stocks=active_stocks,
            archived_stocks=archived_stocks,
            used_stocks=used_stocks,
            finished_products=finished_products,
            raw_materials=raw_materials
        )
        
        print(f"📋 Final response: {response_data.dict()}")
        return response_data
        
    except Exception as e:
        print(f"❌ MAJOR ERROR in get_stock_stats: {e}")
        print(f"❌ Error type: {type(e)}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        
        print("🔄 Returning safe default values due to error...")
        return StockStatsResponse(
            total_stocks=0,
            active_stocks=0,
            archived_stocks=0,
            used_stocks=0,
            finished_products=0,
            raw_materials=0
        )

@app.get("/stocks/batches", response_model=List[BatchSummaryResponse])
async def get_all_stock_batches(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all stock batches with summary information. Available to all authenticated users."""
    try:
        print("📦 Getting all stock batches...")
        
        # Check if stock table has any data first
        total_count = db.query(Stock).count()
        print(f"📊 Total stocks in database: {total_count}")
        
        if total_count == 0:
            print("📝 No stocks found in database - returning empty list")
            return []
        
        # Get all unique batch numbers with their creation dates
        batches_query = db.query(Stock.batch, func.min(Stock.created_at).label('created_at')).group_by(Stock.batch).order_by(func.min(Stock.created_at).desc()).all()
        
        batches = []
        for batch_number, created_at in batches_query:
            # Get stocks for this batch
            batch_stocks = get_batch_stocks(db, batch_number)
            
            if batch_stocks:
                # Calculate summary statistics
                total_items = len(batch_stocks)
                total_quantity = sum(stock.quantity for stock in batch_stocks)
                
                # Count by category
                categories = {}
                for stock in batch_stocks:
                    category = stock.category.value if hasattr(stock.category, 'value') else str(stock.category)
                    categories[category] = categories.get(category, 0) + 1
                
                # Get user name from first stock
                first_stock = batch_stocks[0]
                user = db.query(User).filter(User.id == first_stock.users_id).first()
                user_name = f"{user.first_name} {user.last_name}" if user else "Unknown User"
                
                batch_data = BatchSummaryResponse(
                    batch_number=batch_number,
                    total_items=total_items,
                    total_quantity=total_quantity,
                    categories=categories,
                    created_at=created_at,
                    user_name=user_name
                )
                batches.append(batch_data)
        
        print(f"✅ Found {len(batches)} batches")
        return batches
        
    except Exception as e:
        print(f"❌ Error getting batches: {e}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        
        # Return empty list instead of error for better UX
        print("🔄 Returning empty list due to error")
        return []

@app.get("/stocks/batches/{batch_number}", response_model=BatchSummary)
async def get_batch_details(
    batch_number: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed information for a specific batch. Available to all authenticated users."""
    try:
        print(f"🔍 Getting details for batch: {batch_number}")
        
        # Get all stocks for this batch
        batch_stocks = get_batch_stocks(db, batch_number)
        
        if not batch_stocks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch '{batch_number}' not found"
            )
        
        # Calculate summary statistics
        total_items = len(batch_stocks)
        total_quantity = sum(stock.quantity for stock in batch_stocks)
        
        # Count by category
        categories = {}
        for stock in batch_stocks:
            category = stock.category.value if hasattr(stock.category, 'value') else str(stock.category)
            categories[category] = categories.get(category, 0) + 1
        
        # Get user name from first stock
        first_stock = batch_stocks[0]
        user = db.query(User).filter(User.id == first_stock.users_id).first()
        user_name = f"{user.first_name} {user.last_name}" if user else "Unknown User"
        
        # Convert stocks to response format
        stock_responses = []
        for stock in batch_stocks:
            # Get related data
            product = db.query(Product).filter(Product.id == stock.product_id).first()
            supplier = db.query(Supplier).filter(Supplier.id == stock.supplier_id).first()
            
            stock_dict = {
                "id": stock.id,
                "batch": stock.batch,
                "piece": stock.piece,
                "quantity": stock.quantity,
                "unit": stock.unit,
                "category": stock.category,
                "archive": bool(stock.archive),
                "product_id": stock.product_id,
                "supplier_id": stock.supplier_id,
                "users_id": stock.users_id,
                "used": bool(stock.used),
                "created_at": stock.created_at,
                "updated_at": stock.updated_at,
                "product_name": product.name if product else f"Product {stock.product_id} (Not Found)",
                "supplier_name": supplier.name if supplier else f"Supplier {stock.supplier_id} (Not Found)",
                "user_name": user_name
            }
            stock_responses.append(StockResponse(**stock_dict))
        
        return BatchSummary(
            batch_number=batch_number,
            total_items=total_items,
            total_quantity=total_quantity,
            categories=categories,
            created_at=first_stock.created_at,
            user_name=user_name,
            items=stock_responses
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting batch details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving batch details"
        )

@app.get("/stocks/next-batch-number")
async def get_next_batch_number(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the next batch number that will be assigned. Available to all authenticated users."""
    try:
        next_batch = generate_batch_number(db)
        return {"next_batch_number": next_batch}
    except Exception as e:
        print(f"❌ Error getting next batch number: {e}")
        return {"next_batch_number": "batch-000001"}

@app.post("/stocks/batch", response_model=BatchResponse, status_code=status.HTTP_201_CREATED)
async def create_batch_stocks(
    batch_data: BatchStockCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create multiple stock items in a single batch. Available to all authenticated users."""
    try:
        print(f"📦 Creating batch with {len(batch_data.items)} items...")
        
        if not batch_data.items:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one item is required for batch creation"
            )
        
        # Use current user if users_id not provided
        users_id = batch_data.users_id if batch_data.users_id is not None else current_user.id
        
        # Generate batch number
        batch_number = generate_batch_number(db)
        print(f"📝 Generated batch number: {batch_number}")
        
        # Validate all items first
        for i, item in enumerate(batch_data.items):
            # Validate foreign keys
            product = db.query(Product).filter(Product.id == item.product_id).first()
            if not product:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Item {i+1}: Product not found (ID: {item.product_id})"
                )
            
            supplier = db.query(Supplier).filter(Supplier.id == item.supplier_id).first()
            if not supplier:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Item {i+1}: Supplier not found (ID: {item.supplier_id})"
                )
            
            # Validate values
            if item.quantity <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Item {i+1}: Quantity must be greater than 0"
                )
            
            if item.piece <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Item {i+1}: Piece count must be greater than 0"
                )
        
        # Validate user
        user = db.query(User).filter(User.id == users_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User not found"
            )
        
        # Create all stock items
        created_stocks = []
        for item in batch_data.items:
            db_stock = Stock(
                batch=batch_number,
                piece=item.piece,
                quantity=item.quantity,
                unit=item.unit,
                category=item.category,
                product_id=item.product_id,
                supplier_id=item.supplier_id,
                users_id=users_id
            )
            db.add(db_stock)
            created_stocks.append(db_stock)
        
        # Commit all at once
        db.commit()
        
        # Refresh all stocks to get IDs
        for stock in created_stocks:
            db.refresh(stock)
        
        print(f"✅ Created {len(created_stocks)} stock items in batch {batch_number}")
        
        # Prepare response with related data
        stock_responses = []
        for stock in created_stocks:
            product = db.query(Product).filter(Product.id == stock.product_id).first()
            supplier = db.query(Supplier).filter(Supplier.id == stock.supplier_id).first()
            
            stock_dict = {
                "id": stock.id,
                "batch": stock.batch,
                "piece": stock.piece,
                "quantity": stock.quantity,
                "unit": stock.unit,
                "category": stock.category,
                "archive": bool(stock.archive),
                "product_id": stock.product_id,
                "supplier_id": stock.supplier_id,
                "users_id": stock.users_id,
                "used": bool(stock.used),
                "created_at": stock.created_at,
                "updated_at": stock.updated_at,
                "product_name": product.name if product else None,
                "supplier_name": supplier.name if supplier else None,
                "user_name": f"{user.first_name} {user.last_name}"
            }
            stock_responses.append(StockResponse(**stock_dict))
        
        return BatchResponse(
            batch_number=batch_number,
            items_created=len(created_stocks),
            items=stock_responses
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error creating batch: {e}")
        db.rollback()  # Rollback on error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating batch stocks"
        )

@app.delete("/stocks/batches/{batch_number}")
async def delete_batch(
    batch_number: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete all stock items in a batch. Available to all authenticated users."""
    try:
        print(f"🗑️ Deleting batch: {batch_number}")
        
        # Get all stocks for this batch
        batch_stocks = get_batch_stocks(db, batch_number)
        
        if not batch_stocks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch '{batch_number}' not found"
            )
        
        items_count = len(batch_stocks)
        
        # Delete all stocks in this batch
        for stock in batch_stocks:
            db.delete(stock)
        
        db.commit()
        
        print(f"✅ Deleted {items_count} items from batch {batch_number}")
        
        return {"message": f"Batch '{batch_number}' with {items_count} items has been deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting batch: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting batch"
        )

@app.put("/stocks/batches/{batch_number}/archive")
async def archive_batch(
    batch_number: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Archive/unarchive all stock items in a batch. Available to all authenticated users."""
    try:
        print(f"📥 Archiving/unarchiving batch: {batch_number}")
        
        # Get all stocks for this batch
        batch_stocks = get_batch_stocks(db, batch_number)
        
        if not batch_stocks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch '{batch_number}' not found"
            )
        
        # Determine new archive status (toggle based on first item)
        new_archive_status = 1 if batch_stocks[0].archive == 0 else 0
        
        # Update all stocks in this batch
        for stock in batch_stocks:
            stock.archive = new_archive_status
        
        db.commit()
        
        status_text = "archived" if new_archive_status == 1 else "unarchived"
        print(f"✅ {status_text.capitalize()} {len(batch_stocks)} items in batch {batch_number}")
        
        return {"message": f"Batch '{batch_number}' has been {status_text} successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error archiving batch: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error archiving batch"
        )
    
@app.put("/stocks/batches/{batch_number}/set-archive")
async def set_batch_archive_status(
    batch_number: str,
    archive_request: BatchArchiveRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Set archive status for all stock items in a batch to a specific boolean value. Available to all authenticated users."""
    try:
        print(f"📥 Setting batch {batch_number} archive status to: {archive_request.archive}")
        
        # Get all stocks for this batch
        batch_stocks = get_batch_stocks(db, batch_number)
        
        if not batch_stocks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch '{batch_number}' not found"
            )
        
        # Set archive status for all items in batch
        archive_value = 1 if archive_request.archive else 0
        items_updated = 0
        
        for stock in batch_stocks:
            stock.archive = archive_value
            items_updated += 1
        
        db.commit()
        
        status_text = "archived" if archive_request.archive else "unarchived"
        print(f"✅ {status_text.capitalize()} {items_updated} items in batch {batch_number}")
        
        return {
            "message": f"Batch '{batch_number}' has been {status_text} successfully",
            "batch_number": batch_number,
            "items_updated": items_updated,
            "archive_status": archive_request.archive
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error setting archive status for batch: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error setting archive status for batch"
        )

@app.get("/process-management/stats")
async def get_process_management_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get process management statistics. Available to all authenticated users."""
    try:
        # Check if table exists and has data
        try:
            table_exists_check = db.execute(text("SELECT COUNT(*) FROM process_management")).scalar()
            
            if table_exists_check == 0:
                return {
                    "total_processes": 0,
                    "active_processes": 0,
                    "archived_processes": 0,
                    "total_batches": 0
                }
        except Exception as table_error:
            return {
                "total_processes": 0,
                "active_processes": 0,
                "archived_processes": 0,
                "total_batches": 0
            }
        
        # Get basic counts
        total_processes = db.query(ProcessManagement).count() or 0
        archived_processes = db.query(ProcessManagement).filter(ProcessManagement.archive == 1).count() or 0
        active_processes = max(0, total_processes - archived_processes)
        
        # Count unique batches
        total_batches = db.query(ProcessManagement.process_id_batch).filter(ProcessManagement.process_id_batch.isnot(None)).distinct().count() or 0
        
        return {
            "total_processes": total_processes,
            "active_processes": active_processes,
            "archived_processes": archived_processes,
            "total_batches": total_batches
        }
        
    except Exception as e:
        print(f"❌ Error getting process management stats: {e}")
        return {
            "total_processes": 0,
            "active_processes": 0,
            "archived_processes": 0,
            "total_batches": 0
        }

@app.get("/process-management/batches", response_model=List[ProcessBatchSummaryResponse])
async def get_all_process_batches(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all process management batches with summary information. Available to all authenticated users."""
    try:
        print("📦 Getting all process management batches...")
        
        # Check if table has any data first
        total_count = db.query(ProcessManagement).count()
        print(f"📊 Total processes in database: {total_count}")
        
        if total_count == 0:
            print("📝 No processes found in database - returning empty list")
            return []
        
        # Get all unique batch numbers with their creation dates
        batches_query = db.query(ProcessManagement.process_id_batch, func.min(ProcessManagement.manufactured_date).label('manufactured_date')).filter(ProcessManagement.process_id_batch.isnot(None)).group_by(ProcessManagement.process_id_batch).order_by(func.min(ProcessManagement.manufactured_date).desc()).all()
        
        batches = []
        for batch_number, manufactured_date in batches_query:
            # Get processes for this batch
            batch_processes = get_process_batch_items(db, batch_number)
            
            if batch_processes:
                # Calculate summary statistics
                total_items = len(batch_processes)
                
                # Get user name from first process
                first_process = batch_processes[0]
                user = db.query(User).filter(User.id == first_process.users_id).first()
                user_name = f"{user.first_name} {user.last_name}" if user else "Unknown User"
                
                batch_data = ProcessBatchSummaryResponse(
                    process_batch_number=batch_number,
                    total_items=total_items,
                    manufactured_date=manufactured_date,
                    user_name=user_name
                )
                batches.append(batch_data)
        
        print(f"✅ Found {len(batches)} process batches")
        return batches
        
    except Exception as e:
        print(f"❌ Error getting process batches: {e}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        return []

@app.get("/process-management/batches/{process_batch_number}", response_model=ProcessBatchSummary)
async def get_process_batch_details(
    process_batch_number: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed information for a specific process batch. Available to all authenticated users."""
    try:
        print(f"🔍 Getting details for process batch: {process_batch_number}")
        
        # Get all processes for this batch
        batch_processes = get_process_batch_items(db, process_batch_number)
        
        if not batch_processes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Process batch '{process_batch_number}' not found"
            )
        
        # Calculate summary statistics
        total_items = len(batch_processes)
        
        # Get user name from first process
        first_process = batch_processes[0]
        user = db.query(User).filter(User.id == first_process.users_id).first()
        user_name = f"{user.first_name} {user.last_name}" if user else "Unknown User"
        
        # Convert processes to response format
        process_responses = []
        for process in batch_processes:
            # Get related data
            stock = db.query(Stock).filter(Stock.id == process.stock_id).first()
            finished_product = db.query(Product).filter(Product.id == process.finished_product_id).first()
            
            process_dict = {
                "id": process.id,
                "process_id_batch": process.process_id_batch,
                "stock_id": process.stock_id,
                "users_id": process.users_id,
                "finished_product_id": process.finished_product_id,
                "archive": bool(process.archive),
                "manufactured_date": process.manufactured_date,
                "updated_at": process.updated_at,
                "stock_batch": stock.batch if stock else f"Stock {process.stock_id} (Not Found)",
                "finished_product_name": finished_product.name if finished_product else f"Product {process.finished_product_id} (Not Found)",
                "user_name": user_name
            }
            process_responses.append(ProcessManagementResponse(**process_dict))
        
        return ProcessBatchSummary(
            process_batch_number=process_batch_number,
            total_items=total_items,
            manufactured_date=first_process.manufactured_date,
            user_name=user_name,
            items=process_responses
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting process batch details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving process batch details"
        )

@app.get("/process-management/next-batch-number")
async def get_next_process_batch_number(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the next process batch number that will be assigned. Available to all authenticated users."""
    try:
        next_batch = generate_process_batch_number(db)
        return {"next_process_batch_number": next_batch}
    except Exception as e:
        print(f"❌ Error getting next process batch number: {e}")
        return {"next_process_batch_number": "process-000001"}

@app.post("/process-management/batch", response_model=ProcessBatchResponse, status_code=status.HTTP_201_CREATED)
async def create_process_batch(
    batch_data: BatchProcessCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create multiple process management items in a single batch. Available to all authenticated users."""
    try:
        print(f"📦 Creating process batch with {len(batch_data.items)} items...")
        
        if not batch_data.items:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one item is required for batch creation"
            )
        
        # Use current user if users_id not provided
        users_id = batch_data.users_id if batch_data.users_id is not None else current_user.id
        
        # Generate process batch number
        process_batch_number = generate_process_batch_number(db)
        print(f"📝 Generated process batch number: {process_batch_number}")
        
        # Validate all items first
        for i, item in enumerate(batch_data.items):
            # Validate foreign keys
            stock = db.query(Stock).filter(Stock.id == item.stock_id).first()
            if not stock:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Item {i+1}: Stock not found (ID: {item.stock_id})"
                )
            
            finished_product = db.query(Product).filter(Product.id == item.finished_product_id).first()
            if not finished_product:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Item {i+1}: Finished product not found (ID: {item.finished_product_id})"
                )
        
        # Validate user
        user = db.query(User).filter(User.id == users_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User not found"
            )
        
        # Create all process management items
        created_processes = []
        for item in batch_data.items:
            db_process = ProcessManagement(
                process_id_batch=process_batch_number,
                stock_id=item.stock_id,
                users_id=users_id,
                finished_product_id=item.finished_product_id
            )
            db.add(db_process)
            created_processes.append(db_process)
        
        # Commit all at once
        db.commit()
        
        # Refresh all processes to get IDs
        for process in created_processes:
            db.refresh(process)
        
        print(f"✅ Created {len(created_processes)} process items in batch {process_batch_number}")
        
        # Prepare response with related data
        process_responses = []
        for process in created_processes:
            stock = db.query(Stock).filter(Stock.id == process.stock_id).first()
            finished_product = db.query(Product).filter(Product.id == process.finished_product_id).first()
            
            process_dict = {
                "id": process.id,
                "process_id_batch": process.process_id_batch,
                "stock_id": process.stock_id,
                "users_id": process.users_id,
                "finished_product_id": process.finished_product_id,
                "archive": bool(process.archive),
                "manufactured_date": process.manufactured_date,
                "updated_at": process.updated_at,
                "stock_batch": stock.batch if stock else None,
                "finished_product_name": finished_product.name if finished_product else None,
                "user_name": f"{user.first_name} {user.last_name}"
            }
            process_responses.append(ProcessManagementResponse(**process_dict))
        
        return ProcessBatchResponse(
            process_batch_number=process_batch_number,
            items_created=len(created_processes),
            items=process_responses
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error creating process batch: {e}")
        db.rollback()  # Rollback on error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating process batch"
        )

@app.delete("/process-management/batches/{process_batch_number}")
async def delete_process_batch(
    process_batch_number: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete all process management items in a batch. Available to all authenticated users."""
    try:
        print(f"🗑️ Deleting process batch: {process_batch_number}")
        
        # Get all processes for this batch
        batch_processes = get_process_batch_items(db, process_batch_number)
        
        if not batch_processes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Process batch '{process_batch_number}' not found"
            )
        
        items_count = len(batch_processes)
        
        # Delete all processes in this batch
        for process in batch_processes:
            db.delete(process)
        
        db.commit()
        
        print(f"✅ Deleted {items_count} items from process batch {process_batch_number}")
        
        return {"message": f"Process batch '{process_batch_number}' with {items_count} items has been deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting process batch: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting process batch"
        )

@app.put("/process-management/batches/{process_batch_number}/archive")
async def archive_process_batch(
    process_batch_number: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Archive/unarchive all process management items in a batch. Available to all authenticated users."""
    try:
        print(f"📥 Archiving/unarchiving process batch: {process_batch_number}")
        
        # Get all processes for this batch
        batch_processes = get_process_batch_items(db, process_batch_number)
        
        if not batch_processes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Process batch '{process_batch_number}' not found"
            )
        
        # Determine new archive status (toggle based on first item)
        new_archive_status = 1 if batch_processes[0].archive == 0 else 0
        
        # Update all processes in this batch
        for process in batch_processes:
            process.archive = new_archive_status
        
        db.commit()
        
        status_text = "archived" if new_archive_status == 1 else "unarchived"
        print(f"✅ {status_text.capitalize()} {len(batch_processes)} items in process batch {process_batch_number}")
        
        return {"message": f"Process batch '{process_batch_number}' has been {status_text} successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error archiving process batch: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error archiving process batch"
        )

@app.put("/process-management/batches/{process_batch_number}/set-archive")
async def set_process_batch_archive_status(
    process_batch_number: str,
    archive_request: ProcessBatchArchiveRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Set archive status for all process management items in a batch to a specific boolean value. Available to all authenticated users."""
    try:
        print(f"📥 Setting process batch {process_batch_number} archive status to: {archive_request.archive}")
        
        # Get all processes for this batch
        batch_processes = get_process_batch_items(db, process_batch_number)
        
        if not batch_processes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Process batch '{process_batch_number}' not found"
            )
        
        # Set archive status for all items in batch
        archive_value = 1 if archive_request.archive else 0
        items_updated = 0
        
        for process in batch_processes:
            process.archive = archive_value
            items_updated += 1
        
        db.commit()
        
        status_text = "archived" if archive_request.archive else "unarchived"
        print(f"✅ {status_text.capitalize()} {items_updated} items in process batch {process_batch_number}")
        
        return {
            "message": f"Process batch '{process_batch_number}' has been {status_text} successfully",
            "process_batch_number": process_batch_number,
            "items_updated": items_updated,
            "archive_status": archive_request.archive
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error setting archive status for process batch: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error setting archive status for process batch"
        )

# Individual Process Management CRUD endpoints
@app.get("/process-management", response_model=List[ProcessManagementResponse])
async def get_all_process_management(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    archive: Optional[bool] = None
):
    """Get all process management items with optional filtering. Available to all authenticated users."""
    try:
        print(f"🔍 Getting process management items with filters - archive: {archive}")
        
        query = db.query(ProcessManagement)
        
        # Apply filters
        if archive is not None:
            query = query.filter(ProcessManagement.archive == (1 if archive else 0))
            print(f"🔍 Applied archive filter: {archive}")
        
        processes = query.all()
        print(f"📋 Found {len(processes)} process management items after filtering")
        
        if not processes:
            return []
        
        # Enhance with related data
        process_responses = []
        for process in processes:
            try:
                # Get related data safely
                stock = db.query(Stock).filter(Stock.id == process.stock_id).first()
                finished_product = db.query(Product).filter(Product.id == process.finished_product_id).first()
                user = db.query(User).filter(User.id == process.users_id).first()
                
                process_dict = {
                    "id": process.id,
                    "process_id_batch": process.process_id_batch,
                    "stock_id": process.stock_id,
                    "users_id": process.users_id,
                    "finished_product_id": process.finished_product_id,
                    "archive": bool(process.archive),
                    "manufactured_date": process.manufactured_date,
                    "updated_at": process.updated_at,
                    "stock_batch": stock.batch if stock else f"Stock {process.stock_id} (Not Found)",
                    "finished_product_name": finished_product.name if finished_product else f"Product {process.finished_product_id} (Not Found)",
                    "user_name": f"{user.first_name} {user.last_name}" if user else f"User {process.users_id} (Not Found)"
                }
                process_responses.append(ProcessManagementResponse(**process_dict))
                
            except Exception as e:
                print(f"⚠️ Error processing process management item {process.id}: {e}")
                continue
        
        print(f"✅ Successfully processed {len(process_responses)} process management items")
        return process_responses
        
    except Exception as e:
        print(f"❌ Error in get_all_process_management: {e}")
        return []

@app.post("/process-management", response_model=ProcessManagementResponse, status_code=status.HTTP_201_CREATED)
async def create_process_management(
    process_data: ProcessManagementCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new process management item. Available to all authenticated users."""
    
    # Validate foreign keys
    stock = db.query(Stock).filter(Stock.id == process_data.stock_id).first()
    if not stock:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stock not found"
        )
    
    finished_product = db.query(Product).filter(Product.id == process_data.finished_product_id).first()
    if not finished_product:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Finished product not found"
        )
    
    user = db.query(User).filter(User.id == process_data.users_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User not found"
        )
    
    # Create process management item without batch (single item)
    db_process = ProcessManagement(
        stock_id=process_data.stock_id,
        users_id=process_data.users_id,
        finished_product_id=process_data.finished_product_id
        # process_id_batch will be None for individual items
    )
    db.add(db_process)
    db.commit()
    db.refresh(db_process)
    
    # Return with related data
    process_dict = {
        "id": db_process.id,
        "process_id_batch": db_process.process_id_batch,
        "stock_id": db_process.stock_id,
        "users_id": db_process.users_id,
        "finished_product_id": db_process.finished_product_id,
        "archive": bool(db_process.archive),
        "manufactured_date": db_process.manufactured_date,
        "updated_at": db_process.updated_at,
        "stock_batch": stock.batch,
        "finished_product_name": finished_product.name,
        "user_name": f"{user.first_name} {user.last_name}"
    }
    
    return ProcessManagementResponse(**process_dict)

@app.get("/stocks", response_model=List[StockResponse])
async def get_all_stocks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    category: Optional[StockCategory] = None,
    archive: Optional[bool] = None,
    used: Optional[bool] = None
):
    """Get all stocks with optional filtering. Available to all authenticated users."""
    try:
        print(f"🔍 Getting stocks with filters - category: {category}, archive: {archive}, used: {used}")
        
        # Check if stock table has any data first
        total_count = db.query(Stock).count()
        print(f"📊 Total stocks in database: {total_count}")
        
        if total_count == 0:
            print("📝 No stocks found in database - returning empty list")
            return []
        
        query = db.query(Stock)
        
        # Apply filters
        if category:
            query = query.filter(Stock.category == category)
            print(f"🔍 Applied category filter: {category}")
        if archive is not None:
            query = query.filter(Stock.archive == (1 if archive else 0))
            print(f"🔍 Applied archive filter: {archive}")
        if used is not None:
            query = query.filter(Stock.used == (1 if used else 0))
            print(f"🔍 Applied used filter: {used}")
        
        stocks = query.all()
        print(f"📋 Found {len(stocks)} stocks after filtering")
        
        if not stocks:
            print("📝 No stocks match the current filters - returning empty list")
            return []
        
        # Enhance with related data
        stock_responses = []
        for stock in stocks:
            try:
                # Get related data safely
                product = db.query(Product).filter(Product.id == stock.product_id).first()
                supplier = db.query(Supplier).filter(Supplier.id == stock.supplier_id).first()
                user = db.query(User).filter(User.id == stock.users_id).first()
                
                stock_dict = {
                    "id": stock.id,
                    "batch": stock.batch,
                    "piece": stock.piece,
                    # REMOVED: "quantity": stock.quantity,
                    # REMOVED: "unit": stock.unit,
                    "category": stock.category,
                    "archive": bool(stock.archive),
                    "product_id": stock.product_id,
                    "supplier_id": stock.supplier_id,
                    "users_id": stock.users_id,
                    "used": bool(stock.used),
                    "created_at": stock.created_at,
                    "updated_at": stock.updated_at,
                    "product_name": product.name if product else f"Product {stock.product_id} (Not Found)",
                    "product_unit": product.unit if product else None,      # ADDED: From Product table
                    "product_quantity": product.quantity if product else None,  # ADDED: From Product table
                    "supplier_name": supplier.name if supplier else f"Supplier {stock.supplier_id} (Not Found)",
                    "user_name": f"{user.first_name} {user.last_name}" if user else f"User {stock.users_id} (Not Found)"
                }
                stock_responses.append(StockResponse(**stock_dict))
                
            except Exception as e:
                print(f"⚠️ Error processing stock {stock.id}: {e}")
                # Continue with next stock instead of failing completely
                continue
        
        print(f"✅ Successfully processed {len(stock_responses)} stocks")
        return stock_responses
        
    except Exception as e:
        print(f"❌ Error in get_all_stocks: {e}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        
        # Return empty list instead of error for better UX
        print("🔄 Returning empty list due to error")
        return []

@app.get("/stocks/{stock_id}", response_model=StockResponse)
async def get_stock_by_id(
    stock_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific stock item by ID. Available to all authenticated users."""
    stock = db.query(Stock).filter(Stock.id == stock_id).first()
    if not stock:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stock item not found"
        )
    
    # Get related data including product unit and quantity
    product = db.query(Product).filter(Product.id == stock.product_id).first()
    supplier = db.query(Supplier).filter(Supplier.id == stock.supplier_id).first()
    user = db.query(User).filter(User.id == stock.users_id).first()
    
    stock_dict = {
        "id": stock.id,
        "batch": stock.batch,
        "piece": stock.piece,
        # REMOVED: "quantity": stock.quantity,
        # REMOVED: "unit": stock.unit,
        "category": stock.category,
        "archive": bool(stock.archive),
        "product_id": stock.product_id,
        "supplier_id": stock.supplier_id,
        "users_id": stock.users_id,
        "used": bool(stock.used),
        "created_at": stock.created_at,
        "updated_at": stock.updated_at,
        "product_name": product.name if product else None,
        "product_unit": product.unit if product else None,      # ADDED: From Product table
        "product_quantity": product.quantity if product else None,  # ADDED: From Product table
        "supplier_name": supplier.name if supplier else None,
        "user_name": f"{user.first_name} {user.last_name}" if user else None
    }
    
    return StockResponse(**stock_dict)

@app.post("/stocks", response_model=StockResponse, status_code=status.HTTP_201_CREATED)
async def create_stock(
    stock_data: StockCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new stock item. Available to all authenticated users."""
    
    # Validate foreign keys
    product = db.query(Product).filter(Product.id == stock_data.product_id).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Product not found"
        )
    
    supplier = db.query(Supplier).filter(Supplier.id == stock_data.supplier_id).first()
    if not supplier:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Supplier not found"
        )
    
    user = db.query(User).filter(User.id == stock_data.users_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User not found"
        )
    
    # REMOVED: Validate quantity (no longer in stock)
    # if stock_data.quantity <= 0:
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail="Quantity must be greater than 0"
    #     )
    
    # Validate piece count
    if stock_data.piece <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Piece count must be greater than 0"
        )
    
    # Create stock item without unit and quantity
    db_stock = Stock(
        batch=stock_data.batch,
        piece=stock_data.piece,
        # REMOVED: quantity=stock_data.quantity,
        # REMOVED: unit=stock_data.unit,
        category=stock_data.category,
        product_id=stock_data.product_id,
        supplier_id=stock_data.supplier_id,
        users_id=stock_data.users_id
    )
    db.add(db_stock)
    db.commit()
    db.refresh(db_stock)
    
    # Return with related data including product unit and quantity
    stock_dict = {
        "id": db_stock.id,
        "batch": db_stock.batch,
        "piece": db_stock.piece,
        # REMOVED: "quantity": db_stock.quantity,
        # REMOVED: "unit": db_stock.unit,
        "category": db_stock.category,
        "archive": bool(db_stock.archive),
        "product_id": db_stock.product_id,
        "supplier_id": db_stock.supplier_id,
        "users_id": db_stock.users_id,
        "used": bool(db_stock.used),
        "created_at": db_stock.created_at,
        "updated_at": db_stock.updated_at,
        "product_name": product.name,
        "product_unit": product.unit,      # ADDED: From Product table
        "product_quantity": product.quantity,  # ADDED: From Product table
        "supplier_name": supplier.name,
        "user_name": f"{user.first_name} {user.last_name}"
    }
    
    return StockResponse(**stock_dict)

@app.put("/stocks/{stock_id}", response_model=StockResponse)
async def update_stock(
    stock_id: int,
    stock_update: StockUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a stock item. Available to all authenticated users."""
    stock = db.query(Stock).filter(Stock.id == stock_id).first()
    if not stock:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stock item not found"
        )
    
    # Validate foreign keys if being updated
    if stock_update.product_id:
        product = db.query(Product).filter(Product.id == stock_update.product_id).first()
        if not product:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Product not found"
            )
    
    if stock_update.supplier_id:
        supplier = db.query(Supplier).filter(Supplier.id == stock_update.supplier_id).first()
        if not supplier:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Supplier not found"
            )
    
    if stock_update.users_id:
        user = db.query(User).filter(User.id == stock_update.users_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User not found"
            )
    
    # REMOVED: Validate quantity (no longer in stock)
    # if stock_update.quantity is not None and stock_update.quantity <= 0:
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail="Quantity must be greater than 0"
    #     )
    
    # Validate piece count
    if stock_update.piece is not None and stock_update.piece <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Piece count must be greater than 0"
        )
    
    # Update only the fields that are provided (not None)
    update_data = stock_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        if hasattr(stock, field):
            # Handle boolean conversion for archive and used fields
            if field in ['archive', 'used'] and isinstance(value, bool):
                setattr(stock, field, 1 if value else 0)
            else:
                setattr(stock, field, value)
    
    db.commit()
    db.refresh(stock)
    
    # Get related data for response including product unit and quantity
    product = db.query(Product).filter(Product.id == stock.product_id).first()
    supplier = db.query(Supplier).filter(Supplier.id == stock.supplier_id).first()
    user = db.query(User).filter(User.id == stock.users_id).first()
    
    stock_dict = {
        "id": stock.id,
        "batch": stock.batch,
        "piece": stock.piece,
        # REMOVED: "quantity": stock.quantity,
        # REMOVED: "unit": stock.unit,
        "category": stock.category,
        "archive": bool(stock.archive),
        "product_id": stock.product_id,
        "supplier_id": stock.supplier_id,
        "users_id": stock.users_id,
        "used": bool(stock.used),
        "created_at": stock.created_at,
        "updated_at": stock.updated_at,
        "product_name": product.name if product else None,
        "product_unit": product.unit if product else None,      # ADDED: From Product table
        "product_quantity": product.quantity if product else None,  # ADDED: From Product table
        "supplier_name": supplier.name if supplier else None,
        "user_name": f"{user.first_name} {user.last_name}" if user else None
    }
    
    return StockResponse(**stock_dict)

@app.delete("/stocks/{stock_id}")
async def delete_stock(
    stock_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a stock item. Available to all authenticated users."""
    stock = db.query(Stock).filter(Stock.id == stock_id).first()
    if not stock:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stock item not found"
        )
    
    db.delete(stock)
    db.commit()
    
    return {"message": f"Stock item with batch '{stock.batch}' has been deleted successfully"}

@app.put("/stocks/{stock_id}/archive")
async def archive_stock(
    stock_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Archive/unarchive a stock item. Available to all authenticated users."""
    stock = db.query(Stock).filter(Stock.id == stock_id).first()
    if not stock:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stock item not found"
        )
    
    # Toggle archive status
    stock.archive = 1 if stock.archive == 0 else 0
    db.commit()
    
    status_text = "archived" if stock.archive == 1 else "unarchived"
    return {"message": f"Stock item has been {status_text} successfully"}

@app.put("/stocks/{stock_id}/use")
async def mark_stock_used(
    stock_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Mark/unmark a stock item as used. Available to all authenticated users."""
    stock = db.query(Stock).filter(Stock.id == stock_id).first()
    if not stock:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stock item not found"
        )
    
    # Toggle used status
    stock.used = 1 if stock.used == 0 else 0
    db.commit()
    
    status_text = "marked as used" if stock.used == 1 else "marked as unused"
    return {"message": f"Stock item has been {status_text} successfully"}

@app.get("/")
async def root():
    """API Health Check"""
    return {"message": "Stock Inventory API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4567)