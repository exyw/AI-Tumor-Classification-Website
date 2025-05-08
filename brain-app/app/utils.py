from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class MRIImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)