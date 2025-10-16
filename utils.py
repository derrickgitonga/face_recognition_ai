import sqlite3
from datetime import datetime, timedelta
import json
import os

def setup_directories():
    """Create all necessary directories"""
    directories = ["datasets", "unknown_faces", "output"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("[INFO] Created all necessary directories")

def get_access_stats(hours=24):
    """Get access statistics for the last N hours"""
    conn = sqlite3.connect('access_logs.db')
    
    time_threshold = datetime.now() - timedelta(hours=hours)
    
    stats = conn.execute('''
        SELECT 
            COUNT(*) as total_attempts,
            SUM(success) as successful_attempts,
            AVG(confidence) as avg_confidence
        FROM access_logs 
        WHERE access_time > ?
    ''', (time_threshold,)).fetchone()
    
    # Get unknown face count
    unknown_count = conn.execute('''
        SELECT COUNT(*) FROM access_logs 
        WHERE user_name = 'Unknown' 
        AND access_time > ?
    ''', (time_threshold,)).fetchone()
    
    conn.close()
    
    return {
        'total_attempts': stats[0] or 0,
        'successful_attempts': stats[1] or 0,
        'failed_attempts': (stats[0] or 0) - (stats[1] or 0),
        'unknown_faces': unknown_count[0] or 0,
        'success_rate': (stats[1] / stats[0] * 100) if stats[0] else 0,
        'avg_confidence': stats[2] or 0
    }

def list_known_persons():
    """List all known persons in the dataset"""
    if not os.path.exists("datasets"):
        return []
    
    persons = []
    for item in os.listdir("datasets"):
        if os.path.isdir(os.path.join("datasets", item)):
            image_count = len([f for f in os.listdir(os.path.join("datasets", item)) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            persons.append({"name": item, "image_count": image_count})
    
    return persons