import sqlite3
from datetime import datetime, timedelta
import time
import os

def get_detailed_stats():
    conn = sqlite3.connect('access_logs.db')
    
    # Today's stats
    today = datetime.now().strftime("%Y-%m-%d")
    today_stats = conn.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(success) as success,
            AVG(confidence) as avg_conf
        FROM access_logs 
        WHERE DATE(access_time) = ?
    ''', (today,)).fetchone()
    
    # Last hour stats
    one_hour_ago = datetime.now() - timedelta(hours=1)
    hour_stats = conn.execute('''
        SELECT COUNT(*) 
        FROM access_logs 
        WHERE access_time > ?
    ''', (one_hour_ago,)).fetchone()
    
    # Recent activity (last 10 entries)
    recent = conn.execute('''
        SELECT user_name, success, confidence, access_time
        FROM access_logs 
        ORDER BY access_time DESC 
        LIMIT 10
    ''').fetchall()
    
    conn.close()
    
    return {
        'today_total': today_stats[0] or 0,
        'today_success': today_stats[1] or 0,
        'today_avg_conf': today_stats[2] or 0,
        'last_hour': hour_stats[0] or 0,
        'recent_activity': recent
    }

def monitor_system():
    """Enhanced monitoring dashboard"""
    print("Starting Face Access Control Monitor...")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            stats = get_detailed_stats()
            os.system('cls' if os.name == 'nt' else 'clear')
            
            today = datetime.now().strftime("%Y-%m-%d")
            success_rate = (stats['today_success'] / stats['today_total'] * 100) if stats['today_total'] > 0 else 0
            
            print("=" * 60)
            print("           FACE ACCESS CONTROL - SYSTEM MONITOR")
            print("=" * 60)
            print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 60)
            
            print(f"\n📊 TODAY'S STATISTICS ({today}):")
            print(f"   Total Attempts: {stats['today_total']}")
            print(f"   Successful: {stats['today_success']}")
            print(f"   Failed: {stats['today_total'] - stats['today_success']}")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   Avg Confidence: {stats['today_avg_conf']:.2f}")
            print(f"   Last Hour Activity: {stats['last_hour']} attempts")
            
            print(f"\n🕒 RECENT ACTIVITY (Last 10):")
            if stats['recent_activity']:
                for entry in stats['recent_activity']:
                    name, success, confidence, access_time = entry
                    status = "✅ GRANTED" if success else "❌ DENIED"
                    time_str = datetime.strptime(access_time, '%Y-%m-%d %H:%M:%S').strftime('%H:%M:%S')
                    print(f"   {time_str} - {status} - {name} ({confidence:.2f})")
            else:
                print("   No recent activity")
            
            print(f"\n📁 SYSTEM INFO:")
            unknown_count = len(os.listdir('unknown_faces')) if os.path.exists('unknown_faces') else 0
            datasets_count = len([f for f in os.listdir('datasets') if os.path.isdir(os.path.join('datasets', f))]) if os.path.exists('datasets') else 0
            print(f"   Unknown faces stored: {unknown_count}")
            print(f"   Known persons: {datasets_count}")
            
            print("\n" + "=" * 60)
            print("Monitoring... (updates every 10 seconds)")
            
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n[INFO] Monitoring stopped")

if __name__ == "__main__":
    monitor_system()