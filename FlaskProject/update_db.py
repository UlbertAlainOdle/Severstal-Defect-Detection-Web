import sqlite3
import os
import hashlib

# 数据库路径
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detection_records.db')

def hash_password(password):
    """对密码进行哈希处理"""
    return hashlib.sha256(password.encode()).hexdigest()

def update_database():
    """更新数据库结构，添加is_admin字段"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 检查是否已存在is_admin字段
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        
        if 'is_admin' not in column_names:
            print("添加is_admin字段...")
            cursor.execute("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0")
            
            # 创建默认管理员账号
            admin_username = "admin"
            admin_email = "admin@example.com"
            admin_password = hash_password("111111aA")
            
            # 检查用户名是否已存在
            cursor.execute("SELECT id FROM users WHERE username = ?", (admin_username,))
            user = cursor.fetchone()
            
            if user:
                # 更新现有用户为管理员
                cursor.execute("UPDATE users SET is_admin = 1 WHERE username = ?", (admin_username,))
                print(f"已将用户 {admin_username} 设置为管理员")
            else:
                # 创建新管理员账号
                cursor.execute(
                    "INSERT INTO users (username, email, password, created_at, is_admin) VALUES (?, ?, ?, datetime('now'), ?)",
                    (admin_username, admin_email, admin_password, 1)
                )
                print(f"已创建默认管理员账号: {admin_username}")
            
            conn.commit()
            print("数据库更新成功！")
        else:
            print("is_admin字段已存在，无需更新")
        
    except Exception as e:
        print(f"更新数据库时出错: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    update_database() 