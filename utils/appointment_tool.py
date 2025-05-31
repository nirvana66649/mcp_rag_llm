# appointment_tool.py
from utils.db_utils import get_connection
import re
from datetime import datetime, timedelta
from uuid import uuid4


def manage_appointment(action, username=None, id_card=None, department=None, date=None,
                       time=None, appointment_id=None, access_token=None):
    try:
        connection = get_connection()
        with connection.cursor() as cursor:
            action_lower = action.lower()

            # 添加
            if action_lower in ["预约", "添加", "add", "create"]:
                if not username or not id_card:
                    return "❌ 预约失败：姓名和身份证号为必填项"

                if not re.match(r'^\d{15}(\d{2}[0-9xX])?$', id_card):
                    return "❌ 身份证号格式不正确"

                cursor.execute("SELECT id FROM appointment WHERE username = %s AND id_card = %s", (username, id_card))
                existing = cursor.fetchone()
                if existing:
                    return f"❌ 该用户已有预约记录（ID: {existing['id']}），请先取消现有预约或选择修改"

                token = uuid4().hex
                expire_at = (datetime.now() + timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')

                insert_sql = """
                INSERT INTO appointment (username, id_card, department, date, time, access_token, token_expire_at) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(insert_sql, (username, id_card, department, date, time, token, expire_at))
                connection.commit()

                new_id = cursor.lastrowid
                return (
                    f"✅ 预约成功！\n预约ID: {new_id}\n姓名: {username}\n身份证: {id_card}\n科室: {department or '未指定'}\n"
                    f"日期: {date or '未指定'}\n时间: {time or '未指定'}\n🔐 验证码: {token}（24小时内有效）"
                )

            # 修改
            elif action_lower in ["修改", "更新", "update", "modify"]:
                if not access_token:
                    return "❌ 修改失败：缺少验证码"

                query_sql = "SELECT * FROM appointment WHERE access_token = %s AND token_expire_at > NOW()"
                if appointment_id:
                    query_sql += " AND id = %s"
                    cursor.execute(query_sql, (access_token, appointment_id))
                elif username and id_card:
                    query_sql += " AND username = %s AND id_card = %s"
                    cursor.execute(query_sql, (access_token, username, id_card))
                else:
                    return "❌ 修改失败：请提供预约ID或姓名+身份证"

                existing = cursor.fetchone()
                if not existing:
                    return "❌ 验证失败或验证码已过期"

                update_fields, update_values = [], []
                if username and username != existing['username']:
                    update_fields.append("username = %s")
                    update_values.append(username)
                if id_card and id_card != existing['id_card']:
                    if not re.match(r'^\d{15}(\d{2}[0-9xX])?$', id_card):
                        return "❌ 身份证号格式错误"
                    update_fields.append("id_card = %s")
                    update_values.append(id_card)
                if department is not None:
                    update_fields.append("department = %s")
                    update_values.append(department)
                if date:
                    try:
                        datetime.strptime(date, '%Y-%m-%d')
                        update_fields.append("date = %s")
                        update_values.append(date)
                    except:
                        return "❌ 日期格式错误"
                if time:
                    try:
                        datetime.strptime(time, '%H:%M:%S')
                        update_fields.append("time = %s")
                        update_values.append(time)
                    except:
                        return "❌ 时间格式错误"

                if not update_fields:
                    return "❌ 没有需要修改的内容"

                update_sql = f"UPDATE appointment SET {', '.join(update_fields)} WHERE id = %s"
                update_values.append(existing['id'])
                cursor.execute(update_sql, update_values)
                connection.commit()

                return "✅ 修改成功！"

            # 删除
            elif action_lower in ["删除", "取消", "delete", "cancel", "remove"]:
                if not access_token:
                    return "❌ 删除失败：缺少验证码"

                query_sql = "SELECT * FROM appointment WHERE access_token = %s AND token_expire_at > NOW()"
                if appointment_id:
                    query_sql += " AND id = %s"
                    cursor.execute(query_sql, (access_token, appointment_id))
                elif username and id_card:
                    query_sql += " AND username = %s AND id_card = %s"
                    cursor.execute(query_sql, (access_token, username, id_card))
                else:
                    return "❌ 删除失败：请提供预约ID或姓名+身份证"

                existing = cursor.fetchone()
                if not existing:
                    return "❌ 验证失败或验证码已过期"

                cursor.execute("DELETE FROM appointment WHERE id = %s", (existing['id'],))
                connection.commit()
                return "✅ 删除成功！"

            return "❌ 不支持的操作类型"

    except Exception as e:
        return f"❌ 错误：{str(e)}"
    finally:
        if 'connection' in locals():
            connection.close()
