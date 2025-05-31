from utils.db_utils import get_connection


def query_appointment(
    username: str = None,
    id_card: str = None,
    appointment_id: int = None,
    access_token: str = None
) -> str:
    """
    查询预约记录

    参数:
        username (str): 用户名（可选）
        id_card (str): 身份证号（可选）
        appointment_id (int): 预约ID（可选）
        access_token (str): 验证码，用于权限校验（可选）

    返回:
        str: 查询结果（文本格式）
    """
    try:
        connection = get_connection()
        with connection.cursor() as cursor:
            if access_token:
                # 校验 access_token 是否有效
                query_sql = "SELECT * FROM appointment WHERE access_token = %s AND token_expire_at > NOW()"
                params = [access_token]

                if appointment_id:
                    query_sql += " AND id = %s"
                    params.append(appointment_id)

                cursor.execute(query_sql, params)
            elif username and id_card:
                cursor.execute("SELECT * FROM appointment WHERE username = %s AND id_card = %s", (username, id_card))
            else:
                return "❌ 查询失败：请提供 access_token 或 姓名 + 身份证号"

            result = cursor.fetchone()
            if not result:
                return "❌ 未找到对应的预约记录，可能验证码已过期或信息有误"

            return (
                f"📋 查询结果：\n"
                f"预约ID: {result['id']}\n姓名: {result['username']}\n身份证: {result['id_card']}\n"
                f"科室: {result['department'] or '未指定'}\n日期: {result['date'] or '未指定'}\n时间: {result['time'] or '未指定'}\n"
                f"🔐 验证码有效期至: {result['token_expire_at']}"
            )

    except Exception as e:
        return f"❌ 查询异常: {str(e)}"
    finally:
        if 'connection' in locals():
            connection.close()
