import openai
import json
from typing import Optional

# 步骤 1：定义实际执行的工具函数（业务逻辑）
def record_expense(category: str, amount: float, payment_method: str) -> dict:
    """
    实际执行“记录消费”的函数，可扩展为写入数据库、生成账单等真实逻辑
    :param category: 消费类别（如餐饮、交通）
    :param amount: 消费金额（元）
    :param payment_method: 支付方式（支付宝/微信等）
    :return: 执行结果（可包含状态、消息等）
    """
    # 这里仅做示例返回，实际可扩展为写入文件/数据库等操作
    return {
        "status": "success",
        "message": f"记录成功：{category} 消费 {amount} 元（支付方式：{payment_method}）"
    }

# 步骤 2：定义 Function Calling 元信息（告诉 LLM 工具的结构）
tools = [
    {
        "type": "function",
        "function": {
            "name": "record_expense",  # 工具函数名，必须与实际函数名一致
            "description": "用于记录日常消费明细，需填写消费类别、金额、支付方式",  # 让 LLM 判断何时调用
            "parameters": {  # 定义参数规则，与函数入参对应
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "消费类别，如餐饮、交通、购物"  # 引导 LLM 填合适的值
                    },
                    "amount": {
                        "type": "number",
                        "description": "消费金额（单位：元，需为数字）"
                    },
                    "payment_method": {
                        "type": "string",
                        "enum": ["支付宝", "微信", "现金", "信用卡"],  # 限定可选值
                        "description": "支付方式，需从枚举中选择"
                    }
                },
                "required": ["category", "amount"]  # 必填参数
            }
        }
    }
]

# 步骤 3：发送请求，让 LLM 决定是否调用工具
try:
    # 构造用户提问（可替换为实际业务中的动态问题）
    user_question = "今天在星巴克喝咖啡花了52元，用支付宝付款"
    
    response = openai.chat.completions.create(
        model="gpt-4o",  # 替换为实际可用的模型
        messages=[{"role": "user", "content": user_question}],
        tools=tools,  # 传入步骤 2 定义的工具元信息
        tool_choice="auto"  # 让 LLM 自动判断是否调用工具
    )
    
    # 提取 LLM 响应（含工具调用指令或直接回答）
    message = response.choices[0].message
    if message.tool_calls:  # 如果 LLM 要求调用工具
        tool_call = message.tool_calls[0]
        print(f"LLM 决定调用工具：{tool_call.function.name}")
        print(f"调用参数：{tool_call.function.arguments}")
    else:  # LLM 直接回答，无需调用工具
        print("LLM 直接回答：")
        print(message.content)

except Exception as e:
    print(f"API 调用异常：{str(e)}")

# 步骤 4：执行工具函数 + 二次请求生成最终回答
try:
    # （承接步骤 3，判断有工具调用时执行）
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        if tool_call.function.name == "record_expense":
            # 解析 LLM 传的参数
            args = json.loads(tool_call.function.arguments)
            
            # 执行实际工具函数（步骤 1 定义的函数）
            result = record_expense(**args)
            print(f"工具执行结果：{result}")
            
            # 二次请求：把工具结果发回给 LLM，让其生成最终回答
            final_response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": user_question},  # 原始问题
                    {"role": "assistant", "tool_calls": [tool_call]},  # LLM 最初的工具调用指令
                    {
                        "role": "tool", 
                        "content": json.dumps(result),  # 工具执行结果
                        "tool_call_id": tool_call.id  # 关联工具调用
                    }
                ]
            )
            
            # 提取 LLM 结合工具结果生成的最终回答
            final_answer = final_response.choices[0].message.content
            print(f"最终回答：{final_answer}")
        else:
            print("未匹配到预期工具")
except json.JSONDecodeError:
    print("参数解析失败，请检查函数结构")
except Exception as e:
    print(f"二次请求异常：{str(e)}")
    

# 最后输出：记录成功：餐饮 52元（支付宝）。