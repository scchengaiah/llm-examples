import re
import html

AVATAR_URL = "https://img.icons8.com/color/48/bot.png"

def format_message(text):
    """
    This function is used to format the messages in the chatbot UI.

    Parameters:
    text (str): The text to be formatted.
    """
    text_blocks = re.split(r"```[\s\S]*?```", text)
    code_blocks = re.findall(r"```([\s\S]*?)```", text)

    text_blocks = [html.escape(block) for block in text_blocks]

    formatted_text = ""
    for i in range(len(text_blocks)):
        formatted_text += text_blocks[i].replace("\n", "<br>")
        if i < len(code_blocks):
            formatted_text += f'<pre style="white-space: pre-wrap; word-wrap: break-word;"><code>{html.escape(code_blocks[i])}</code></pre>'

    return formatted_text

def get_bot_message_container(text):
    """Generate the bot's message container style for the given text."""
    formatted_text = format_message(text)
    container_content = f"""
        <div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: flex-start;">
            <img src="{AVATAR_URL}" class="bot-avatar" alt="avatar" style="width: 30px; height: 30px;" />
            <div style="background: #71797E; color: white; border-radius: 20px; padding: 10px; margin-right: 5px; margin-left: 5px; max-width: 75%; font-size: 14px;">
                {formatted_text} \n </div>
        </div>
    """
    return container_content