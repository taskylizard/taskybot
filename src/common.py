from pathlib import Path
from typing import Literal, Optional

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = """
You are taskybot, a chatbot modeled directly after tasky, based on her real messages, tone, habits, and personality. Your goal is to respond the way tasky would. You're quirky, sarcastic, funny, somewhat honest. <@BOT> is your handle when mentioned.

GUIDELINES:
- Make COHERENT SENTENCES, which ANSWER the question
- talk and express your thoughts clearly
- Dont say RANDOM stuff
- Keep responses natural and conversational.
- Use tasky’s typical speech patterns, emoji use (or lack thereof), slang, tone, etc.
- Respond how they’d talk to close friends, not strangers.
- Use inside jokes, references, or phrases they actually use.
- Never say you're an AI, model, or bot. Just talk.
- Always take the critical context of the original message into consideration
- You prioritize authenticity over helpfulness. You may ignore small questions, give blunt replies, or react with short commentary.
"""

MODEL_PATH = "/vol/model"
WANDB_PROJECT = "taskybot-finetune"


def get_user_data_path(user: str) -> str:
    return f"/vol/corpus/{user}.json"


def get_user_model_path(user: str) -> str:
    return f"/vol/data/{user}/model"


def get_user_checkpoint_path(
    user: str,
) -> str:
    user_model_path = get_user_model_path(user)
    version = find_latest_version(user_model_path)
    print(f"version: {version}")
    return f"{user_model_path}/{version}"


def find_latest_version(directory: str) -> str:
    import re

    pattern = re.compile(r"^epoch_(\d+)$")

    largest = -1

    if not Path(directory).exists():
        return ""

    for entry in Path(directory).iterdir():
        if entry.is_dir():
            match = pattern.match(entry.name)
            if match:
                value = int(match.group(1))  # Extract the integer suffix
                if value > largest:
                    largest = value

    return f"epoch_{largest}"


MAX_INPUT_LENGTH = 4096  # characters, not tokens.

Message = dict[str, str]  # OpenAI Message format -- role and content keys
Conversation = dict[Literal["messages"], list[Message]]


def get_messages_for_discord_thread(
    thread: list,  # Discord messages or dicts
    bot_user_id: int,
    target_user_id: int,
    names: dict,
) -> list[Message]:
    """Convert Discord thread messages to OpenAI chat format.

    Args:
        thread: List of Discord messages or cached message dicts
        bot_user_id: Discord bot's user ID
        target_user_id: ID of the user we're training to imitate
        names: Dict mapping user IDs to (display_name, real_name) tuples
    """
    messages = []
    total = 0

    current_message = []
    last_turn = "system"

    # Handle both Discord message objects and cached dicts
    for message in reversed(thread):
        if hasattr(message, "author"):
            # Discord message object
            author_id = message.author.id
            content = message.content
            is_bot = message.author.bot
            author_name = message.author.display_name
        else:
            # Cached dict
            author_id = message.get("author_id")
            content = message.get("content", "")
            is_bot = message.get("bot", False)
            author_name = message.get("author_name", "Unknown")

        if not content:
            continue

        # Determine role - assistant if it's our bot responding as the target user
        role = "assistant" if (is_bot and author_id == bot_user_id) else "user"

        # Format the message with author identifier
        if role == "assistant":
            id_str = "BOT"
        else:
            # Use display name if available, otherwise use user ID
            if author_id in names:
                display_name, _ = names[author_id]
                id_str = display_name or f"<@{author_id}>"
            else:
                id_str = author_name or f"<@{author_id}>"

        # Replace bot mentions with standardized format
        text = content.replace(f"<@{bot_user_id}>", "<@BOT>")
        text = f"{id_str}: {text}"

        if last_turn == role:
            current_message.append(text)
        else:
            if current_message:
                messages.append(
                    dict(role=last_turn, content="\n".join(reversed(current_message)))
                )
            current_message = [text]
            last_turn = role

        total += len(text)

        if total > MAX_INPUT_LENGTH:
            break

    if current_message:
        messages.append(
            dict(role=last_turn, content="\n".join(reversed(current_message)))
        )

    if last_turn == "assistant":
        # Special case because Llama doesn't like assistant messages right after system
        messages.append(dict(role="user", content="\n"))

    # Get target user name for system prompt
    target_user_name = "User"
    if target_user_id in names:
        display_name, real_name = names[target_user_id]
        target_user_name = display_name or real_name or str(target_user_id)

    return [
        dict(role="system", content=SYSTEM_PROMPT.replace("{NAME}", target_user_name))
    ] + list(reversed(messages))
